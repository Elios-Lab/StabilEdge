"""Nonblocking Qt process transport for the optional Transformer backend."""
import json
import os
from pathlib import Path
import sys
import time
import uuid
from PyQt5.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal
from verification.core import Status
from verification.transformer_runner import atomic_json


class TransformerProcess(QObject):
    event_received = pyqtSignal(object)
    log = pyqtSignal(str)
    finished = pyqtSignal(object)
    running_changed = pyqtSignal(bool)

    def __init__(self, parent=None, runs_root=None):
        super().__init__(parent)
        self.process = None
        self.runs_root = Path(runs_root or Path.home() / ".local/share/StabilEdge/runs")
        self.run_directory = None
        self._buffer = b""
        self._terminal_status = None
        self._error = ""
        self._started_at = 0
        self._completed = []
        self._saw_finished = False
        self.watchdog = QTimer(self)
        self.watchdog.setSingleShot(True)
        self.watchdog.timeout.connect(self._timeout)

    @property
    def running(self):
        return self.process is not None

    def start(self, request, python_executable, timeout=0):
        if self.running:
            raise RuntimeError("A Transformer run is already active.")
        request.validate()
        executable = Path(python_executable).expanduser().absolute()
        if not executable.is_file() or not os.access(str(executable), os.X_OK):
            raise ValueError("Select an executable Python interpreter for the Transformer environment.")
        self.run_directory = self.runs_root / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
        self.run_directory.mkdir(parents=True)
        atomic_json(self.run_directory / "request.json", request.to_dict())
        self._buffer = b""
        self._terminal_status = None
        self._error = ""
        self._completed = []
        self._saw_finished = False
        self._started_at = time.monotonic()
        process = self.process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHOME", "QT_PLUGIN_PATH",
                     "QT_QPA_PLATFORM_PLUGIN_PATH", "CONDA_PREFIX", "QTWEBENGINE_CHROMIUM_FLAGS"):
            environment.remove(name)
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("PYTHONDONTWRITEBYTECODE", "1")
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(Path(request.repository).expanduser().resolve()))
        runner = Path(__file__).with_name("transformer_runner.py")
        process.readyReadStandardOutput.connect(self._read_output)
        process.readyReadStandardError.connect(self._read_errors)
        process.finished.connect(self._finish)
        process.errorOccurred.connect(self._process_error)
        self.running_changed.emit(True)
        process.start(str(executable), ["-u", str(runner), "--request", str(self.run_directory / "request.json"),
                                       "--output", str(self.run_directory / "results.json"), "--timeout", str(timeout)])
        if timeout:
            self.watchdog.start(min(int((timeout + 3) * 1000), 2147483647))
        QTimer.singleShot(15000, lambda: self._check_start(process))

    def _check_start(self, process):
        if self.process is process and process.state() == QProcess.Starting:
            self._error = "The Transformer Python process did not start."
            self._terminate(Status.ERROR)

    def _read_output(self):
        if self.process is None:
            return
        self._buffer += bytes(self.process.readAllStandardOutput())
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                event = json.loads(line.decode("utf-8"))
                if not isinstance(event, dict) or event.get("schema_version") != 1:
                    raise ValueError("Invalid engine event.")
                with (self.run_directory / "events.jsonl").open("a") as handle:
                    handle.write(json.dumps(event) + "\n")
                if event.get("event") == "result":
                    self._completed.append(event["result"])
                if event.get("event") == "error":
                    self._error = event.get("message", "Engine error")
                if event.get("event") == "finished":
                    self._saw_finished = True
                self.event_received.emit(event)
            except (ValueError, KeyError, UnicodeDecodeError) as exc:
                self._error = "Invalid Transformer event: %s" % exc
                self._terminate(Status.ERROR)
        if len(self._buffer) > 8 * 1024 * 1024:
            self._error = "Transformer event exceeded the transport limit."
            self._terminate(Status.ERROR)

    def _read_errors(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardError())
        with (self.run_directory / "engine.log").open("ab") as handle:
            handle.write(data)
        self.log.emit(data.decode("utf-8", errors="replace"))

    def _process_error(self, error):
        if self.process is None:
            return
        self._error = self.process.errorString()
        if error == QProcess.FailedToStart:
            self._terminal_status = Status.ERROR
            self._finish(-1, QProcess.CrashExit)

    def _timeout(self):
        self._terminate(Status.TIMEOUT)

    def cancel(self):
        self._terminate(Status.CANCELLED)

    def _terminate(self, status):
        process = self.process
        if process is None:
            return
        # The first terminal cause wins (e.g. timeout followed by Stop).
        if self._terminal_status is None:
            self._terminal_status = status
        process.terminate()
        QTimer.singleShot(2000, lambda: self._kill_if_active(process))

    def _kill_if_active(self, process):
        if self.process is process and process.state() != QProcess.NotRunning:
            process.kill()

    def _finish(self, exit_code, exit_status):
        if self.process is None:
            return
        self.watchdog.stop()
        self._read_output()
        self._read_errors()
        path = self.run_directory / "results.json"
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError):
            result = {"schema_version": 1, "engine": "transformer", "status": Status.ERROR.value,
                      "results": self._completed}
        if self._terminal_status is not None:
            result["status"] = self._terminal_status.value
        elif exit_code != 0 or exit_status == QProcess.CrashExit or self._buffer.strip() or not self._saw_finished:
            result["status"] = Status.ERROR.value
        elif result.get("status") not in {status.value for status in Status}:
            result["status"] = Status.ERROR.value
            self._error = "Engine returned an invalid final status."
        if self._error and result["status"] == Status.ERROR.value:
            result["error"] = self._error
        result["elapsed_seconds"] = time.monotonic() - self._started_at
        result["run_directory"] = str(self.run_directory)
        atomic_json(path, result)
        process, self.process = self.process, None
        process.deleteLater()
        self.running_changed.emit(False)
        self.finished.emit(result)

    def shutdown(self):
        """Bounded shutdown used only while the window is closing."""
        if self.process is not None:
            self._terminal_status = Status.CANCELLED
            process = self.process
            process.terminate()
            if not process.waitForFinished(1000):
                process.kill()
                process.waitForFinished(1000)
