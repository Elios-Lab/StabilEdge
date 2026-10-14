"""Transformer-specific configuration and result views for the shared GUI."""
import json
from pathlib import Path
import sys
from PyQt5.QtCore import QSettings, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLineEdit,
    QPushButton, QFileDialog, QPlainTextEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QLabel, QProgressBar, QTreeWidget,
    QTreeWidgetItem, QSplitter, QScrollArea, QMessageBox, QSizePolicy,
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from verification.process import TransformerProcess
from verification.transformer_request import TransformerRequest, read_examples


class TransformerPanel(QWidget):
    running_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("StabilEdge", "TransformerEngine")
        self.controller = TransformerProcess(self)
        self.last_result = None
        self._preview = False
        self._build_ui()
        self.controller.event_received.connect(self._event)
        self.controller.log.connect(lambda text: self.console.appendPlainText(text.rstrip()))
        self.controller.finished.connect(self._finished)
        self.controller.running_changed.connect(self._running_changed)

    def _path_row(self, initial, directory=True):
        field = QLineEdit(str(initial))
        button = QPushButton("Browse…")
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        layout.addWidget(button)

        def browse():
            path = (QFileDialog.getExistingDirectory(self, "Select folder", field.text()) if directory else
                    QFileDialog.getOpenFileName(self, "Select file", field.text())[0])
            if path:
                field.setText(path)
        button.clicked.connect(browse)
        return field, row

    @staticmethod
    def _float(initial, maximum=1000):
        widget = QDoubleSpinBox()
        widget.setDecimals(8)
        widget.setRange(0.00000001, maximum)
        widget.setValue(initial)
        return widget

    def _build_ui(self):
        root = Path(__file__).resolve().parents[1]
        default_repo = root.parent / "Transformer-Position-Scheduling-Certification"
        candidates = [root / ".venv-transformer/bin/python", Path.home() / "anaconda3/envs/pbverify/bin/python"]
        default_python = next((path for path in candidates if path.exists()), Path(sys.executable))
        container = QWidget()
        form = QFormLayout(container)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        intro = QLabel("Certify binary Transformer classifiers under embedding perturbations.\n"
                       "A certified result covers the displayed token tasks; an unknown result supplies no counterexample.")
        intro.setWordWrap(True)
        intro.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        form.addRow(intro)
        setup = QGroupBox("Engine setup")
        setup_form = QFormLayout(setup)
        self.repository, row = self._path_row(self.settings.value("repository", str(default_repo)))
        setup_form.addRow("Backend folder:", row)
        self.python, row = self._path_row(self.settings.value("python", str(default_python)), False)
        setup_form.addRow("Python environment:", row)
        form.addRow(setup)
        self.checkpoint, row = self._path_row("")
        form.addRow("Transformer checkpoint:", row)
        self.input_mode = QComboBox()
        self.input_mode.addItems(["Single sentence", "JSON batch"])
        form.addRow("Input:", self.input_mode)
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText("Enter the sentence to verify.")
        self.text.setMaximumHeight(100)
        form.addRow("Sentence:", self.text)
        self.label = QComboBox()
        self.label.addItems(["0", "1"])
        form.addRow("Expected class:", self.label)
        self.batch, row = self._path_row("", False)
        self.batch_row = row
        form.addRow("Batch JSON:", row)
        row.setEnabled(False)
        self.input_mode.currentIndexChanged.connect(self._input_changed)

        self.workflow = QComboBox()
        for title, key in [("Check an input radius", "check"), ("Search certified radii", "search"),
                           ("Schedule positions with validation and fallback", "schedule")]:
            self.workflow.addItem(title, key)
        form.addRow("Workflow:", self.workflow)
        self.method = QComboBox()
        self.method.addItem("Cascade: IBP → Forward → Backward", ["ibp", "forward", "backward"])
        for title, key in [("IBP", "ibp"), ("Forward", "forward"), ("Backward", "backward")]:
            self.method.addItem(title, [key])
        form.addRow("Bound method:", self.method)
        self.norm = QComboBox()
        for title, value in [("L2", "2"), ("L1", "1"), ("L∞", "inf")]:
            self.norm.addItem(title, value)
        form.addRow("Embedding norm:", self.norm)
        self.radius = self._float(0.01)
        form.addRow("Input radius / search start:", self.radius)
        self.maximum = self._float(2.0)
        form.addRow("Search ceiling:", self.maximum)
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 100)
        self.iterations.setValue(4)
        form.addRow("Refinement iterations:", self.iterations)
        self.adaptive = QCheckBox("Refine to bracket tolerance")
        self.tolerance = self._float(0.001)
        form.addRow(self.adaptive, self.tolerance)
        self.tolerance.setEnabled(False)
        self.adaptive.toggled.connect(lambda _: self._workflow_changed())
        self.tokens = QComboBox()
        self.tokens.addItem("One token at a time", 1)
        self.tokens.addItem("Two-token pairs", 2)
        form.addRow("Perturbation tasks:", self.tokens)
        self.length = QSpinBox()
        self.length.setRange(3, 512)
        self.length.setValue(32)
        form.addRow("Maximum token length:", self.length)
        self.positions = QLineEdit()
        self.positions.setPlaceholderText("All eligible positions, or indices such as 1,3,5")
        self.positions.setToolTip("Indices refer to checkpoint tokenization including [CLS] at index 0. Preview tokens first.")
        form.addRow("Token positions:", self.positions)
        self.budget = QSpinBox()
        self.budget.setRange(1, 10000)
        self.budget.setValue(1)
        form.addRow("Full-search budget:", self.budget)
        self.ranker = QComboBox()
        for title, value in [("Gradient magnitude", "gradient"), ("Left to right", "left_to_right"), ("Random", "random")]:
            self.ranker.addItem(title, value)
        form.addRow("Position ranking:", self.ranker)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        form.addRow("Random seed:", self.seed)
        self.device = QLineEdit("cpu")
        self.device.setPlaceholderText("cpu or cuda:0")
        form.addRow("Compute device:", self.device)
        self.timeout = QSpinBox()
        self.timeout.setRange(0, 86400)
        self.timeout.setSuffix(" s (0 = unlimited)")
        form.addRow("Run deadline:", self.timeout)
        self.workflow.currentIndexChanged.connect(self._workflow_changed)
        self._workflow_changed()
        for combo in (self.input_mode, self.workflow, self.method, self.norm, self.tokens, self.ranker):
            combo.setMinimumContentsLength(16)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setToolTip(combo.currentText())
            combo.currentTextChanged.connect(combo.setToolTip)
        self.configuration = container
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)

        results = QWidget()
        layout = QVBoxLayout(results)
        buttons = QHBoxLayout()
        self.preview = QPushButton("Preview tokens")
        self.run = QPushButton("Run certification")
        self.stop = QPushButton("Stop")
        self.stop.setEnabled(False)
        self.save = QPushButton("Save config…")
        self.load = QPushButton("Load config…")
        self.export = QPushButton("Export results…")
        self.export.setEnabled(False)
        for widget in (self.preview, self.run, self.stop, self.save, self.load, self.export):
            buttons.addWidget(widget)
        layout.addLayout(buttons)
        self.status = QLabel("Choose a compatible checkpoint and labelled input.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.token_tree = QTreeWidget()
        self.token_tree.setHeaderLabels(["Index", "Token", "Eligible"])
        self.token_tree.setMaximumHeight(180)
        layout.addWidget(self.token_tree)
        self.results = QTreeWidget()
        self.results.setHeaderLabels(["Sample / task", "Status", "Certified radius", "Details"])
        layout.addWidget(self.results)
        self.figure = Figure(figsize=(5, 2))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMaximumHeight(220)
        layout.addWidget(self.canvas)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(3000)
        layout.addWidget(self.console)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(results)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        outer = QVBoxLayout(self)
        outer.addWidget(splitter)
        self.preview.clicked.connect(lambda: self._start(True))
        self.run.clicked.connect(lambda: self._start(False))
        self.stop.clicked.connect(self.controller.cancel)
        self.save.clicked.connect(self._save_config)
        self.load.clicked.connect(self._load_config)
        self.export.clicked.connect(self._export_results)

    def _input_changed(self, index):
        self.text.setEnabled(index == 0)
        self.label.setEnabled(index == 0)
        self.batch_row.setEnabled(index == 1)

    def _workflow_changed(self):
        searching = self.workflow.currentData() in ("search", "schedule")
        scheduling = self.workflow.currentData() == "schedule"
        for widget in (self.maximum, self.iterations, self.adaptive):
            widget.setEnabled(searching)
        self.tolerance.setEnabled(searching and self.adaptive.isChecked())
        self.iterations.setEnabled(searching and not self.adaptive.isChecked())
        self.ranker.setEnabled(scheduling)
        self.budget.setEnabled(scheduling)

    def request(self, preview=False):
        examples = (read_examples(self.batch.text()) if self.input_mode.currentIndex() == 1 else
                    [{"id": getattr(self, "_single_id", "0"), "text": self.text.toPlainText().strip(), "label": int(self.label.currentText())}])
        positions = [int(part.strip()) for part in self.positions.text().split(",") if part.strip()]
        return TransformerRequest(
            repository=self.repository.text().strip(), checkpoint=self.checkpoint.text().strip(), examples=examples,
            action="preview" if preview else self.workflow.currentData(), methods=self.method.currentData(),
            norm=self.norm.currentData(), radius=self.radius.value(),
            maximum_radius=max(self.radius.value(), self.maximum.value()) if self.workflow.currentData() == "check" else self.maximum.value(),
            iterations=self.iterations.value(), tolerance=self.tolerance.value() if self.adaptive.isChecked() else None,
            max_length=self.length.value(), perturbed_tokens=self.tokens.currentData(), positions=positions,
            budget=self.budget.value(), ranker=self.ranker.currentData(), device=self.device.text().strip(), seed=self.seed.value())

    def _start(self, preview):
        try:
            request = self.request(preview)
            request.validate()
            self._preview = preview
            self.results.clear()
            self.token_tree.clear()
            self.figure.clear()
            self.canvas.draw_idle()
            self.last_result = None
            self.export.setEnabled(False)
            self.progress.setRange(0, 0)
            self.status.setText("Loading the Transformer checkpoint…")
            self.settings.setValue("repository", request.repository)
            self.settings.setValue("python", self.python.text().strip())
            self.controller.start(request, self.python.text().strip(), self.timeout.value())
        except (OSError, ValueError, RuntimeError) as exc:
            self.status.setText(str(exc))
            QMessageBox.warning(self, "Transformer setup", str(exc))

    def _running_changed(self, running):
        self.configuration.setEnabled(not running)
        for widget in (self.run, self.preview, self.save, self.load):
            widget.setEnabled(not running)
        self.stop.setEnabled(running)
        self.running_changed.emit(running)

    def _event(self, event):
        kind = event.get("event")
        if kind in ("loading", "error"):
            self.status.setText(event.get("message", ""))
        elif kind == "sample_started":
            self._show_tokens(event)
            self.status.setText("Sample %s: %s perturbation tasks" % (event["sample_id"], len(event["tasks"])))
        elif kind == "task_finished":
            self.status.setText("Sample %s · %s · token positions %s" % (event["sample_id"], event["stage"], event["positions"]))
        elif kind == "result":
            self.progress.setRange(0, event["total"])
            self.progress.setValue(event["completed"])
            self._show_result(event["result"])

    def _show_tokens(self, details):
        self.token_tree.clear()
        eligible = details.get("valid_positions", [])
        for index, token in enumerate(details.get("tokens", [])):
            QTreeWidgetItem(self.token_tree, [str(index), token, "Yes" if index in eligible else "No"])

    def _show_result(self, result):
        details = result.get("details", {})
        self._show_tokens(details)
        radius = result.get("radius")
        row = QTreeWidgetItem(self.results, [result["sample_id"], result["status"],
                                           "%.8g" % radius if radius is not None else "", result.get("reason", "")])
        for task, bound in details.get("bounds", {}).items():
            QTreeWidgetItem(row, ["Tokens " + task, "CERTIFIED" if bound["certified"] else "UNKNOWN",
                                 "%.8g" % bound["radius"] if bound["radius"] is not None else "",
                                 "No certified endpoint" if not bound["certified"] else
                                 "Search ceiling reached" if bound.get("at_cap") else "Certified lower endpoint"])
        self.results.resizeColumnToContents(0)

    def _finished(self, result):
        self.last_result = result
        self.export.setEnabled(True)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
        self.status.setText("%s · %.1f seconds%s" % (result["status"], result["elapsed_seconds"],
                                                     " · " + result["error"] if result.get("error") else ""))
        self.console.appendPlainText("Run saved to " + result["run_directory"])
        certified = [row for row in result.get("results", []) if row["status"] == "CERTIFIED"]
        if certified and not self._preview:
            axis = self.figure.add_subplot(111)
            axis.bar([row["sample_id"] for row in certified], [row["radius"] for row in certified])
            axis.set(xlabel="Sample", ylabel="Certified radius", title="Certified lower bounds")
            self.figure.tight_layout()
            self.canvas.draw_idle()

    def _save_config(self):
        try:
            payload = {"schema_version": 1, "engine": "transformer", "request": self.request().to_dict(),
                       "python_executable": self.python.text(), "timeout": self.timeout.value()}
            path = QFileDialog.getSaveFileName(self, "Save Transformer configuration", "", "JSON (*.json)")[0]
            if path:
                Path(path).write_text(json.dumps(payload, indent=2))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Configuration", str(exc))

    def _load_config(self):
        path = QFileDialog.getOpenFileName(self, "Load Transformer configuration", "", "JSON (*.json)")[0]
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text())
            if payload.get("schema_version") != 1 or payload.get("engine") != "transformer":
                raise ValueError("Choose a StabilEdge Transformer configuration.")
            request = TransformerRequest.from_dict(payload["request"], validate_paths=False)
            self.repository.setText(request.repository)
            self.python.setText(payload.get("python_executable", self.python.text()))
            self.checkpoint.setText(request.checkpoint)
            if len(request.examples) == 1:
                self.input_mode.setCurrentIndex(0)
                self.text.setPlainText(request.examples[0]["text"])
                self.label.setCurrentText(str(request.examples[0]["label"]))
                self._single_id = str(request.examples[0].get("id", "0"))
            else:
                # The batch reader also accepts examples embedded in saved configs.
                self.batch.setText(str(Path(path).resolve()))
                self.input_mode.setCurrentIndex(1)
            for widget, value in ((self.workflow, request.action), (self.norm, request.norm),
                                  (self.tokens, request.perturbed_tokens), (self.ranker, request.ranker)):
                widget.setCurrentIndex(widget.findData(value))
            for index in range(self.method.count()):
                if self.method.itemData(index) == request.methods:
                    self.method.setCurrentIndex(index)
                    break
            else:
                self.method.addItem("Custom cascade: " + " → ".join(request.methods), request.methods)
                self.method.setCurrentIndex(self.method.count() - 1)
            self.radius.setValue(request.radius)
            self.maximum.setValue(request.maximum_radius)
            self.iterations.setValue(request.iterations)
            self.adaptive.setChecked(request.tolerance is not None)
            if request.tolerance is not None:
                self.tolerance.setValue(request.tolerance)
            self.length.setValue(request.max_length)
            self.positions.setText(",".join(map(str, request.positions)))
            self.budget.setValue(request.budget)
            self.seed.setValue(request.seed)
            self.device.setText(request.device)
            self.timeout.setValue(payload.get("timeout", 0))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "Configuration", str(exc))

    def _export_results(self):
        path = QFileDialog.getSaveFileName(self, "Export Transformer results", "", "JSON (*.json)")[0]
        if path and self.last_result is not None:
            try:
                Path(path).write_text(json.dumps(self.last_result, indent=2))
            except OSError as exc:
                QMessageBox.warning(self, "Export", str(exc))
