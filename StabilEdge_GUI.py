#!/usr/bin/env python3
import os
import sys
import gc
import time
from threading import Thread


if __name__ == "__main__" and sys.argv[0].endswith("run_gui.sh"):
    os.execvp(sys.executable, [sys.executable] + sys.argv[1:])

# ─── 1) High-DPI ENVIRONMENT (must come before any Qt import) ────────────────
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"]   = "1"

# ─── 1a) Fix Qt plugin path (before any PyQt5 import) ───────────────────────
from PyQt5.QtCore import QLibraryInfo
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
    QLibraryInfo.location(QLibraryInfo.PluginsPath), "platforms"
)

# ─── 2) Qt Attributes for High-DPI (before any other Qt imports) ───────────
from PyQt5.QtCore import QCoreApplication, Qt
QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

QCoreApplication.setAttribute(Qt.AA_UseOpenGLES)

# ─── 3) Standard imports ───────────────────────────────────────────────────
import json
import tempfile
import glob
import collections
import subprocess
import onnx
import socket
import traceback
import io, contextlib
import re
from PIL import Image
from typing import Optional, List, Tuple, Set, Deque, Dict, Any, Callable

from PyQt5.QtCore   import pyqtSignal, QThread, QUrl, QProcess, pyqtSlot
from PyQt5.QtGui    import QIcon, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QComboBox,
    QCheckBox, QTreeWidget, QTreeWidgetItem,
    QMessageBox, QFileDialog, QPlainTextEdit, QSplitter,
    QSpinBox, QDoubleSpinBox, QTabWidget, QGridLayout, QMenu, QAction, QToolButton, QWidgetAction,
    QSizePolicy, QProgressBar
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QTimer
import matplotlib
matplotlib.use("Qt5Agg") 
#matplotlib.use("Agg")
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar
)
import pyqtgraph as pg  # Import PyQtGraph for plotting

# Chromium flags (if needed)
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
    "--disable-gpu "
    "--disable-software-rasterizer "
    "--disable-extensions "
    "--disable-plugins "
    "--disable-translate "
    "--disable-sync "
    "--disable-background-timer-throttling "
    "--disable-renderer-backgrounding "
    "--disable-dev-shm-usage "
    "--no-sandbox "
    "--disable-web-security "
    "--force-dark-mode --enable-features=WebContentsForceDark "
    "--log-level=3"
)


import netron
from ITD import local_rob_sens, targeted_rob_sens,plot_aggregated_itd, plot_grouped_confusion
from TOF import run_tof
from Jena_gui import run_jena


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_MAP = {
    "ITD":  os.path.join(BASE_DIR, "ITD_features.json"),
    "TOF":  os.path.join(BASE_DIR, "TOF_features.json"),
    "Jena": os.path.join(BASE_DIR, "Jena_features.json"),
}

3

# ── Enhanced Regex patterns for ALL Marabou statistics ──────────────────────

RX_ACTIVE_PL         = re.compile(r"Number of active piecewise-linear constraints:\s+(\d+)\s*/\s*(\d+)")
RX_SMT_DEPTH         = re.compile(r"Total depth is\s+(\d+)")
RX_NUM_SPLITS        = re.compile(r"Number of splits:\s+(\d+)")
RX_TOTAL_TIME        = re.compile(r"Total time elapsed:\s+(\d+)\s+milli")
RX_MAIN_LOOP_TIME    = re.compile(r"Main loop:\s+(\d+)\s+milli")
RX_MAIN_LOOP_ITERATIONS = re.compile(r"Number of main loop iterations:\s+(\d+)")
RX_SIMPLEX_PIVOTS    = re.compile(r"Total number of pivots performed:\s+(\d+)")
RX_DEGENERATE_PIVOTS = re.compile(r"Degenerate:\s+(\d+)")
RX_AVG_PIVOT_TIME    = re.compile(r"Average time per pivot:\s+([0-9.]+)\s+milli")
RX_LAST_DEGRADATION  = re.compile(r"Last reported degradation:\s+([0-9.eE+-]+)")
RX_TIGHTENED_BOUNDS  = re.compile(r"Number of tightened bounds:\s+(\d+)")
RX_BASIS_REFACTOR    = re.compile(r"Number of basis refactorizations:\s+(\d+)")
RX_PRECISION_RESTORATION = re.compile(r"Performing precision restoration")

# ancillary patterns
RX_TIMESTAMP      = re.compile(r"(\d{2}:\d{2}:\d{2})\s+Statistics update:")
RX_INITIAL_STATS  = re.compile(r"Engine::solve: Initial statistics")
RX_BLOCK_START    = re.compile(r"Statistics update:|Engine::solve: Initial statistics")
RX_BLOCK_END      = re.compile(r"^---|^Performing precision restoration|^$")


# ── Comprehensive Statistics Container with ALL metrics ──────────────────────
ComprehensiveStats =  collections.namedtuple("KeyStats", [
    "timestamp",
    "active_pl_current", "active_pl_total",
    "smt_depth", "num_splits",
    "total_time_ms", "main_loop_time_ms", "main_loop_iterations",
    "simplex_pivots", "degenerate_pivots", "avg_pivot_time_ms",
    "last_degradation",
    "tightened_bounds",
    "basis_refactorizations",
    "precision_restorations"   # running count of restoration events
])

# ── Enhanced thresholds with more sophisticated criteria ──────────────────────
ENHANCED_THRESHOLDS = {
    "stats_analysis_delay": 30,
    # active PL constraints
    "active_pl_absolute": 10,
    "active_pl_ratio":    0.25,   # stop when ≤ 25 % still active
    # SMT core depth
    "max_smt_depth": 5,
    # bound-tightening progress
    "min_tightening_rate": 0.01,  # tightenings per second
    "stagnation_seconds": 180,
    # numerical stability
    "max_degradation": 1e-6,
    # pivot health
    "max_degenerate_ratio": 0.01,
    "pivot_time_plateau_ms": 1.0,
}

def is_close_enhanced(stats):
    """Enhanced closeness detection using multiple sophisticated metrics"""
    active, total = stats.active_pl
    depth = stats.smt_depth
    
    # 1. Original depth + active PL check
    depth_close = (depth >= ENHANCED_THRESHOLDS["max_smt_depth"] and
                   (active <= ENHANCED_THRESHOLDS["active_pl_absolute"] or
                    (total and active / total <= ENHANCED_THRESHOLDS["active_pl_ratio"])))
    
    # 2. Search tree exploration metrics
    exploration_close = (stats.visited_states > 1000 and 
                        stats.num_splits > 500 and 
                        stats.max_stack_depth > 10)
    
    # 3. Solver efficiency metrics (high pivot count might indicate difficulty)
    solver_struggling = (stats.simplex_pivots > 10000 and 
                        stats.degenerate_pivots > stats.real_pivots * 0.5)
    
    # 4. Tableau size indicators (large tableau suggests complex problem)
    tableau_large = (stats.tableau_m > 5000 and stats.tableau_n > 10000)
    
    return depth_close or exploration_close or solver_struggling or tableau_large

def estimate_remaining_time(history: List[ComprehensiveStats]) -> Optional[float]:
    """Time estimation for irregular Marabou updates"""
    if len(history) < 2:
        return None
    
    try:
        recent_stats = history[-min(3, len(history)):]  # Use last 3 snapshots
        
        if len(recent_stats) < 2:
            return None
        
        # Calculate progress rate
        wall_times = [getattr(stat, 'wall_time', 0) for stat in recent_stats]
        active_counts = [stat.active_pl_current for stat in recent_stats]
        
        total_time_span = wall_times[-1] - wall_times[0]
        total_progress = active_counts[0] - active_counts[-1]
        
        if total_time_span <= 0 or total_progress <= 0:
            return None
        
        progress_rate = total_progress / total_time_span
        current_active = active_counts[-1]
        
        if progress_rate > 0:
            estimated_seconds = current_active / progress_rate
            return min(estimated_seconds, 2 * 3600)  # Cap at 2 hours
        
        return None
        
    except Exception:
        return None

class FDRedirector:
    def __init__(self, sig):
        self.sig = sig
        self.old_stdout = None
        self.old_stderr = None

    def start(self):
        # Store original stdout/stderr
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        
        class TeeWriter:
            def __init__(self, signal, original_stream):
                self.signal = signal
                self.original_stream = original_stream
                self.buffer = ""
                
            def write(self, text):
                if text:  # Only process non-empty text
                    # Always write to original stream first
                    self.original_stream.write(text)
                    self.original_stream.flush()
                    
                    # Add text to buffer
                    self.buffer += text
                    
                    # Process complete lines
                    lines = self.buffer.split('\n')
                    # Keep last incomplete line in buffer
                    self.buffer = lines[-1]
                    
                    # Emit all complete lines (except the last incomplete one)
                    for line in lines[:-1]:
                        if line.strip():  # Only emit non-empty lines
                            self.signal.emit(line)
                    
                    # Also emit partial lines that end with carriage return (for progress updates)
                    if '\r' in self.buffer:
                        parts = self.buffer.split('\r')
                        for part in parts[:-1]:
                            if part.strip():
                                self.signal.emit(part)
                        self.buffer = parts[-1]
                    
            def flush(self):
                self.original_stream.flush()
                # Emit any remaining buffer content
                if self.buffer.strip():
                    self.signal.emit(self.buffer)
                    self.buffer = ""
        
        # Replace stdout/stderr with TeeWriter that writes to both destinations
        sys.stdout = TeeWriter(self.sig, self.old_stdout)
        sys.stderr = TeeWriter(self.sig, self.old_stderr)

    def stop(self):
        # Flush any remaining buffer before restoring
        if hasattr(sys.stdout, 'flush'):
            sys.stdout.flush()
        if hasattr(sys.stderr, 'flush'):
            sys.stderr.flush()
            
        # Restore original stdout/stderr
        if self.old_stdout:
            sys.stdout = self.old_stdout
        if self.old_stderr:
            sys.stderr = self.old_stderr



class MultiWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(float, int, int, int, int, int, float)
    jena_heatmap = pyqtSignal(object)
    timeout_dialog_requested = pyqtSignal(object, float, float, float)
    
    def __init__(
        self,
        pipeline: str,
        vt_mode: str,
        combos: List[Tuple[float, Optional[float], int]],
        perturb_all: bool,
        model_file: str,
        input_path: str,
        json_path: str,
        selected_idxs: List[int],
        plot: bool,
        verbosity: int,
        feature_cfg: dict,
        selected_classes: Optional[List[int]],
        suggestion_threshold: float = 0.8,
        progress_callback: Optional[Callable[[float, int, int, int, int, int], None]] = None,
        *args,
        **kwargs,
        
    ):
        super().__init__(*args, **kwargs)
        self.pipeline         = pipeline
        self.vt_mode          = vt_mode
        self.combos           = combos
        self.perturb_all      = perturb_all
        self.model_file       = model_file
        self.input_path       = input_path
        self.json_path        = json_path
        self.selected_idxs    = selected_idxs
        self.plot             = plot
        self.verbosity        = verbosity
        self.feature_cfg      = feature_cfg
        self.selected_classes = selected_classes
        self._stop_requested  = False
        self.suggestion_threshold = suggestion_threshold
        
        # Essential timeout handling
        self._current_timeout = 0
        self._run_start_time = 0
        self._current_delta = 0
        self._marabou_start_time = 0  
        
        # Initialize comprehensive statistics tracking
        self._comprehensive_history = []
        self._latest_comprehensive_stats = {}
        self._initial_active = None
        self._suggested_timeout = False
        self._stagnation_start = None
        self._last_progress_time = None
        self._stats_analysis_enabled = False  # 
        
        # Add response handling for timeout dialog
        self._timeout_dialog_response = None
        self._timeout_dialog_event = None

        # FIX: Remove the duplicate _progress_cb definition and make one clean version
        self._progress_cb = self._emit_progress_signal
    
    def _emit_progress_signal(self, done, total, sat, unsat, timeout):
        """Emit progress signal with elapsed time in a thread-safe way"""
        current_time = time.time()
        elapsed = current_time - self._run_start_time if self._run_start_time else 0
        # Simply emit the signal directly - PyQt handles thread safety for signals
        self.progress.emit(self._current_delta, done, total, sat, unsat, timeout, elapsed)
    
    @pyqtSlot(int, int, int, int, int)
    def _do_emit_progress(self, done, total, sat, unsat, timeout):
        """Actually emit the progress signal"""
        self.progress.emit(self._current_delta, done, total, sat, unsat, timeout)
    
    
    def _on_stdout(self, proc: QProcess, delta: float):
        # read whatever's come in
        raw = proc.readAllStandardOutput()
        text = bytes(raw).decode('utf-8', 'ignore')
        for line in text.splitlines():
            # Process line for statistics if verbosity >= 2
            if self.verbosity >= 2:
                self._collect_and_maybe_warn(line, delta)
            
            # Always emit the line to GUI console
            self.log.emit(line)

    def _collect_and_maybe_warn(self, line: str, delta: float):
        """Comprehensive statistics collection with ALL Marabou metrics"""
        
        # NEW: Start the delay timer as soon as we see ANY Marabou output
        if self._marabou_start_time == 0:
            # Detect Marabou starting by looking for typical startup patterns
            marabou_indicators = [
                "Marabou", "Starting engine", "Input query", "Number of variables:",
                "Number of equations:", "Engine::solve", "Starting resolution",
                "Preprocessing", "Starting main solving loop"
            ]
            
            if any(indicator in line for indicator in marabou_indicators):
                self._marabou_start_time = time.time()
                self.log.emit(f"[STATS] Marabou detected - analysis will start in {ENHANCED_THRESHOLDS['stats_analysis_delay']}s")
        
        # Check if enough time has passed to enable statistics analysis
        if not self._stats_analysis_enabled and self._marabou_start_time > 0:
            elapsed = time.time() - self._marabou_start_time
            if elapsed >= ENHANCED_THRESHOLDS["stats_analysis_delay"]:
                self._stats_analysis_enabled = True
                self.log.emit(f"[STATS] Analysis enabled after {elapsed:.1f}s delay from Marabou start")
            else:
                # Still in delay period - just pass through the raw line without analysis
                return
        
        # If Marabou hasn't been detected yet, don't do any statistics analysis
        if self._marabou_start_time == 0:
            return
        
        # Check for timestamp in statistics update first
        timestamp_match = RX_TIMESTAMP.search(line)
        if timestamp_match:
            self._current_timestamp = timestamp_match.group(1)
            self._latest_comprehensive_stats.clear()
            self._stats_block_active = True
            if self._stats_analysis_enabled:  # Only log if analysis is enabled
                self.log.emit(f"[STATS] Started stats block at {self._current_timestamp}")
            return
        
        # Check for initial statistics (no timestamp)
        if RX_INITIAL_STATS.search(line):
            self._current_timestamp = time.strftime("%H:%M:%S")
            self._latest_comprehensive_stats.clear()
            self._stats_block_active = True
            if self._stats_analysis_enabled:  # Only log if analysis is enabled
                self.log.emit(f"[STATS] Started initial stats block")
            return
        
        # End of stats block?
        if RX_BLOCK_END.search(line) and hasattr(self, '_stats_block_active') and self._stats_block_active:
            self._stats_block_active = False
            
            # Create comprehensive snapshot if we have key metrics AND analysis is enabled
            if 'active_pl' in self._latest_comprehensive_stats and self._stats_analysis_enabled:
                self.log.emit(f"[STATS] Creating snapshot with {len(self._latest_comprehensive_stats)} metrics")
                self._create_comprehensive_snapshot(delta)
            elif not self._stats_analysis_enabled:
                # Clear stats but don't analyze during delay period
                self._latest_comprehensive_stats.clear()
            return
        
        # Only parse if we're in a stats block
        if not hasattr(self, '_stats_block_active') or not self._stats_block_active:
            return
    
        patterns = [
            (RX_ACTIVE_PL, lambda m: ('active_pl', (int(m.group(1)), int(m.group(2))))),
            (RX_SMT_DEPTH, lambda m: ('smt_depth', int(m.group(1)))),
            (RX_NUM_SPLITS, lambda m: ('num_splits', int(m.group(1)))),
            (RX_TOTAL_TIME, lambda m: ('total_time_ms', int(m.group(1)))),
            (RX_MAIN_LOOP_TIME, lambda m: ('main_loop_time_ms', int(m.group(1)))),
            (RX_MAIN_LOOP_ITERATIONS, lambda m: ('main_loop_iterations', int(m.group(1)))),
            (RX_SIMPLEX_PIVOTS, lambda m: ('simplex_pivots', int(m.group(1)))),
            (RX_DEGENERATE_PIVOTS, lambda m: ('degenerate_pivots', int(m.group(1)))),
            (RX_AVG_PIVOT_TIME, lambda m: ('avg_pivot_time_ms', float(m.group(1)))),
            (RX_LAST_DEGRADATION, lambda m: ('last_degradation', float(m.group(1)))),
            (RX_TIGHTENED_BOUNDS, lambda m: ('tightened_bounds', int(m.group(1)))),
            (RX_BASIS_REFACTOR, lambda m: ('basis_refactorizations', int(m.group(1)))),
            (RX_PRECISION_RESTORATION, lambda m: ('precision_restorations', 1))
        ]
        
        for pattern, extractor in patterns:
            match = pattern.search(line)
            if match:
                key, value = extractor(match)
                if key == 'tableau_dims':
                    self._latest_comprehensive_stats['tableau_m'] = value[0]
                    self._latest_comprehensive_stats['tableau_n'] = value[1]
                else:
                    self._latest_comprehensive_stats[key] = value
                break

    def _create_comprehensive_snapshot(self, delta: float):
        """Create a comprehensive statistics snapshot with better timing handling"""
        try:
            current_wall_time = time.time()
            timestamp = getattr(self, '_current_timestamp', time.strftime("%H:%M:%S"))
            
            # **FIX**: Create namedtuple with actual data
            active_pl = self._latest_comprehensive_stats.get('active_pl', (0, 0))
            
            snap = ComprehensiveStats(
                timestamp=timestamp,
                active_pl_current=active_pl[0],
                active_pl_total=active_pl[1],
                smt_depth=self._latest_comprehensive_stats.get('smt_depth', 0),
                num_splits=self._latest_comprehensive_stats.get('num_splits', 0),
                total_time_ms=self._latest_comprehensive_stats.get('total_time_ms', 0),
                main_loop_time_ms=self._latest_comprehensive_stats.get('main_loop_time_ms', 0),
                main_loop_iterations=self._latest_comprehensive_stats.get('main_loop_iterations', 0),
                simplex_pivots=self._latest_comprehensive_stats.get('simplex_pivots', 0),
                degenerate_pivots=self._latest_comprehensive_stats.get('degenerate_pivots', 0),
                avg_pivot_time_ms=self._latest_comprehensive_stats.get('avg_pivot_time_ms', 0.0),
                last_degradation=self._latest_comprehensive_stats.get('last_degradation', 0.0),
                tightened_bounds=self._latest_comprehensive_stats.get('tightened_bounds', 0),
                basis_refactorizations=self._latest_comprehensive_stats.get('basis_refactorizations', 0),
                precision_restorations=self._latest_comprehensive_stats.get('precision_restorations', 0)
            )
            
            # Add wall_time as attribute for time comparison
            snap.wall_time = current_wall_time
            
            # Skip duplicates
            if self._comprehensive_history:
                last_snap = self._comprehensive_history[-1]
                time_diff = current_wall_time - getattr(last_snap, 'wall_time', 0)
                if time_diff < 5:  # Skip if less than 5 seconds apart
                    self._latest_comprehensive_stats.clear()
                    return
            
            self._comprehensive_history.append(snap)
            
            # Keep only last 3 snapshots for memory efficiency
            if len(self._comprehensive_history) > 3:
                self._comprehensive_history = self._comprehensive_history[-3:]
            
            self._latest_comprehensive_stats.clear()
            
            # Analysis
            if len(self._comprehensive_history) % 2 == 0:
                self._log_comprehensive_stats(snap, delta)
            
            self._analyze_progress_and_suggest_timeout(snap, delta)
            
        except Exception as e:
            self.log.emit(f"[ERROR] Failed to create statistics snapshot: {e}")
            self._latest_comprehensive_stats.clear()

    def _log_comprehensive_stats(self, snap: ComprehensiveStats, delta: float):
        """Log detailed solver statistics"""
        active = snap.active_pl_current
        total = snap.active_pl_total
        completion_pct = ((total - active) / total * 100) if total > 0 else 0
        
        stats_msg = (
            f"[{snap.timestamp}]  Stats (δ={delta}):\n"
            f"    Progress: {completion_pct:.1f}% ({active}/{total} active constraints)\n"
            f"    SMT Depth: {snap.smt_depth}, Splits: {snap.num_splits}\n"
            f"    Iterations: {snap.main_loop_iterations}, Pivots: {snap.simplex_pivots}\n"
            f"    Time: {snap.total_time_ms}ms total, {snap.main_loop_time_ms}ms main loop"
        )
        self.log.emit(stats_msg)

    def _analyze_progress_and_suggest_timeout(self, snap: ComprehensiveStats, delta: float):
        """Enhanced progress analysis with timeout suggestions"""
        current_time = time.time()
        active_count = snap.active_pl_current
        
        # Check for progress stagnation
        if len(self._comprehensive_history) >= 2:
            prev_snap = self._comprehensive_history[-2]
            
            # Check if solver is making progress
            pl_stagnant = (prev_snap.active_pl_current == active_count)
            splits_stagnant = (prev_snap.num_splits == snap.num_splits)
            
            if pl_stagnant and splits_stagnant:
                if self._stagnation_start is None:
                    self._stagnation_start = current_time
                elif current_time - self._stagnation_start > ENHANCED_THRESHOLDS["stagnation_seconds"]:
                    self.log.emit(f"[{snap.timestamp}] 🚨 Solver stagnation detected: "
                                f"No progress in {int(current_time - self._stagnation_start)}s")
            else:
                self._stagnation_start = None
        
        # Timeout analysis
        if (self._current_timeout > 0 and not self._suggested_timeout):
            elapsed = current_time - self._run_start_time
            time_left = self._current_timeout - elapsed
            
            if elapsed >= self.suggestion_threshold * self._current_timeout:
                estimated_remaining = estimate_remaining_time(self._comprehensive_history)
                
                if estimated_remaining is not None:
                    self._show_enhanced_timeout_dialog(snap, estimated_remaining, time_left, delta)
                    self._suggested_timeout = True
                else:
                    self.log.emit(f"[{snap.timestamp}] ⏳ Timeout in ~{int(time_left)}s")

    def _show_enhanced_timeout_dialog(self, snap: ComprehensiveStats, estimated_remaining: float, time_left: float, delta: float):
        """Request timeout dialog from main thread"""
        import threading
        
        # Create an event to wait for the response
        self._timeout_dialog_event = threading.Event()
        self._timeout_dialog_response = None
        
        # Emit signal to main thread
        self.timeout_dialog_requested.emit(snap, estimated_remaining, time_left, delta)
        
        # Wait for response (with timeout to avoid hanging)
        if self._timeout_dialog_event.wait(timeout=30):  # 30 second timeout
            response = self._timeout_dialog_response
        else:
            response = False  # Default to "No" if timeout
        
        # Calculate extension time based on the original logic
        if estimated_remaining > time_left:
            extension_time = int(estimated_remaining - time_left + 120)
        else:
            extension_time = 300  # Default 5 minutes
        
        if response:
            self._current_timeout += extension_time
            self.log.emit(f"[{snap.timestamp}] ✅ Timeout extended by {extension_time}s. "
                        f"New limit: {int(self._current_timeout)}s")
        else:
            self.log.emit(f"[{snap.timestamp}] ⏭️ Continuing with current timeout limit")
    
    def set_timeout_dialog_response(self, response):
        """Called from main thread to set the response"""
        self._timeout_dialog_response = response
        if self._timeout_dialog_event:
            self._timeout_dialog_event.set()


    def run(self):
        old_py_out, old_py_err = sys.stdout, sys.stderr
        fd_redir = FDRedirector(self.log)
        fd_redir.start()
        
        sys.stdout.flush()
        
        old_py_out.flush()



        pipeline = self.pipeline
        # collect per-pipeline results
        tof_results = []       # List[Tuple[delta, sat, unsat, timeout]]
        itd_results = []       # List[Tuple[delta, sat, unsat, timeout]]
        jena_results = []      # List[Tuple[eps, delta, sat, unsat, timeout]]
        confusion_data = {}    # delta → num_classes×num_classes % matrix

        # build base_cfg once
        base_cfg = dict(self.feature_cfg)
        base_cfg.setdefault("file_paths", {})["onnx_model"] = self.model_file
        if pipeline in ("ITD", "TOF"):
            base_cfg["file_paths"]["image_directory"] = self.input_path
        else:
            base_cfg["input_path"] = self.input_path

        # set time out for Marabou
        mopts = base_cfg.setdefault("marabou", {})
        mopts["verbosity"]       = self.verbosity  


        # Log the δ→timeout mapping so the user sees it up front
        # (and call out any zeros)
        mapping = {d: t for d, _, t in self.combos}
        for d, t in mapping.items():
            if t == 0:
                self.log.emit(f"→ δ={d}: timeout=0 (no time-limit)")
            else:
                self.log.emit(f"→ δ={d}: timeout={t}s")

        if self.isInterruptionRequested():
            return

        try:
            for idx, (delta, eps, timeout) in enumerate(self.combos, start=1):
                mopts["timeoutInSeconds"] = timeout
                self.log.emit(
                    f"\n=== Run {idx}/{len(self.combos)}: "
                    f"δ={delta}, ε={eps}, timeout={timeout}s ==="
                )

                # Reset comprehensive statistics for each run - THIS WAS MISSING!
                self._current_timeout = timeout
                self._run_start_time = time.time()
                self._current_delta = delta

                # THESE LINES WERE MISSING - causing the crash:
                self._comprehensive_history.clear()
                self._latest_comprehensive_stats.clear()
                self._initial_active = None
                self._suggested_timeout = False
                self._stagnation_start = None
                self._last_progress_time = None
                self._stats_analysis_enabled = False  # NEW: Reset analysis flag for each run
                self._marabou_start_time = 0  # NEW: Reset Marabou start timer for each run

                # bail as soon as Stop is hit
                if self.isInterruptionRequested():
                    self.log.emit("=== Stop requested, skipping final aggregation ===")
                    return
                


                # ── TOF pipeline ─────────────────────────
                if pipeline == "TOF":
                    # 1) dump config
                    tmpf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
                    cfg  = dict(base_cfg); cfg["plot"] = False
                    json.dump(cfg, tmpf); tmpf.close()

                    sat, unsat, to = run_tof(
                        json_cfg      = tmpf.name,
                        delta         = delta,
                        perturb_all   = self.perturb_all,
                        progress_callback = self._progress_cb,
                        verbosity     = self.verbosity,
                        marabou_opts  = mopts,
                        stop_requested=self.isInterruptionRequested
                    )

                    # 3) record results
                    tof_results.append((delta, sat, unsat, to))
                    os.unlink(tmpf.name)
                    continue

                # ── ITD pipeline ────────────────────────
                elif pipeline == "ITD":
                    # gather image paths
                    if os.path.isdir(self.input_path):
                        exts = ('*.png','*.jpg','*.jpeg','*.npy','*.csv')
                        image_paths = sum((glob.glob(os.path.join(self.input_path, e)) for e in exts), [])
                    else:
                        image_paths = [self.input_path]

                    if not image_paths:
                        self.log.emit(f"No images found in directory {self.input_path}")
                        continue

                    delta_sat = delta_unsat = delta_timeout = 0

                    # if targeted mode, initialize confusion matrix once by reading model
                    cmatrix = None
                    if self.vt_mode.startswith("Targeted"):
                        from maraboupy import Marabou
                        net0 = Marabou.read_onnx(self.model_file)
                        out0 = net0.outputVars[0][0]
                        num_classes = out0.shape[0]
                        cmatrix = np.zeros((num_classes, num_classes), dtype=int)

                    feat = self.feature_cfg["features"][0]
                    tmin = feat["properties"]["input_domain"]["min"]
                    tmax = feat["properties"]["input_domain"]["max"]

                    for img_path in image_paths:
                        if self.isInterruptionRequested():
                            self.log.emit(f"=== Stop requested, aborting δ={delta} image loop ===")
                            return
                        # load image/array into [0,1]
                        ext = os.path.splitext(img_path)[1].lower()
                        if ext in ('.png', '.jpg', '.jpeg'):
                            img = Image.open(img_path).convert('RGB')
                            image = np.array(img, dtype=np.float32) / 255.0
                        elif ext == '.csv':
                            df = pd.read_csv(img_path, header=None)
                            flat = df.to_numpy(dtype=float).flatten()
                            side = int(np.sqrt(flat.size))
                            image = flat.reshape(side, side).astype(np.float32)
                            image /= image.max() if image.max() > 1 else 1
                        elif ext == '.npy':
                            image = np.load(img_path).astype(np.float32)
                            if image.max() > 1.0:
                                image /= 255.0
                        else:
                            self.log.emit(f"Skipping unsupported file {img_path}")
                            continue
                        filename = os.path.basename(img_path)

                        
                        if self.vt_mode.startswith("Local"):
                            report, is_robust = local_rob_sens(
                                image=image,
                                filename=filename,
                                delta=delta,
                                onnx_path=self.model_file,
                                tmin=tmin,
                                tmax=tmax,
                                selected_classes=self.selected_classes,
                                verbosity=self.verbosity,
                                marabou_opts=mopts,
                                progress_callback=self._progress_cb,
                                stop_requested=self.isInterruptionRequested
                            )
                        else:
                            report, is_robust = targeted_rob_sens(
                                image=image,
                                filename=filename,
                                delta=delta,
                                onnx_path=self.model_file,
                                tmin=tmin,
                                tmax=tmax,
                                perturb_all_pixels=self.perturb_all,
                                selected_idxs=self.selected_idxs,
                                selected_classes=self.selected_classes,
                                verbosity=self.verbosity,
                                marabou_opts=mopts,
                                progress_callback=self._progress_cb,
                                stop_requested=self.isInterruptionRequested
                            )

                        # only update when cmatrix was initialized (i.e. in Targeted modes)
                        if cmatrix is not None:
                            gt = report["original_prediction"]
                            pred = (report["adversarial_results"][0]["new_prediction"]
                                    if report["adversarial_results"] else gt)
                            cmatrix[gt, pred] += 1


                        # interpret robust→UNSAT, non-robust→SAT
                        status = is_robust
                        if status == "sat":
                            delta_sat += 1
                        elif status == "unsat":
                            delta_unsat += 1
                        elif status == "timeout":
                            delta_timeout += 1
                        else:
                            # unexpected
                            delta_timeout += 1

                         # if we broke out early, also bail on this δ
                    if self.isInterruptionRequested():
                        return

                    # record aggregated counts for this delta
                    itd_results.append((delta, delta_sat, delta_unsat, delta_timeout))
                    self.log.emit(f"δ={delta}: SAT={delta_sat}, UNSAT={delta_unsat}, TIMEOUT={delta_timeout}")
                
                    
                    # for Targeted, store percentage confusion
                    if self.vt_mode.startswith("Targeted") and cmatrix is not None:
                        totals = cmatrix.sum(axis=1, keepdims=True)
                        pct = np.divide(cmatrix, totals, where=totals>0) * 100
                        confusion_data[delta] = pct
                    
                    continue  # move on to next (delta, eps)
                    
                # ── After all δ’s have run ────────────────────────────────
                if self.plot and pipeline == "ITD" and itd_results:
                    if self.vt_mode.startswith("Targeted"):
                        self.log.emit("=== Final grouped‐confusion bar chart ===")
                        gt_labels = [f"C{i}" for i in range(next(iter(confusion_data.values())).shape[0])]
                        fig = plot_grouped_confusion(confusion_data, gt_labels)
                    else:
                        self.log.emit("=== Final aggregated SAT/UNSAT bar chart ===")
                        fig = plot_aggregated_itd(itd_results)

                    out_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
                    fig.savefig(out_png, bbox_inches="tight")
                    plt.close(fig)
                    self.log.emit(f"__PLOT__:{out_png}")

                # ── Jena pipeline ─────────────────────────
                else:
                    tmpf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
                    cfg  = dict(base_cfg); cfg["plot"] = False
                    json.dump(cfg, tmpf); tmpf.close()

                    combo = run_jena(
                        json_path         = tmpf.name,
                        delta_orig        = delta,
                        epsilon_orig      = eps or None,
                        perturb_all       = self.perturb_all,
                        selected_flat_idxs= self.selected_idxs,
                        plot              = False,
                        marabou_opts      = mopts,
                        stop_requested    = self.isInterruptionRequested,
                        progress_callback  = self._progress_cb
                    )

                    jena_results.extend(combo)
                    os.unlink(tmpf.name)

            # ── aggregated Jena heatmap ───────────────────────────
            if pipeline == "Jena" and self.plot and jena_results:
                self.log.emit("=== Generating aggregated Jena heatmap ===")
                # now each row is (Epsilon_raw, Epsilon_norm, Delta, SAT, UNSAT, TIMEOUT)
                df = pd.DataFrame(
                    jena_results,
                    columns=["Epsilon_raw","Epsilon_norm","Delta","SAT","UNSAT","TIMEOUT"]
                 )
                 # group by the raw ε (for display) and δ
                grouped = df.groupby(["Epsilon_raw","Delta"]).sum().reset_index()
                sat_heat = (
                    grouped
                    .pivot(index="Epsilon_raw", columns="Delta", values="SAT")
                    .sort_index(ascending=False)[sorted(grouped["Delta"].unique())]
                )
                timeout_heat = (
                    grouped
                    .pivot(index="Epsilon_raw", columns="Delta", values="TIMEOUT")
                    .sort_index(ascending=False)[sorted(grouped["Delta"].unique())]
                )

                    # 1) Mask so SAT & TIMEOUT never overlap
                sat_only = sat_heat.copy()
                sat_only[timeout_heat > 0] = np.nan

                to_only = timeout_heat.copy()
                to_only[to_only == 0] = np.nan

                # 2) Plot both layers on one axes
                fig = Figure(figsize=(10,6))
                ax  = fig.add_subplot(111)

                # — SAT layer (coolwarm) —
                if sat_only.notna().any().any():
                    sns.heatmap(
                        sat_only,
                        annot=sat_only.fillna(0).astype(int),
                        fmt="d",
                        cmap="coolwarm",
                        vmin=0,
                        vmax=grouped["SAT"].max(),
                        cbar_kws={"label":"# SAT"},
                        mask=sat_only.isna(),
                        ax=ax
                    )

                # — TIMEOUT layer (gray) —
                if to_only.notna().any().any():
                    max_to = int(np.nanmax(to_only.values))
                    sns.heatmap(
                        to_only,
                        annot=to_only.fillna(0).astype(int),
                        fmt="d",
                        cmap="Greys",
                        vmin=0,
                        vmax=max_to,
                        cbar=False,
                        mask=to_only.isna(),
                        linewidths=0.5,
                        linecolor="white",
                        ax=ax
                    )

                ax.set(xlabel="δ", ylabel="ε")
                fig.tight_layout()

                # 3) Emit single PNG
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                fig.savefig(tmp.name, bbox_inches="tight")
                self.log.emit(f"__PLOT__:{tmp.name}")

            # ── After all runs, emit one aggregated TOF or ITD chart ─────────────────
            if self.plot:
                import numpy as _np
                import matplotlib.pyplot as _plt

                # TOF
                if pipeline == "TOF" and tof_results:
                    self.log.emit("=== Generating aggregated TOF bar-chart ===")
                    deltas   = [d for d, _, _, _ in tof_results]
                    sat      = np.array([s for _, s, _, _ in tof_results], dtype=float)
                    unsat    = np.array([u for _, _, u, _ in tof_results], dtype=float)
                    timeout  = np.array([t for _, _, _, t in tof_results], dtype=float)
                    total    = sat + unsat + timeout
                    sat_pct   = np.where(total>0, 100*sat/total, 0)
                    unsat_pct = np.where(total>0, 100*unsat/total, 0)
                    to_pct    = np.where(total>0, 100*timeout/total, 0)

                    x = np.arange(len(deltas))
                    width = 0.8
                    plt.figure(figsize=(10,6))
                    plt.bar(x, sat_pct,    width, color='navy',    label='SAT')
                    plt.bar(x, unsat_pct,  width, bottom=sat_pct,  color='skyblue', label='UNSAT')
                    plt.bar(x, to_pct,     width, bottom=sat_pct+unsat_pct, color='darkgrey', label='TIMEOUT')
                    plt.xticks(x, deltas)
                    plt.xlabel('Delta')
                    plt.ylabel('Percentage (%)')
                    plt.ylim(0,100)
                    plt.legend()
                    plt.tight_layout()

                    out_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
                    plt.savefig(out_png, bbox_inches="tight")
                    plt.close()
                    self.log.emit(f"__PLOT__:{out_png}")

                # ITD
                if pipeline == "ITD" and itd_results:
                    if self.vt_mode in ("Local Robustness", "Local Sensitivity"):
                        self.log.emit("=== Generating multi-category stacked bar chart ===")
                        fig = plot_aggregated_itd(itd_results)
                    else:  # Targeted Sensitivity or Targeted Robustness
                        self.log.emit("=== Generating grouped confusion bar chart across perturbations ===")
                        # confusion_data: { delta -> num_classes×num_classes (%) matrix }
                        # grab num_classes from any entry
                        num_classes = next(iter(confusion_data.values())).shape[0]
                        gt_labels = [f"C{i}" for i in range(num_classes)]
                        fig = plot_grouped_confusion(confusion_data, gt_labels)

                    # save and emit to GUI
                    out_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
                    fig.savefig(out_png, bbox_inches="tight")
                    self.log.emit(f"__PLOT__:{out_png}")

                    gc.collect()  # Force garbage collection after completing the task
        except Exception:
            self.log.emit("Error in worker:\n" + traceback.format_exc())
        finally:
            self._comprehensive_history.clear()
            self._latest_comprehensive_stats.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Restore stdout/stderr
            fd_redir.stop()
            sys.stdout, sys.stderr = old_py_out, old_py_err

class MainWindow(QMainWindow):
    progress = pyqtSignal(float, int, int, int, int, int, float) 
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StabilEdge")
        self._netron_server = None
        self.threads: List[QThread] = []
        self.feature_cfg = None
        self.has_multiclass = False
        self.input_shape = None   # (rows,cols) once input is loaded
        self.pipeline: Optional[str] = None
        self.progress.connect(self._update_progress)

        # Add timer tracking variables
        self._run_start_time = None
        self._elapsed_timer = QTimer()
        self._elapsed_timer.timeout.connect(self._update_elapsed_time)
        self._current_elapsed = 0

        # ─── pixel‐grid setup (must be before build_ui) ─────────────────
        self.gridWidget       = QWidget()
        self.gridLayout       = QGridLayout(self.gridWidget)
        self.selected_pixels: Set[int] = set()
        
        # Initialize pixel_controls to None first
        self.pixel_controls = None

        # ─── now build the rest of the UI ─────────────────────────────
        self._build_ui()
        self._connect_signals()
    
    def _on_plot_toggle(self, checked: bool):
        self.pgPlot.setVisible(checked)
        if checked:
            # Force an immediate update of the plot with the current progress data
            if hasattr(self, '_latest_progress'):
                self._update_progress(*self._latest_progress)
            else:
                # default zero values if no progress yet - use correct number of args including elapsed time
                self._update_progress(0.0, 0, 0, 0, 0, 0, 0.0)
        else:
            # Optionally clear plot when hiding
            self.pgPlot.clear()

    def _build_ui(self):
        # ─── Fix Qt plugin path ──────────────────────────────────
        qt_plug = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(qt_plug, "platforms")

        # ─── Top row: Model & Input ─────────────────────────────
        self.leModel, self.btnModel = QLineEdit(), QPushButton("Import ONNX…")
        self.leInput, self.btnInput = QLineEdit(), QPushButton("Import Input…")
        self.btnJson                = QPushButton("Import Features JSON…")

        # ─── Class-selector menu button (hidden until needed) ─────────
        self.btnClassMenu = QToolButton(self)
        self.btnClassMenu.setText("Select Classes")
        self.btnClassMenu.setPopupMode(QToolButton.InstantPopup)
        self.btnClassMenu.hide()

        # ─── build a little vertical sub-layout just for the Input column
        input_col = QVBoxLayout()
        input_col.setContentsMargins(0,0,0,0)
        input_col.setSpacing(2)
        input_col.addWidget(self.leInput)
        input_col.addWidget(self.btnClassMenu, alignment=Qt.AlignLeft)

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("ONNX Model:"))
        file_row.addWidget(self.leModel)
        file_row.addWidget(self.btnModel)
        file_row.addStretch(1)

        # ← use the nested column here
        file_row.addWidget(QLabel("Input:"))
        file_row.addLayout(input_col)      # ← stacked under leInput
        file_row.addWidget(self.btnInput)
        file_row.addWidget(self.btnJson)

        # ─── Class-selector grid (multi-select) ────────────────
        self.classGridWidget    = QWidget()
        self.classGridLayout    = QGridLayout(self.classGridWidget)
        self.classGridLayout.setContentsMargins(0,0,0,0)
        self.classGridLayout.setHorizontalSpacing(8)
        self.classGridLayout.setVerticalSpacing(8)
        self.selected_classes   = set()
        # hidden until populated in _pick_model & Targeted mode
        self.classGridWidget.hide()

        # ─── Controls row ───────────────────────────────────────
        ctl = QHBoxLayout()
        self.comboVT = QComboBox()
        self.comboVT.addItems([
            "Local Robustness", "Targeted Robustness",
            "Local Sensitivity", "Targeted Sensitivity"
        ])
        ctl.addWidget(QLabel("Verification:"))
        ctl.addWidget(self.comboVT)

        self.leDeltas = QLineEdit(); self.leDeltas.setPlaceholderText("e.g. 0.1,0.2,0.4")
        ctl.addWidget(QLabel("δs:")); ctl.addWidget(self.leDeltas)

        self.lblEpsilons = QLabel("εs:")
        self.leEpsilons = QLineEdit()
        self.leEpsilons.setPlaceholderText("e.g. 0.025,0.03")
        ctl.addWidget(self.lblEpsilons)
        ctl.addWidget(self.leEpsilons)

        self.chkPlot = QCheckBox("Enable Plotting"); self.chkPlot.setChecked(True)
        ctl.addWidget(self.chkPlot)

        self.spinVerb = QSpinBox(); self.spinVerb.setRange(0,2)
        # Tell users in the tooltip or label that 2 is required for progress notifications.
        self.spinVerb.setToolTip("2 = full solver logs (needed for progress warnings); "
                         "1 = brief; 0 = silent")
        ctl.addWidget(QLabel("Verbosity:")); ctl.addWidget(self.spinVerb)

        self.chkEnableTimeout = QCheckBox("Enable timeout")
        ctl.addWidget(self.chkEnableTimeout)
        # single-value fallback
        self.spinTimeout = QSpinBox()
        self.spinTimeout.setRange(0, 24*3600)
        self.spinTimeout.setSuffix(" s")
        self.spinTimeout.setEnabled(False)
        ctl.addWidget(self.spinTimeout)

        # if the user wants per-δ timeouts
        self.chkMultiTimeout = QCheckBox("Separate timeouts for each δ")
        self.chkMultiTimeout.setEnabled(False)
        ctl.addWidget(self.chkMultiTimeout)

        self.leTimeouts = QLineEdit()
        self.leTimeouts.setPlaceholderText("e.g. 0,5,10   (0=no limit)")
        self.leTimeouts.setEnabled(False)
        ctl.addWidget(self.leTimeouts)

        self.spinThreshold = QDoubleSpinBox()
        self.spinThreshold.setRange(0.0, 1.0)
        self.spinThreshold.setSingleStep(0.05)
        self.spinThreshold.setValue(0.8)
        self.spinThreshold.setToolTip("When to suggest extending timeout (fraction of limit)")

        # ─── NEW: progress-bar and toggle ─────────────────────
        self.chkShowProgress = QCheckBox("Show progress")
        self.chkShowProgress.setChecked(False)
        ctl.addWidget(self.chkShowProgress)

        # Optional: add a stretch after the progress bar so buttons don't squeeze it
        ctl.addStretch(1)


        self.btnRun, self.btnStop = QPushButton("Run"), QPushButton("Stop")
        self.btnStop.setEnabled(False)
        ctl.addWidget(self.btnRun); ctl.addWidget(self.btnStop)

        # ─── Jena tree (hidden unless pipeline == "Jena") ────────
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.hide()

        # ─── Tabs: Outline + Netron ─────────────────────────────
        self.tab_widget = QTabWidget()
        self.modelTree  = QTreeWidget()
        self.modelTree.setHeaderLabels(["ONNX Graph"])
        self.tab_widget.addTab(self.modelTree, "Outline")
        self.netron_view = QWebEngineView()
        self.tab_widget.addTab(self.netron_view, "Netron")
        self.tab_widget.setMinimumWidth(300)

        # ─── Console ────────────────────────────────────────────
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        # let it expand when its splitter grows
        self.console.setSizePolicy(QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        self.console.setMinimumHeight(100)    # sensible default floor

        # ─── Plot & Toolbar ─────────────────────────────────────
        self.fig = Figure(figsize=(5,5))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)
        self.navbar = NavigationToolbar(self.canvas, self)

        # Create PyQtGraph PlotWidget for real-time updates
        self.pgPlot = pg.PlotWidget()
        self.pgPlot.setMinimumHeight(500)
        self.pgPlot.showGrid(x=False, y=True)  # Disable X-axis grid lines, keep Y-axis grid


        # ─── Pixel-selection controls (for Sensitivity modes) ──────────
        self.pixel_controls = QWidget()
        pixel_layout = QHBoxLayout(self.pixel_controls)
        pixel_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btnMeaningfulPixels = QPushButton("Select Meaningful Pixels")
        self.btnMeaningfulPixels.setToolTip("Use Otsu threshold to select meaningful pixels automatically")
        self.btnClearPixels = QPushButton("Clear Selection")
        self.btnClearPixels.setToolTip("Clear all selected pixels")
        
        pixel_layout.addWidget(self.btnMeaningfulPixels)
        pixel_layout.addWidget(self.btnClearPixels)
        pixel_layout.addStretch()
        
        self.pixel_controls.hide()  # Hidden until Sensitivity mode


        # ─── Feature-tree & Pixel-grid ──────────────────────────
        self.tree      = QTreeWidget()
        self.tree.hide()
        self.gridWidget = QWidget()
        self.gridLayout = QGridLayout(self.gridWidget)
        self.gridWidget.hide()
        # (you already set spacing & policies on these elsewhere)

        # ─── Build a vertical splitter for right side ──────────
        right_splitter = QSplitter(Qt.Vertical)

        # top container: tree → pixel_controls → grid → toolbar → canvas
        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0,0,0,0)
        top_layout.setSpacing(0)
        top_layout.addWidget(self.tree,       stretch=1)
        top_layout.addWidget(self.pixel_controls, stretch=0)  # Add pixel controls
        top_layout.addWidget(self.gridWidget, stretch=1)
        top_layout.addWidget(self.navbar,     stretch=0)
        top_layout.addWidget(self.canvas,     stretch=3)
        top_layout.addWidget(self.pgPlot, stretch=1)
        self.canvas.hide()  # optionally hide matplotlib canvas by default


        right_splitter.addWidget(top)
        right_splitter.addWidget(self.console)
        # proportions: top takes 3× height, console 1×
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)

        # ─── Main horizontal splitter ──────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tab_widget)
        splitter.addWidget(right_splitter)
        # proportions: tabs 1×, right side 4×
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        # ─── Assemble main window ───────────────────────────────
        central = QWidget()
        vlay = QVBoxLayout(central)
        vlay.addLayout(file_row)
        vlay.addWidget(self.classGridWidget)
        vlay.addLayout(ctl)
        vlay.addWidget(splitter, stretch=4)
        self.setCentralWidget(central)

    def _connect_signals(self):
        self.btnModel.clicked.connect(self._pick_model)
        self.btnInput.clicked.connect(self._pick_input)
        self.btnJson.clicked.connect(self._pick_json)
        self.comboVT.currentTextChanged.connect(self._on_vt_changed)
        self.btnRun.clicked.connect(self._start)
        self.btnStop.clicked.connect(self._stop)
        self.btnMeaningfulPixels.clicked.connect(self._select_meaningful_pixels)
        self.btnClearPixels.clicked.connect(self._clear_pixel_selection)
        self.chkEnableTimeout.toggled.connect(self.spinTimeout.setEnabled)
        self.chkEnableTimeout.toggled.connect(self.chkMultiTimeout.setEnabled)
        self.chkMultiTimeout.toggled.connect(self.leTimeouts.setEnabled)
        self.chkMultiTimeout.toggled.connect(lambda checked: self.spinTimeout.setEnabled(not checked))
        # initialize ε-field enable/disable
        self._on_vt_changed(self.comboVT.currentText())

        # Connect progress visibility toggles
        self.chkPlot.toggled.connect(lambda _: self._toggle_progress_visibility(self.chkShowProgress.isChecked()))
        self.chkShowProgress.toggled.connect(self._toggle_progress_visibility)

        # Set initial progress widget visibility
        self._toggle_progress_visibility(self.chkShowProgress.isChecked())


    def _handle_log(self, msg: str):
        if msg.startswith("__PLOT__:"):
            _, png = msg.split(":",1)
            self._display_plot(png)
        else:
            self.console.appendPlainText(msg)

    def _toggle_progress_visibility(self, checked: bool):
        if checked and self.chkPlot.isChecked():
            self.pgPlot.show()
            # Hide pixel controls and grids when progress bar enabled
            self.gridWidget.hide()
            if hasattr(self, 'pixel_controls') and self.pixel_controls is not None:
                self.pixel_controls.hide()

            # Force immediate redraw with latest known progress (if any)
            if hasattr(self, '_latest_progress'):
                self._update_progress(*self._latest_progress)
        else:
            self.pgPlot.hide()
            # Show grids and controls again when progress bar is disabled
            # Only show if current VT mode is Sensitivity (same as original logic)
            if self.comboVT.currentText().endswith("Sensitivity") and hasattr(self, 'feature_cfg'):
                # We need to decide which grid to show based on JSON features
                px_feats = [
                    f for f in self.feature_cfg.get("features", [])
                    if "pixel" in f.get("name", "").lower()
                ]

                if px_feats and self.input_shape:
                    rows, cols = self.input_shape
                    self._build_pixel_grid(rows, cols)
                    if hasattr(self, 'pixel_controls') and self.pixel_controls is not None:
                        self.pixel_controls.show()  # Show pixel controls
                else:
                    self._build_feature_timestep_grid()
                    if hasattr(self, 'pixel_controls') and self.pixel_controls is not None:
                        self.pixel_controls.hide()  # Hide for non-pixel features
            else:
                # Hide grid and controls if not Sensitivity mode or no feature config
                self.gridWidget.hide()
                if hasattr(self, 'pixel_controls') and self.pixel_controls is not None:
                    self.pixel_controls.hide()

    def _display_plot(self, png_path: str):
        # collapse the feature‐tree and grid so our canvas can fill the space
        self.tree.hide()
        self.gridWidget.hide()

        # now load & render the heatmap PNG
        img = plt.imread(png_path)
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_axes([0,0,1,1], frameon=False)
        ax.imshow(img, aspect="auto")
        ax.set_axis_off()
        self.canvas.draw()

        # clean up
        try: os.unlink(png_path)
        except: pass

    def _pick_model(self):
        """
        Choose an ONNX model, populate the 'Outline' tree, embed Netron,
        and—if it's a multi-class classifier—prepare the class-selector menu.
        """
        # ── 0) Shut down any previous Netron instance
        if self._netron_server:
            try:
                self._netron_server.shutdown()
                self._netron_server.server_close()
            except Exception:
                pass
            self._netron_server = None

        # ── 1) UI reset
        self.modelTree.clear()
        self.tree.clear()
        # hide/clear any old class-selection UI
        self.classGridWidget.hide()
        self.btnClassMenu.hide()
        self.btnClassMenu.setMenu(None)
        self.selected_classes.clear()
        self.has_multiclass = False

        # ── 2) Ask for a *.onnx file
        fn, _ = QFileDialog.getOpenFileName(
            self, "Select ONNX Model", "", "ONNX Files (*.onnx)"
        )
        if not fn:
            return
        self.leModel.setText(fn)

        # ── 3) Helper for attrs
        def _pretty_attr(attr):
            if attr.HasField("f"):
                return attr.f
            if attr.HasField("i"):
                return attr.i
            if attr.HasField("s"):
                try:
                    return attr.s.decode("utf-8")
                except UnicodeDecodeError:
                    blob = attr.s
                    return f"0x{blob[:32].hex()}{'…' if len(blob) > 32 else ''}"
            return "<complex>"

        # ── 4) Load the model
        try:
            model = onnx.load(fn)
        except Exception as e:
            QMessageBox.critical(self, "ONNX load error", str(e))
            return

        # ── 5) Populate Outline tree
        inputs_item = QTreeWidgetItem(["Graph Inputs"])
        for inp in model.graph.input:
            inputs_item.addChild(QTreeWidgetItem([inp.name]))
        self.modelTree.addTopLevelItem(inputs_item)

        outputs_item = QTreeWidgetItem(["Graph Outputs"])
        for out in model.graph.output:
            outputs_item.addChild(QTreeWidgetItem([out.name]))
        self.modelTree.addTopLevelItem(outputs_item)

        for node in model.graph.node:
            label = node.name or f"<{node.op_type}>"
            parent = QTreeWidgetItem([f"{label}  ({node.op_type})"])
            for inp in node.input:
                parent.addChild(QTreeWidgetItem([f"⟵ input: {inp}"]))
            for out in node.output:
                parent.addChild(QTreeWidgetItem([f"output: {out} ⟶"]))
            for attr in node.attribute:
                parent.addChild(
                    QTreeWidgetItem([f" • {attr.name} = {_pretty_attr(attr)}"])
                )
            self.modelTree.addTopLevelItem(parent)
        self.modelTree.expandAll()

        # ── 6) Spin up Netron
        sock = socket.socket()
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()
        self._netron_server = netron.start(fn, address=("127.0.0.1", port), browse=False)
        self.netron_view.load(QUrl(f"http://127.0.0.1:{port}"))
        self.tab_widget.setCurrentWidget(self.netron_view)

        # ── 7) Detect multiclass (>2 classes)
        out_type    = model.graph.output[0].type.tensor_type
        dims        = [d.dim_value for d in out_type.shape.dim if d.dim_value > 0]
        num_classes = dims[-1] if dims else 0
        self.has_multiclass = (num_classes > 2)

        # clear any old menu so nothing lingers
        self.btnClassMenu.setMenu(None)

        if self.has_multiclass:
            menu = QMenu(self.btnClassMenu)
            class_checkboxes = []

            # “All classes”
            all_cb = QCheckBox("All classes")
            all_act = QWidgetAction(menu)
            all_act.setDefaultWidget(all_cb)
            menu.addAction(all_act)

            for i in range(num_classes):
                cb = QCheckBox(f"Class {i}")
                cb.toggled.connect(lambda checked, idx=i: (
                    self.selected_classes.add(idx) if checked
                    else self.selected_classes.discard(idx)
                ))
                act = QWidgetAction(menu)
                act.setDefaultWidget(cb)
                menu.addAction(act)
                class_checkboxes.append(cb)

            all_cb.toggled.connect(lambda checked: [
                cb.setChecked(checked) for cb in class_checkboxes
            ])

            self.btnClassMenu.setMenu(menu)
        else:
            # explicitly hide and clear for 0–2 classes
            self.btnClassMenu.hide()
            self.selected_classes.clear()

        # ── 8) Re-apply the VT‐mode rules so only Targeted+3+ actually shows it
        self._on_vt_changed(self.comboVT.currentText())

    def _pick_input(self):
        """
        1) Let the user choose an image / CSV / NumPy array.
        2) Record its shape in self.input_shape.
        3) Keep the grid hidden until JSON arrives.
        """
        filter_str = (
            "All Supported (*.png *.jpg *.jpeg *.csv *.npy);;"
            "Images (*.png *.jpg *.jpeg);;"
            "CSV (*.csv);;"
            "NumPy Array (*.npy)"
        )
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input File",
            "",
            filter_str
        )

        if not fn:
            return

        self.leInput.setText(fn)
        ext = os.path.splitext(fn)[1].lower()

        # determine shape
        if ext in ('.png', '.jpg', '.jpeg'):
            from PIL import Image
            import numpy as np
            arr = np.array(Image.open(fn).convert('L'))
            rows, cols = arr.shape[:2]

        elif ext == '.csv':
            import numpy as np
            import pandas as pd
            import re

            # 1) Peek at the first line to see if it’s a header
            with open(fn, 'r') as f:
                first = f.readline().strip()
            tokens = first.split(',')

            # if any token isn’t a pure float syntax, treat as header
            float_re = re.compile(r'^-?\d+\.?\d*(?:[eE][-+]?\d+)?$')
            has_header = any(not float_re.match(tok.strip()) for tok in tokens)

            # 2) Read with pandas, skipping header if needed
            if has_header:
                df = pd.read_csv(fn, header=0)   # skip first line
            else:
                df = pd.read_csv(fn, header=None)

            # 3) Flatten to 1D float array
            try:
                flat = df.to_numpy(dtype=float).flatten()
            except ValueError as e:
                QMessageBox.warning(
                    self, "Error loading CSV",
                    f"Non-numeric data found:\n{e}"
                )
                return

            total = flat.size
            side  = int(np.sqrt(total))
            if side*side != total:
                QMessageBox.warning(
                    self, "Invalid Image Size",
                    f"CSV contains {total} pixels which is not a perfect square."
                )
                return

            # 4) Reshape, stash shape & return
            img_arr = flat.reshape(side, side)
            self.input_shape = (side, side)
            self.image_array = img_arr     # if you want to display it later
            self.gridWidget.hide()
            return

        elif ext == '.npy':
            import numpy as np
            try:
                arr = np.load(fn)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not load .npy file:\n{e}")
                return
            if arr.ndim < 2:
                QMessageBox.warning(self, "Error", "NumPy array must have at least 2 dimensions")
                return
            rows, cols = arr.shape[:2]

        else:
            QMessageBox.warning(self, "Error", "Unsupported input format")
            return

        # stash shape but keep grid hidden until JSON arrives
        self.input_shape = (rows, cols)
        self.gridWidget.hide()

    def _pick_json(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Select Features JSON", "", "JSON Files (*.json)"
        )
        if not fn:
            return

        # 1) Ensure an input is loaded first
        if not hasattr(self, 'input_shape') or not self.leInput.text().strip():
            QMessageBox.information(
                self, "Load Input First",
                "Please import an input file before loading the JSON."
            )
            return

        # 2) Load & parse JSON
        try:
            with open(fn, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except json.JSONDecodeError as e:
            QMessageBox.critical(
                self, "JSON Parse Error",
                f"Could not parse JSON:\n{e}"
            )
            return

        # 3) Stash the config for later
        self.feature_cfg = cfg
        self.json_path   = fn

        # 4) Detect pipeline from filename
        base = os.path.basename(fn)
        if base == "ITD_features.json":
            self.pipeline = "ITD"
        elif base == "TOF_features.json":
            self.pipeline = "TOF"
        elif base == "Jena_features.json":
            self.pipeline = "Jena"
        else:
            QMessageBox.warning(
                self, "Unknown JSON",
                f"Unrecognized JSON {base}, defaulting to ITD."
            )
            self.pipeline = "ITD"

        # Disable/enable ε-input
        is_jena = (self.pipeline == "Jena")
        self.lblEpsilons.setEnabled(is_jena)
        self.leEpsilons.setEnabled(is_jena)

        # 5) Input vs JSON compatibility
        ext = os.path.splitext(self.leInput.text().lower())[1]
        px_feats = [
            f for f in cfg.get("features", [])
            if "pixel" in f.get("name", "").lower()
        ]

        # —— new .npy handling block —— 
        if ext == '.npy':
            try:
                arr = np.load(self.leInput.text())
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not load .npy file:\n{e}")
                return   # still bail if we can’t even load

            if px_feats:
                # JSON expects pixel input → must be 2D image
                if arr.ndim != 2:
                    QMessageBox.warning(
                        self, "Unsupported Input",
                        f"JSON defines pixel features but .npy has ndim={arr.ndim}; expected 2."
                    )
                    return  # bail on truly invalid shape
                # OK: record image shape (2D)
                self.input_shape = (arr.shape[0], arr.shape[1])
                # <— no return here, let the grid-builder run below

            else:
                # JSON expects time-series → accept 1D or 2D
                if arr.ndim == 0:
                    QMessageBox.warning(
                        self, "Unsupported Input",
                        ".npy must contain at least one time-series sample."
                    )
                    return  # bail on truly invalid shape
                # OK: no per-pixel grid for time-series
                self.input_shape = None

        # 5b) Existing checks for CSV / image files
        allowed_px = {'.png', '.jpg', '.jpeg', '.csv'}
        if px_feats and ext not in allowed_px.union({'.npy'}):
            QMessageBox.warning(
                self, "Unsupported Input",
                "JSON defines pixel features but input isn’t image/CSV/NPY."
            )
            return

        if not px_feats and ext != '.npy':
            QMessageBox.warning(
                self, "Unsupported Input",
                "JSON defines time-series features but input isn’t an .npy."
            )
            return

        # 6) Build the grid only for Sensitivity modes
        if self.comboVT.currentText().endswith("Sensitivity"):
            if px_feats:
                rows, cols = self.input_shape
                self._build_pixel_grid(rows, cols)
            else:
                self._build_feature_timestep_grid()
        else:
            self.gridWidget.hide()

    def _build_feature_timestep_grid(self):
        # clear old
        for i in reversed(range(self.gridLayout.count())):
            w = self.gridLayout.itemAt(i).widget()
            if w:
                w.setParent(None)
        self.selected_pixels.clear()

        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setHorizontalSpacing(8)
        self.gridLayout.setVerticalSpacing(8)
        self.gridLayout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        feats = self.feature_cfg["features"]
        T = len(feats[0]["properties"]["starting_indexes"])

        # headers
        self.gridLayout.addWidget(QLabel(""), 0, 0)
        for col in range(T):
            lbl = QLabel(f"t{col}")
            lbl.setAlignment(Qt.AlignCenter)
            self.gridLayout.addWidget(lbl, 0, col+1)
        all_hdr = QLabel("All")
        all_hdr.setAlignment(Qt.AlignCenter)
        self.gridLayout.addWidget(all_hdr, 0, T+1)

        # feature rows
        for row, feat in enumerate(feats, start=1):
            name_lbl = QLabel(feat["name"])
            name_lbl.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
            self.gridLayout.addWidget(name_lbl, row, 0)

            row_buttons = []
            for col, idx in enumerate(feat["properties"]["starting_indexes"], start=1):
                btn = QPushButton()
                btn.setCheckable(True)
                btn.setMinimumSize(30, 30)
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                btn.setStyleSheet("""
                    QPushButton { }
                    QPushButton:checked { background-color: red; }
                """)
                btn.toggled.connect(
                    lambda checked, i=idx: (
                        self.selected_pixels.add(i) if checked
                        else self.selected_pixels.discard(i)
                    )
                )
                self.gridLayout.addWidget(btn, row, col)
                row_buttons.append(btn)

            select_all = QPushButton("▢")
            select_all.setCheckable(True)
            select_all.setMinimumSize(30, 30)
            select_all.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            select_all.setToolTip("Select/clear all time steps in this row")
            select_all.toggled.connect(
                lambda checked, buttons=row_buttons: [
                    b.setChecked(checked) for b in buttons
                ]
            )
            self.gridLayout.addWidget(select_all, row, T+1)

        # stretch every column and row equally
        for col in range(T+2):           # +2 for the feature‐name and "All" columns
            self.gridLayout.setColumnStretch(col, 1)
        for row in range(len(feats)+1):  # +1 for the header row
            self.gridLayout.setRowStretch(row, 1)

        # grid itself can expand
        self.gridWidget.setSizePolicy(QSizePolicy.Expanding,
                                    QSizePolicy.Expanding)
        self.gridWidget.show()

    def _on_vt_changed(self, vt: str):
        """
        Handles switching between Local/Targeted and Robustness/Sensitivity.
        Ensures ε-field is visible but only enabled for Jena, and also
        toggles the grid & class-menu as before.
        """
        pipeline = getattr(self, "pipeline", None)
        sens_mode = vt.endswith("Sensitivity")
        targeted_mode = vt.startswith("Targeted ")

        # ── Epsilon ALWAYS visible, but only editable for Jena ───────
        self.lblEpsilons.setVisible(True)
        self.leEpsilons.setVisible(True)
        self.leEpsilons.setEnabled(pipeline == "Jena")

        # ── grid and pixel controls only in Sensitivity ────────────────
        self.gridWidget.setVisible(sens_mode)
        # FIX: Check if pixel_controls exists and properly evaluate input_shape
        if hasattr(self, 'pixel_controls') and self.pixel_controls is not None:
            should_show = sens_mode and hasattr(self, 'input_shape') and self.input_shape is not None
            self.pixel_controls.setVisible(should_show)

        # ── class-menu only in Targeted & multiclass ───────────────
        if targeted_mode and self.has_multiclass:
            self.btnClassMenu.show()
        else:
            self.btnClassMenu.hide()
            self.selected_classes.clear()

    def _on_pipeline_changed(self, pipe: str):
        if pipe == "Jena":
            cfg = json.load(open(CONFIG_MAP["Jena"], "r"))
            self.tree.clear()
            for feat in cfg["features"]:
                parent = QTreeWidgetItem([feat["name"]])
                parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
                parent.setCheckState(0, Qt.Unchecked)
                for idx in feat["properties"]["starting_indexes"]:
                    child = QTreeWidgetItem([str(idx)])
                    child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                    child.setCheckState(0, Qt.Unchecked)
                    parent.addChild(child)
                self.tree.addTopLevelItem(parent)
            self.tree.show()
        else:
            self.tree.hide()

    def _build_pixel_grid(self, rows: int, cols: int):
        # clear any old cells
        for i in reversed(range(self.gridLayout.count())):
            w = self.gridLayout.itemAt(i).widget()
            if w:
                w.setParent(None)

        self.selected_pixels.clear()

        # spacing
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setHorizontalSpacing(8)
        self.gridLayout.setVerticalSpacing(8)

        # create buttons
        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                btn = QPushButton()
                btn.setCheckable(True)
                btn.setMinimumSize(30, 30)  
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

                btn.setStyleSheet("""
                    QPushButton { /* normal */ }
                    QPushButton:checked { background-color: red; }
                """)

                btn.toggled.connect(
                    lambda checked, i=idx: (
                        self.selected_pixels.add(i) if checked
                        else self.selected_pixels.discard(i)
                    )
                )
                self.gridLayout.addWidget(btn, row, col)

        # stretch every row/column equally
        for col in range(cols):
            self.gridLayout.setColumnStretch(col, 1)
        for row in range(rows):
            self.gridLayout.setRowStretch(row, 1)

        # allow the grid itself to expand
        self.gridWidget.setSizePolicy(QSizePolicy.Expanding,
                                    QSizePolicy.Expanding)
        self.gridWidget.show()
        
        # Show pixel controls for sensitivity modes with pixel features
        if self.comboVT.currentText().endswith("Sensitivity"):
            self.pixel_controls.show()

    def _select_meaningful_pixels(self):
        """Apply Otsu threshold to select meaningful pixels automatically"""
        if not hasattr(self, 'input_shape') or not self.leInput.text().strip():
            QMessageBox.warning(self, "Error", "Please load an input file first.")
            return
        
        input_path = self.leInput.text().strip()
        ext = os.path.splitext(input_path)[1].lower()
        
        try:
            # Load the image based on file type
            if ext in ('.png', '.jpg', '.jpeg'):
                from PIL import Image
                img = Image.open(input_path).convert('L')  # Convert to grayscale
                image = np.array(img, dtype=np.float32) / 255.0
            elif ext == '.csv':
                import pandas as pd
                import re
                
                # Check for header
                with open(input_path, 'r') as f:
                    first = f.readline().strip()
                tokens = first.split(',')
                float_re = re.compile(r'^-?\d+\.?\d*(?:[eE][-+]?\d+)?$')
                has_header = any(not float_re.match(tok.strip()) for tok in tokens)
                
                if has_header:
                    df = pd.read_csv(input_path, header=0)
                else:
                    df = pd.read_csv(input_path, header=None)
                
                flat = df.to_numpy(dtype=float).flatten()
                total = flat.size
                side = int(np.sqrt(total))
                if side * side != total:
                    QMessageBox.warning(self, "Error", "CSV data is not a perfect square.")
                    return
                
                image = flat.reshape(side, side).astype(np.float32)
                # Normalize to [0,1] if needed
                if image.max() > 1.0:
                    image = image / image.max()
                    
            elif ext == '.npy':
                image = np.load(input_path).astype(np.float32)
                if image.ndim != 2:
                    QMessageBox.warning(self, "Error", "NumPy array must be 2D for pixel selection.")
                    return
                if image.max() > 1.0:
                    image = image / 255.0
            else:
                QMessageBox.warning(self, "Error", "Unsupported file format for meaningful pixel selection.")
                return
            
            # Import the function from TOF
            from TOF import compute_meaningful_mask
            
            # Compute the meaningful mask
            mask = compute_meaningful_mask(image)
            rows, cols = mask.shape
            
            # Clear current selection
            self.selected_pixels.clear()
            
            # Update pixel buttons and selection
            meaningful_count = 0
            for row in range(rows):
                for col in range(cols):
                    idx = row * cols + col
                    if mask[row, col]:
                        self.selected_pixels.add(idx)
                        meaningful_count += 1
                        
                        # Update button appearance if it exists
                        button = self.gridLayout.itemAtPosition(row, col)
                        if button and button.widget():
                            button.widget().setChecked(True)
            
            self.console.appendPlainText(f"Selected {meaningful_count} meaningful pixels using Otsu threshold.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to compute meaningful pixels:\n{str(e)}")

    def _clear_pixel_selection(self):
        """Clear all selected pixels"""
        self.selected_pixels.clear()
        
        # Update all pixel buttons to unchecked state
        if hasattr(self, 'input_shape') and self.input_shape:
            rows, cols = self.input_shape
            for row in range(rows):
                for col in range(cols):
                    button = self.gridLayout.itemAtPosition(row, col)
                    if button and button.widget():
                        button.widget().setChecked(False)
        
        self.console.appendPlainText("Cleared all pixel selections.")

    def _start(self):
        # 1) Must have JSON loaded
        if not hasattr(self, 'feature_cfg'):
            QMessageBox.warning(self, "Error", "Load a features JSON first.")
            return

        # 2) Gather params
        pipeline   = getattr(self, "pipeline", "")
        model_path = self.leModel.text().strip()
        input_path = self.leInput.text().strip()
        plot       = self.chkPlot.isChecked()
        verbosity  = self.spinVerb.value()

        if not model_path or not input_path:
            QMessageBox.warning(self, "Error", "Select model & input.")
            return

        # 3) Parse δs
        try:
            deltas = [float(x) for x in self.leDeltas.text().split(",") if x.strip()]
            self._deltas = deltas
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid δ list")
            return

        # 4) Parse εs only for Jena; for ITD/TOF always none
        if pipeline == "Jena":
            try:
                epsilons = [float(x) for x in self.leEpsilons.text().split(",") if x.strip()]
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid ε list")
                return
        else:
            epsilons = [None]
        # 5) Parse timeouts (either single or per‐δ)
        if self.chkEnableTimeout.isChecked():
            if self.chkMultiTimeout.isChecked():
                # user provided a comma list of timeouts
                try:
                    timeouts = [int(x) for x in self.leTimeouts.text().split(",") if x.strip()]
                except ValueError:
                    QMessageBox.warning(self, "Error", "Invalid timeout list")
                    return
                if len(timeouts) != len(deltas):
                    QMessageBox.warning(
                        self, "Error",
                        f"Need {len(deltas)} timeouts (one per δ), you gave {len(timeouts)}"
                    )
                    return
            else:
                # same timeout for every δ
                timeouts = [self.spinTimeout.value()] * len(deltas)
        else:
            # no timeout → all zeros
            timeouts = [0] * len(deltas)

        # 6) Warn if any timeout is 0 (i.e. “no limit”), in either mode
        if self.chkEnableTimeout.isChecked():
            unlimited = [d for d, t in zip(deltas, timeouts) if t == 0]
            if unlimited:
                QMessageBox.information(
                    self, "Timeout Notice",
                    "You’ve set a timeout of 0 (i.e. no limit) for δ = "
                    + ", ".join(str(d) for d in unlimited)
                    + ".\nThose verifications will run without any time cap."
                )

        # 7) Collect pixel‐ and class‐selections
        selected_idxs    = list(self.selected_pixels)
        selected_classes = sorted(self.selected_classes) if self.has_multiclass else None

        if self.comboVT.currentText().endswith("Sensitivity") and not selected_idxs:
            QMessageBox.warning(self, "Error", "Select at least one pixel/time-step.")
            return

        # 8) Disable UI while running and start timer
        self.btnRun.setEnabled(False)
        self.btnStop.setEnabled(True)
        
        # Initialize elapsed time tracking
        self._run_start_time = time.time()
        self._current_elapsed = 0
        self._elapsed_timer.start(1000)  # Update every second
        
        # Show and enable the matplotlib canvas when starting
        self.canvas.setVisible(True)
        self.canvas.setEnabled(True)
        # optionally reset the figure
        self.fig.clear()
        self.canvas.draw()


        # 9) Build (δ, ε, timeout) combos
        combos = [
            (delta, eps, to)
            for (delta, to) in zip(deltas, timeouts)
            for eps in epsilons
        ]
        if not combos:
            QMessageBox.warning(self, "Error", "No (δ,ε,timeout) combinations to run.")
            return

        # 10) Spawn the background worker
        worker = MultiWorker(
            pipeline         = pipeline,
            vt_mode          = self.comboVT.currentText(),
            combos           = combos,              # now includes per‐δ timeout
            perturb_all      = self.comboVT.currentText().endswith("Robustness"),
            model_file       = model_path,
            input_path       = input_path,
            json_path        = self.json_path,
            selected_idxs    = selected_idxs,
            plot             = plot,
            verbosity        = verbosity,
            feature_cfg      = self.feature_cfg,
            selected_classes = selected_classes,
            suggestion_threshold = self.spinThreshold.value(),
            progress_callback = None
        )
        worker.log.connect(self._handle_log, Qt.QueuedConnection)
        worker.progress.connect(self._update_progress, Qt.QueuedConnection)
        worker.finished.connect(lambda: self._on_finished(worker), Qt.QueuedConnection)
        worker.timeout_dialog_requested.connect(self._handle_timeout_dialog, Qt.QueuedConnection)
        self.threads.append(worker)
        worker.start()


    def _update_elapsed_time(self):
        """Update elapsed time display every second"""
        if self._run_start_time and self.threads:
            current_time = time.time()
            self._current_elapsed = current_time - self._run_start_time
            
            # If we have latest progress data, update the display
            if hasattr(self, '_latest_progress') and self.chkShowProgress.isChecked() and self.chkPlot.isChecked():
                # Update with current elapsed time
                delta, done, total, sat, unsat, timeout, _ = self._latest_progress
                self._update_progress(delta, done, total, sat, unsat, timeout, self._current_elapsed)

    def _emit_progress(self, done, total, sat, unsat, timeout):
        # forward from Python into the Qt signal
        self.progress.emit(done, total, sat, unsat, timeout)


    @pyqtSlot(float, int, int, int, int, int, float)
    def _update_progress(self, delta, done, total, sat, unsat, timeout, elapsed_time):
        # Store latest progress for potential reuse
        self._latest_progress = (delta, done, total, sat, unsat, timeout, elapsed_time)
        
        # Format elapsed time nicely
        if elapsed_time >= 3600:  # More than 1 hour
            elapsed_str = f"{int(elapsed_time//3600)}h {int((elapsed_time%3600)//60)}m {int(elapsed_time%60)}s"
        elif elapsed_time >= 60:  # More than 1 minute
            elapsed_str = f"{int(elapsed_time//60)}m {int(elapsed_time%60)}s"
        else:  # Less than 1 minute
            elapsed_str = f"{int(elapsed_time)}s"
        
        est_text = ""
        worker = self.threads[-1] if self.threads else None
        if worker and hasattr(worker, "_comprehensive_history") and len(worker._comprehensive_history) >= 2:
            old = worker._comprehensive_history[0]
            new = worker._comprehensive_history[-1]
            fmt = "%H:%M:%S"
            try:
                t_old = time.mktime(time.strptime(old.timestamp, fmt))
                t_new = time.mktime(time.strptime(new.timestamp, fmt))
                dt = t_new - t_old
                dcount = old.active_pl_current - new.active_pl_current  # Fix: use active_pl_current
                if dt > 0 and dcount > 0:
                    rate = dcount / dt
                    rem = new.active_pl_current / rate if rate > 0 else None
                    if rem is not None:
                        est_text = f"  (est {int(rem)}s left)"
            except (ValueError, AttributeError):
                pass  # Skip estimation if parsing fails
    
        # 1) Figure out i/n for this delta
        if hasattr(self, "_deltas"):
            try:
                i = self._deltas.index(delta) + 1
                n = len(self._deltas)
            except ValueError:
                i = n = None
    
        # 2) Clear old graphics
        self.pgPlot.clear()
        if hasattr(self, '_delta_text_item'):
            self.pgPlot.removeItem(self._delta_text_item)
    
        # 3) If they're hiding the bar, bail
        if not (self.chkShowProgress.isChecked() and self.chkPlot.isChecked()):
            return
    
        # 4) Draw your stacked‐bar segments...
        if total == 0:
            sat_frac = unsat_frac = timeout_frac = 0
        else:
            sat_frac     = sat   / total
            unsat_frac   = unsat / total
            timeout_frac = timeout / total
    
        widths      = [sat_frac, unsat_frac, timeout_frac]
        colors      = ['#d62728', '#1f77b4', '#ff7f0e']
        labels      = [f"SAT: {sat}", f"UNSAT: {unsat}", f"TIMEOUT: {timeout}"]
        text_colors = ['white', 'white', 'black']
    
        x0 = 0
        for w, c, lbl, tc in zip(widths, colors, labels, text_colors):
            if w > 0:
                bar = pg.BarGraphItem(x0=x0, y=0.5, height=0.6, width=w, brush=c)
                self.pgPlot.addItem(bar)
                ti = pg.TextItem(lbl, color=tc, anchor=(0.5,0.5))
                ti.setFont(QFont("Arial",10,QFont.Bold))
                ti.setPos(x0 + w/2, 0.5)
                self.pgPlot.addItem(ti)
                x0 += w
    
        self.pgPlot.setXRange(0,1)
        self.pgPlot.setYRange(0,1)
        self.pgPlot.getAxis('left').setTicks([])
        
        # Set custom X-axis ticks with percentage labels
        x_ticks = [
            (0.0, '0%'),
            (0.25, '25%'),
            (0.5, '50%'),
            (0.75, '75%'),
            (1.0, '100%')
        ]
        self.pgPlot.getAxis('bottom').setTicks([x_ticks])
        
        # Style the bottom axis - Show visible tick marks like ruler
        bottom_axis = self.pgPlot.getAxis('bottom')
        bottom_axis.setStyle(
            tickFont=QFont("Arial", 16, QFont.Bold),
            tickLength=1  # Visible tick marks like ruler marks
        )
        bottom_axis.setPen(pg.mkPen(color='white', width=2))
        bottom_axis.setTextPen(pg.mkPen(color='white'))
        
        # Add custom vertical ruler lines from labels up to progress bar
        ruler_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        for x_pos in ruler_positions:
            # Create vertical line from bottom axis labels (y=-0.05) to progress bar area (y=0.1)
            ruler_line = pg.PlotDataItem([x_pos, x_pos], [-0.05, 0.01], 
                                       pen=pg.mkPen(color='white', width=1))  # Solid straight line
            self.pgPlot.addItem(ruler_line)
        
        # 5) Enhanced title with elapsed time
        if hasattr(self, "_deltas"):
            try:
                idx = self._deltas.index(delta) + 1
                total_d = len(self._deltas)
                delta_label = f"δ {idx}/{total_d} = {delta}   "
            except ValueError:
                delta_label = ""
        else:
            delta_label = ""
    
        # Include elapsed time in the title
        title_text = f"{delta_label}Progress: {done}/{total} | Time Elapsed: {elapsed_str} {est_text}"
        self.pgPlot.setTitle(title_text)
        
        # Style the title
        title_label = self.pgPlot.plotItem.titleLabel
        large_font = QFont("Arial", 16, QFont.Bold)
        title_label.item.setFont(large_font)
        html_title = f'<span style="font-size: 20pt; font-weight: bold;">{title_text}</span>'
        title_label.setText(html_title)

        


    def _handle_timeout_dialog(self, snap, estimated_remaining, time_left, delta):
        """Handle timeout dialog in main thread"""
        active_now, total_pl = snap.active_pl
        completion_pct = ((total_pl - active_now) / total_pl * 100) if total_pl > 0 else 0
        
        # Calculate solver efficiency metrics
        pivot_efficiency = (snap.real_pivots / max(snap.simplex_pivots, 1) * 100) if snap.simplex_pivots > 0 else 0
        
        if estimated_remaining > time_left:
            outlook = f"≈{int(estimated_remaining - time_left)}s past your timeout"
            recommendation = f"Recommended extension: +{int(estimated_remaining - time_left + 120)}s (with 2min buffer)"
        else:
            outlook = f"≈{int(estimated_remaining)}s to finish (within timeout)"
            recommendation = "Current timeout appears sufficient"
        
        # Build comprehensive dialog
        dialog_text = f"""🔍 COMPREHENSIVE TIMEOUT ANALYSIS (δ={delta})
    
     Current Progress: {completion_pct:.1f}% complete ({active_now}/{total_pl} active constraints)
    
     Search Tree Status:
    • SMT Depth: {snap.smt_depth}
    • Visited States: {snap.visited_states:,}
    • Splits/Pops: {snap.num_splits:,} / {snap.num_pops:,}
    • Max Stack Depth: {snap.max_stack_depth}
    
    ⚡ Solver Performance:
    • Main Loop Iterations: {snap.main_loop_iterations:,}
    • Simplex Pivots: {snap.simplex_pivots:,} (Real: {snap.real_pivots:,}, Degenerate: {snap.degenerate_pivots:,})
    • Pivot Efficiency: {pivot_efficiency:.1f}%
    • Tableau Size: {snap.tableau_m:,} × {snap.tableau_n:,}
    • Bounds Tightened: {snap.tightened_bounds:,}
    
     Time Analysis:
    • Total Time: {snap.total_time_ms:,}ms
    • Main Loop Time: {snap.main_loop_time_ms:,}ms
    • Preprocessing: {snap.preprocessing_time_ms:,}ms
    • Estimated Remaining: {outlook}
    • Time Left: ~{int(time_left)}s
    
     {recommendation}
    
    Extend timeout?"""
        
        reply = QMessageBox.question(
            self,
            f"🔍 Enhanced Timeout Analysis - δ={delta}",
            dialog_text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        # Send response back to worker
        if self.threads:
            worker = self.threads[-1]
            worker.set_timeout_dialog_response(reply == QMessageBox.Yes)
    
    def _stop(self):
        if self.threads:
            thr = self.threads[-1]
            thr.requestInterruption()
            self.console.appendPlainText("=== Stop requested… waiting for clean exit ===")
            
            # Don't block GUI - use a timer to check if thread finished
            def check_thread_finished():
                if thr.isFinished():
                    self.console.appendPlainText("=== Thread stopped successfully ===")
                    return  # Timer will be destroyed automatically
                elif not thr.wait(100):  # Check every 100ms
                    # Still running, schedule another check
                    QTimer.singleShot(100, check_thread_finished)
            
            # Start checking
            QTimer.singleShot(100, check_thread_finished)
    
        # Disable or hide the matplotlib canvas when stopping
        self.canvas.setVisible(False)
        self.canvas.setEnabled(False)

    def _on_finished(self, thr):
        # Stop the elapsed timer
        self._elapsed_timer.stop()
        self._run_start_time = None
        
        self.btnRun.setEnabled(True)
        self.btnStop.setEnabled(False)

        # Hide the progress plot
        self.pgPlot.hide()
        self.pgPlot.clear()
    
        if thr in self.threads:
            self.threads.remove(thr)
        
        # FIX: Aggressive cleanup of worker resources
        if hasattr(thr, '_comprehensive_history'):
            thr._comprehensive_history.clear()
        if hasattr(thr, '_latest_comprehensive_stats'):
            thr._latest_comprehensive_stats.clear()
        
        # FIX: Disconnect all signals to prevent late callbacks
        try:
            thr.log.disconnect()
            thr.progress.disconnect()
            thr.timeout_dialog_requested.disconnect()
        except:
            pass
        
        # Clean up the thread
        thr.deleteLater()
        
        # FIX: Force garbage collection after thread cleanup
        import gc
        gc.collect()
    
    def closeEvent(self, event):
        # 1) Stop all worker threads first
        for thr in list(self.threads):
            thr.requestInterruption()
            
            # FIX: Force cleanup of worker resources
            if hasattr(thr, '_comprehensive_history'):
                thr._comprehensive_history.clear()
            if hasattr(thr, '_latest_comprehensive_stats'):
                thr._latest_comprehensive_stats.clear()
            
            if not thr.wait(2000):  # Reduced timeout
                thr.terminate()
                thr.wait(1000)
        
        # 2) Clean up GUI resources
        if hasattr(self, 'pgPlot'):
            self.pgPlot.clear()
        
        if hasattr(self, 'canvas'):
            self.canvas.figure.clear()
        
        # 3) Shut down Netron server
        if self._netron_server:
            try:
                self._netron_server.shutdown()
                self._netron_server.server_close()
            except:
                pass
        
        # 4) Force garbage collection
        import gc
        gc.collect()
        
        self.threads.clear()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # build a multi-size QIcon so Qt never has to upscale
    icon = QIcon()
    base = os.path.dirname(__file__)
    for sz in (16, 32, 64, 128, 256, 512):
        icon.addFile(os.path.join(base, f"icon-{sz}.png"))
    app.setWindowIcon(icon)

    win = MainWindow()
    win.setWindowIcon(icon)
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())

