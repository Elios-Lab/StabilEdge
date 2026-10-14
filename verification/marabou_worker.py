"""Existing Marabou workflows, loaded only when the engine is selected."""
import os
import sys
import gc
import time
import json
import tempfile
import glob
import collections
import traceback
import re
from typing import Optional, List, Tuple, Dict, Any, Callable
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PyQt5.QtCore import QThread, QProcess, pyqtSignal, pyqtSlot
from scripts.ITD import local_rob_sens, targeted_rob_sens, plot_aggregated_itd, plot_grouped_confusion
from scripts.TOF import run_tof
from scripts.Jena import run_jena
from verification.core import VerificationResult, Status, solver_status

# ══════════════════════════════════════════════════════════════════
# LEGACY REGEX PATTERNS (Optional - Only Used with verbosity >= 1)
# ══════════════════════════════════════════════════════════════════
# These patterns are DEPRECATED in favor of Python API (verbosity=0).
# They're kept for backward compatibility when users want detailed 
# text logs (verbosity=2). Safe to remove if API-only mode is enforced.

# Version detection (if available in output)
RX_MARABOU_VERSION = re.compile(r"Marabou\s+(?:version|v)?:?\s*([\d.]+)", re.IGNORECASE)

# Core statistics patterns - flexible whitespace and optional punctuation
RX_ACTIVE_PL         = re.compile(r"(?:Number\s+of\s+)?active\s+piecewise[\s-]?linear\s+constraints?:?\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
RX_SMT_DEPTH         = re.compile(r"(?:Current\s+|Total\s+)?depth(?:\s+is)?:?\s+(\d+)", re.IGNORECASE)
RX_NUM_SPLITS        = re.compile(r"(?:Number\s+of\s+)?splits?:?\s+(\d+)", re.IGNORECASE)
RX_TOTAL_TIME        = re.compile(r"Total\s+(?:time\s+)?elapsed:?\s+(\d+)\s+(?:milli(?:seconds?)?|ms)", re.IGNORECASE)
RX_MAIN_LOOP_TIME    = re.compile(r"Main\s+loop:?\s+(\d+)\s+(?:milli(?:seconds?)?|ms)", re.IGNORECASE)
RX_MAIN_LOOP_ITERATIONS = re.compile(r"(?:Number\s+of\s+)?main\s+loop\s+iterations?:?\s+(\d+)", re.IGNORECASE)
RX_SIMPLEX_PIVOTS    = re.compile(r"(?:Total\s+)?(?:number\s+of\s+)?pivots?(?:\s+performed)?:?\s+(\d+)", re.IGNORECASE)
RX_DEGENERATE_PIVOTS = re.compile(r"Degenerate:?\s+(\d+)", re.IGNORECASE)
RX_AVG_PIVOT_TIME    = re.compile(r"Average\s+(?:time\s+)?per\s+pivot:?\s+([0-9.]+)\s+(?:milli(?:seconds?)?|ms)", re.IGNORECASE)
RX_LAST_DEGRADATION  = re.compile(r"(?:Last\s+reported\s+)?degradation:?\s+([0-9.eE+-]+)", re.IGNORECASE)
RX_TIGHTENED_BOUNDS  = re.compile(r"(?:Number\s+of\s+)?tightened\s+bounds?:?\s+(\d+)", re.IGNORECASE)
RX_BASIS_REFACTOR    = re.compile(r"(?:Number\s+of\s+)?basis\s+refactorizations?:?\s+(\d+)", re.IGNORECASE)
RX_PRECISION_RESTORATION = re.compile(r"Performing\s+precision\s+restoration", re.IGNORECASE)

# ancillary patterns - also made more flexible
RX_TIMESTAMP      = re.compile(r"(\d{2}:\d{2}:\d{2})\s+Statistics\s+update:?", re.IGNORECASE)
RX_INITIAL_STATS  = re.compile(r"Engine::solve:?\s+Initial\s+statistics", re.IGNORECASE)
RX_BLOCK_START    = re.compile(r"Statistics\s+update:?|Engine::solve:?\s+Initial\s+statistics", re.IGNORECASE)
RX_BLOCK_END      = re.compile(r"^---|^Performing\s+precision\s+restoration|^$")


# ── Safe parsing helper with validation ──────────────────────────────────────
def safe_parse_stat(pattern, line, default=None, parse_type=int, group_idx=1):
    """
    Safely parse a statistic from a line using regex pattern.
    
    Args:
        pattern: Compiled regex pattern
        line: Text line to parse
        default: Default value if parsing fails
        parse_type: Type to convert to (int, float, str)
        group_idx: Which regex group to extract (default 1)
    
    Returns:
        Parsed value or default
    """
    try:
        match = pattern.search(line)
        if match:
            value = match.group(group_idx)
            if parse_type == int:
                return int(value)
            elif parse_type == float:
                return float(value)
            else:
                return value
    except (ValueError, IndexError, AttributeError) as e:
        # Log parsing failure for debugging (only in verbose mode)
        pass
    return default

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
        self.stopped = False
        self.thread = None
        self.pipe_r = None
        self.pipe_w = None
        self.old_stdout_fd = None
        self.old_stderr_fd = None

    def start(self):
        import threading
        
        # Create a pipe for capturing output
        self.pipe_r, self.pipe_w = os.pipe()
        
        # Duplicate the original stdout/stderr file descriptors
        self.old_stdout_fd = os.dup(1)  # stdout
        self.old_stderr_fd = os.dup(2)  # stderr
        
        # Redirect stdout and stderr to the pipe
        os.dup2(self.pipe_w, 1)  # Redirect FD 1 (stdout) to pipe
        os.dup2(self.pipe_w, 2)  # Redirect FD 2 (stderr) to pipe
        
        # Also update Python's sys.stdout/stderr
        self.python_stdout, self.python_stderr = sys.stdout, sys.stderr
        self.capture_stdout = os.fdopen(os.dup(self.pipe_w), 'w', buffering=1)
        self.capture_stderr = os.fdopen(os.dup(self.pipe_w), 'w', buffering=1)
        sys.stdout, sys.stderr = self.capture_stdout, self.capture_stderr

        def read_loop():
            """Read from pipe and emit to GUI in batches"""
            buffer = []
            last_emit = time.time()
            
            with os.fdopen(self.pipe_r, 'r', buffering=1) as reader:
                while not self.stopped:
                    try:
                        line = reader.readline()
                        if not line:
                            break
                        
                        # Write to original stdout so it shows in terminal
                        os.write(self.old_stdout_fd, line.encode())
                        
                        # Buffer for GUI
                        buffer.append(line.rstrip())
                        
                        # Emit batched updates
                        now = time.time()
                        if (now - last_emit) >= 0.05 or len(buffer) >= 50:
                            if buffer:
                                for msg in buffer:
                                    if msg.strip():
                                        self.sig.emit(msg)
                                buffer.clear()
                                last_emit = now
                    except Exception as e:
                        break
                
                # Emit remaining buffer
                for msg in buffer:
                    if msg.strip():
                        self.sig.emit(msg)

        self.thread = threading.Thread(target=read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        # Flush Python streams
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except:
            pass
        
        # Restore original file descriptors
        if self.old_stdout_fd is not None:
            os.dup2(self.old_stdout_fd, 1)
        if self.old_stderr_fd is not None:
            os.dup2(self.old_stderr_fd, 2)

        sys.stdout, sys.stderr = self.python_stdout, self.python_stderr
        self.capture_stdout.close()
        self.capture_stderr.close()
        
        # Close the pipe write end
        if self.pipe_w is not None:
            try:
                os.close(self.pipe_w)
            except:
                pass
        
        # Drain the pipe while the original output descriptors remain open.
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.stopped = True
        for descriptor in (self.old_stdout_fd, self.old_stderr_fd):
            if descriptor is not None:
                os.close(descriptor)



class MarabouWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(float, int, int, int, int, int, float)
    jena_heatmap = pyqtSignal(object)
    timeout_dialog_requested = pyqtSignal(object, float, float, float)
    result = pyqtSignal(object)
    
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
        
        # Track Marabou version for compatibility warnings
        self._marabou_version = None
        self._parsing_failures = 0  # Count failed parses to detect format changes
        self._warned_format_change = False

        # FIX: Remove the duplicate _progress_cb definition and make one clean version
        self._progress_cb = self._emit_progress_signal

    def _emit_result(self, sample_id, code, reason=""):
        self.result.emit(VerificationResult(
            "marabou", str(sample_id), solver_status(code), reason=reason,
            details={"pipeline": self.pipeline, "delta": self._current_delta,
                     "verification_type": self.vt_mode}).to_dict())
    
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
        """
        DEPRECATED: Legacy regex-based statistics parsing.
        
        With verbosity=0, this is skipped entirely and stats come from Python API.
        With verbosity >= 1, this provides backward compatibility for text-based tracking.
        """
        # Skip regex parsing if verbosity is 0 (API-only mode)
        if not hasattr(self, 'verbosity') or self.verbosity == 0:
            return
        
        # Continue with regex parsing for verbosity >= 1
        # Comprehensive statistics collection with ALL Marabou metrics
        
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
        
        # Check for Marabou version (only once)
        if self._marabou_version is None:
            version_match = RX_MARABOU_VERSION.search(line)
            if version_match:
                self._marabou_version = version_match.group(1)
                self.log.emit(f"[STATS] Detected Marabou version: {self._marabou_version}")
        
        # Only parse if we're in a stats block
        if not hasattr(self, '_stats_block_active') or not self._stats_block_active:
            return
    
        # Use safe parsing with validation
        parse_success = False
        
        # Try active PL constraints (special case - 2 groups)
        if 'active_pl' not in self._latest_comprehensive_stats:
            match = RX_ACTIVE_PL.search(line)
            if match:
                try:
                    val1 = int(match.group(1))
                    val2 = int(match.group(2))
                    self._latest_comprehensive_stats['active_pl'] = (val1, val2)
                    parse_success = True
                except (ValueError, IndexError):
                    self._parsing_failures += 1
        
        # Parse single-value statistics using safe helper
        if not parse_success:
            patterns = [
                (RX_SMT_DEPTH, 'smt_depth', int),
                (RX_NUM_SPLITS, 'num_splits', int),
                (RX_TOTAL_TIME, 'total_time_ms', int),
                (RX_MAIN_LOOP_TIME, 'main_loop_time_ms', int),
                (RX_MAIN_LOOP_ITERATIONS, 'main_loop_iterations', int),
                (RX_SIMPLEX_PIVOTS, 'simplex_pivots', int),
                (RX_DEGENERATE_PIVOTS, 'degenerate_pivots', int),
                (RX_AVG_PIVOT_TIME, 'avg_pivot_time_ms', float),
                (RX_LAST_DEGRADATION, 'last_degradation', float),
                (RX_TIGHTENED_BOUNDS, 'tightened_bounds', int),
                (RX_BASIS_REFACTOR, 'basis_refactorizations', int),
            ]
            
            for pattern, key, parse_type in patterns:
                if key not in self._latest_comprehensive_stats:
                    value = safe_parse_stat(pattern, line, default=None, parse_type=parse_type)
                    if value is not None:
                        self._latest_comprehensive_stats[key] = value
                        parse_success = True
                        break
        
        # Check for precision restoration event
        if RX_PRECISION_RESTORATION.search(line):
            current = self._latest_comprehensive_stats.get('precision_restorations', 0)
            self._latest_comprehensive_stats['precision_restorations'] = current + 1
            parse_success = True
        
        # Warn if we're seeing many unparsed statistics lines (format may have changed)
        if not parse_success and not self._warned_format_change:
            # Only count lines that look like they should contain stats
            if any(keyword in line.lower() for keyword in ['number', 'total', 'time', 'depth', 'pivot']):
                self._parsing_failures += 1
                if self._parsing_failures > 10:
                    self.log.emit("⚠️  [STATS] Warning: Many statistics lines couldn't be parsed. "
                                "Marabou output format may have changed. Statistics tracking may be incomplete.")
                    if self._marabou_version:
                        self.log.emit(f"    Detected Marabou version: {self._marabou_version}")
                    self._warned_format_change = True

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

    def _process_stats_from_api(self, stats_list, delta: float):
        """
        Process statistics using the Marabou Python API instead of regex parsing.
        This is the modern, robust way to get statistics.
        """
        if not stats_list:
            return
            
        try:
            from maraboupy import MarabouCore
            
            for stats in stats_list:
                if stats is None:
                    continue
                    
                # Extract statistics using the API
                try:
                    active_pl = stats.getUnsignedAttribute(MarabouCore.NUM_ACTIVE_PL_CONSTRAINTS)
                    total_pl = stats.getUnsignedAttribute(MarabouCore.NUM_PL_CONSTRAINTS) if hasattr(MarabouCore, 'NUM_PL_CONSTRAINTS') else 0
                    depth = stats.getUnsignedAttribute(MarabouCore.CURRENT_DECISION_LEVEL)
                    splits = stats.getUnsignedAttribute(MarabouCore.NUM_SPLITS)
                    pivots = stats.getLongAttribute(MarabouCore.NUM_TABLEAU_PIVOTS)
                    
                    # Log statistics to console
                    self.log.emit(f"[API STATS] Active PL: {active_pl}/{total_pl}, Depth: {depth}, Splits: {splits}, Pivots: {pivots}")
                    
                except AttributeError as e:
                    self.log.emit(f"[API STATS] Warning: Could not extract some statistics: {e}")
                    
        except ImportError:
            self.log.emit("[API STATS] Warning: MarabouCore not available for stats extraction")

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
        
        # Inform user about statistics mode
        if self.verbosity == 0:
            self.log.emit("ℹ️  Verbosity = 0: Using Python API for statistics (no text parsing)")
            self.log.emit("   Statistics will appear as [API STATS] messages")
        elif self.verbosity >= 2:
            self.log.emit("⚠️  Verbosity = 2: High output volume may slow GUI")
            self.log.emit("   Consider verbosity=0 to use Python API stats instead")
        
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

                    sat, unsat, to, stats_list = run_tof(
                        json_cfg      = tmpf.name,
                        delta         = delta,
                        perturb_all   = self.perturb_all,
                        progress_callback = self._progress_cb,
                        verbosity     = self.verbosity,
                        marabou_opts  = mopts,
                        stop_requested=self.isInterruptionRequested,
                        result_callback=self._emit_result
                    )

                    # Process statistics from API
                    self._process_stats_from_api(stats_list, delta)

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
                            img = Image.open(img_path)
                            if img.mode not in ('L', 'RGB'):
                                img = img.convert('RGB')
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
                            report, is_robust, img_stats = local_rob_sens(
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
                                stop_requested=self.isInterruptionRequested,
                                perturb_all_pixels=self.perturb_all,
                                selected_idxs=self.selected_idxs
                            )
                        else:
                            report, is_robust, img_stats = targeted_rob_sens(
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

                        self._emit_result(filename, is_robust)

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
                    self.log.emit(f"δ={delta}: SAT={delta_sat}, UNSAT={delta_unsat}, INCONCLUSIVE={delta_timeout}")
                
                    
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

                    combo, jena_stats = run_jena(
                        json_path         = tmpf.name,
                        delta_orig        = delta,
                        epsilon_orig      = eps or None,
                        perturb_all       = self.perturb_all,
                        selected_flat_idxs= self.selected_idxs,
                        plot              = False,
                        marabou_opts      = mopts,
                        stop_requested    = self.isInterruptionRequested,
                        progress_callback  = self._progress_cb,
                        result_callback    = self._emit_result
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
                    plt.bar(x, to_pct,     width, bottom=sat_pct+unsat_pct, color='darkgrey', label='INCONCLUSIVE')
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
        except Exception as exc:
            self.log.emit("Error in worker:\n" + traceback.format_exc())
            self._emit_result(self.model_file, "error", str(exc))
        finally:
            self._comprehensive_history.clear()
            self._latest_comprehensive_stats.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Restore stdout/stderr
            fd_redir.stop()
            sys.stdout, sys.stderr = old_py_out, old_py_err

