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
from PyQt5.QtGui    import QIcon, QFont, QColor, QLinearGradient
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QComboBox,
    QCheckBox, QTreeWidget, QTreeWidgetItem,
    QMessageBox, QFileDialog, QPlainTextEdit, QSplitter,
    QSpinBox, QDoubleSpinBox, QTabWidget, QGridLayout, QMenu, QAction, QToolButton, QWidgetAction,
    QSizePolicy, QProgressBar, QGraphicsDropShadowEffect, QStackedWidget
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
from verification.registry import ENGINES, create_marabou_worker
from verification.transformer_panel import TransformerPanel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_MAP = {
    "ITD":  os.path.join(BASE_DIR, "ITD_features.json"),
    "TOF":  os.path.join(BASE_DIR, "TOF_features.json"),
    "Jena": os.path.join(BASE_DIR, "Jena_features.json"),
}


def _get_input_indices(feature_props):
    """Get input indices from feature properties, supporting both old and new naming."""
    # Try new naming first, fall back to old naming for backward compatibility
    return feature_props.get("input_indices") or feature_props.get("starting_indexes", [])



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
        self.btnExportConfig        = QPushButton("Export Config…")
        self.btnImportConfig        = QPushButton("Import Config…")
        self.btnExportConfig.setEnabled(False)  # Enable only when selections exist
        self.btnImportConfig.setEnabled(True)

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
        file_row.addWidget(self.btnExportConfig)
        file_row.addWidget(self.btnImportConfig)

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
        self.spinVerb.setValue(0)  # Default to API-only mode
        # Tell users in the tooltip or label that 2 is required for progress notifications.
        self.spinVerb.setToolTip(
            "0 = Silent (API stats only - RECOMMENDED, prevents text overflow)\n"
            "1 = Minimal output\n"
            "2 = Full logs (may slow GUI with high output volume)\n\n"
            "💡 All statistics available via Python API regardless of verbosity"
        )
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
        self.resultTree = QTreeWidget()
        self.resultTree.setHeaderLabels(["Sample", "Status", "δ", "Details"])
        self.tab_widget.addTab(self.resultTree, "Verification results")
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
        self.pgPlot.showGrid(x=False, y=False)  # Disable all grid lines for cleaner look
        
        # Add modern styling to the plot widget
        self.pgPlot.setBackground('w')  # White background for better visibility
        
        # Add subtle glow effect to the plot widget
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(100, 200, 255, 80))
        shadow.setOffset(0, 0)
        self.pgPlot.setGraphicsEffect(shadow)


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
        self.engineSelector = QComboBox()
        for engine in ENGINES:
            self.engineSelector.addItem(engine.name, engine.id)
        self.engineSelector.setToolTip("Choose the verification backend. Each backend exposes its supported model and input types.")
        engine_row = QHBoxLayout()
        engine_row.addWidget(QLabel("Verification engine:"))
        engine_row.addWidget(self.engineSelector)
        engine_row.addStretch()
        self.enginePages = QStackedWidget()
        self.enginePages.addWidget(central)
        self.transformerPanel = TransformerPanel(self)
        self.enginePages.addWidget(self.transformerPanel)
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.addLayout(engine_row)
        outer_layout.addWidget(self.enginePages)
        self.engineSelector.currentIndexChanged.connect(self.enginePages.setCurrentIndex)
        self.transformerPanel.running_changed.connect(lambda running: self.engineSelector.setEnabled(not running))
        self.setCentralWidget(outer)

    def _connect_signals(self):
        self.btnModel.clicked.connect(self._pick_model)
        self.btnInput.clicked.connect(self._pick_input)
        self.btnJson.clicked.connect(self._pick_json)
        self.btnExportConfig.clicked.connect(self._export_config)
        self.btnImportConfig.clicked.connect(self._import_config)
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
            # Batch console updates to reduce GUI freezing with high verbosity
            cursor = self.console.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText(msg + '\n')
            # Only auto-scroll if already at bottom
            scrollbar = self.console.verticalScrollBar()
            if scrollbar.value() >= scrollbar.maximum() - 10:
                self.console.setTextCursor(cursor)
                self.console.ensureCursorVisible()

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
                    else self.selected_classes.discard(idx),
                    self._update_export_button_state()
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

        # Prefer explicit configuration fields; retain legacy filenames.
        base = os.path.basename(fn)
        configured = str(cfg.get("pipeline", cfg.get("name", ""))).upper()
        if configured in ("ITD", "TOF", "JENA"):
            self.pipeline = "Jena" if configured == "JENA" else configured
        elif cfg.get("task") == "regression":
            self.pipeline = "Jena"
        elif base == "ITD_features.json":
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
        # T = number of positions/timesteps per feature in the flattened input vector
        # These positions represent where this feature's data appears in the input array
        T = len(_get_input_indices(feats[0]["properties"]))

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
            # Create button for each position this feature occupies in the flattened input
            # idx = position in the input vector, col = visual column (timestep) in the grid
            for col, idx in enumerate(_get_input_indices(feat["properties"]), start=1):
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
                        else self.selected_pixels.discard(i),
                        self._update_export_button_state()
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
                # Add child items for each input position this feature occupies
                for idx in _get_input_indices(feat["properties"]):
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
                        else self.selected_pixels.discard(i),
                        self._update_export_button_state()
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
            from scripts.TOF import compute_meaningful_mask
            
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
            self._update_export_button_state()
            
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
        self._update_export_button_state()

    def _update_export_button_state(self):
        """Enable export button only when there are selections to save"""
        has_config = hasattr(self, 'feature_cfg') and self.feature_cfg
        has_selections = bool(self.selected_pixels or self.selected_classes)
        self.btnExportConfig.setEnabled(has_config and has_selections)

    def _export_config(self):
        """Export the current configuration with selected pixels/features to a JSON file"""
        # Validate that we have a configuration to export
        if not hasattr(self, 'feature_cfg') or not self.feature_cfg:
            QMessageBox.warning(self, "Error", "Load a features JSON first.")
            return
        
        if not self.selected_pixels and not self.selected_classes:
            QMessageBox.warning(
                self, "No Selection", 
                "No pixels or classes are selected. Please make selections before exporting."
            )
            return
        
        # Ask user where to save the config
        fn, _ = QFileDialog.getSaveFileName(
            self,
            "Export Configuration",
            "",
            "JSON Files (*.json)"
        )
        
        if not fn:
            return
        
        # Ensure .json extension
        if not fn.lower().endswith('.json'):
            fn += '.json'
        
        # Build the export configuration
        export_cfg = dict(self.feature_cfg)  # Copy base config
        
        # Add metadata section
        export_cfg["_metadata"] = {
            "engine": "marabou",
            "schema_version": 1,
            "exported_from": "StableNN_GUI",
            "pipeline": getattr(self, "pipeline", "unknown"),
            "verification_type": self.comboVT.currentText(),
            "model_file": self.leModel.text().strip(),
            "input_file": self.leInput.text().strip(),
            "input_shape": self.input_shape if hasattr(self, 'input_shape') else None
        }
        
        # Add selected pixels/indices
        if self.selected_pixels:
            export_cfg["_metadata"]["selected_pixels"] = sorted(list(self.selected_pixels))
        
        # Add selected classes for targeted modes
        if self.selected_classes:
            export_cfg["_metadata"]["selected_classes"] = sorted(list(self.selected_classes))
        
        # Write to file
        try:
            with open(fn, 'w', encoding='utf-8') as f:
                json.dump(export_cfg, f, indent=2)
            
            QMessageBox.information(
                self, "Success",
                f"Configuration exported successfully to:\n{fn}\n\n"
                f"Selected pixels: {len(self.selected_pixels)}\n"
                f"Selected classes: {len(self.selected_classes)}"
            )
            self.console.appendPlainText(f"Configuration exported to: {fn}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export configuration:\n{e}")

    def _import_config(self):
        """Import a previously exported configuration and restore selections"""
        # Ask user to select a config file
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Import Configuration",
            "",
            "JSON Files (*.json)"
        )
        
        if not fn:
            return
        
        try:
            # Load the configuration
            with open(fn, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            
            # Check if this is an exported config with metadata
            if "_metadata" not in cfg:
                QMessageBox.warning(
                    self, "Invalid Config",
                    "This JSON file does not appear to be an exported configuration.\n"
                    "Use 'Import Features JSON' for base feature files."
                )
                return
            
            metadata = cfg["_metadata"]
            
            # Restore the base feature configuration
            base_cfg = {k: v for k, v in cfg.items() if k != "_metadata"}
            self.feature_cfg = base_cfg
            self.json_path = fn
            
            # Restore pipeline
            if "pipeline" in metadata:
                self.pipeline = metadata["pipeline"]
            
            # Restore model and input paths if they exist
            if metadata.get("model_file") and os.path.exists(metadata["model_file"]):
                self.leModel.setText(metadata["model_file"])
                # Trigger model loading
                self._load_model_internal(metadata["model_file"])
            
            if metadata.get("input_file") and os.path.exists(metadata["input_file"]):
                self.leInput.setText(metadata["input_file"])
                if metadata.get("input_shape"):
                    self.input_shape = tuple(metadata["input_shape"])
            
            # Restore verification type
            if metadata.get("verification_type"):
                idx = self.comboVT.findText(metadata["verification_type"])
                if idx >= 0:
                    self.comboVT.setCurrentIndex(idx)
            
            # Determine if we need to build a grid
            is_sensitivity = self.comboVT.currentText().endswith("Sensitivity")
            has_pixel_features = any(
                "pixel" in f.get("name", "").lower() 
                for f in base_cfg.get("features", [])
            )
            
            # Build the appropriate grid if in sensitivity mode
            if is_sensitivity:
                if has_pixel_features and self.input_shape:
                    rows, cols = self.input_shape
                    self._build_pixel_grid(rows, cols)
                else:
                    self._build_feature_timestep_grid()
            
            # Restore selected pixels/indices
            self.selected_pixels.clear()
            if "selected_pixels" in metadata:
                selected_idxs = metadata["selected_pixels"]
                self.selected_pixels.update(selected_idxs)
                
                # Update grid buttons if they exist
                if has_pixel_features and self.input_shape:
                    rows, cols = self.input_shape
                    for idx in selected_idxs:
                        row = idx // cols
                        col = idx % cols
                        button = self.gridLayout.itemAtPosition(row, col)
                        if button and button.widget():
                            button.widget().setChecked(True)
                else:
                    # For time-series features, just mark them as selected
                    # The grid checkboxes will be updated when grid is rebuilt
                    pass
            
            # Restore selected classes
            self.selected_classes.clear()
            if "selected_classes" in metadata:
                self.selected_classes.update(metadata["selected_classes"])
                
                # Update class checkboxes if menu exists
                if hasattr(self, 'btnClassMenu') and self.btnClassMenu.menu():
                    menu = self.btnClassMenu.menu()
                    for action in menu.actions():
                        widget = menu.widgetForAction(action)
                        if isinstance(widget, QCheckBox) and not widget.text().startswith("All"):
                            # Extract class index from text like "Class 0"
                            try:
                                class_idx = int(widget.text().split()[-1])
                                if class_idx in self.selected_classes:
                                    widget.setChecked(True)
                            except (ValueError, IndexError):
                                pass
            
            # Update export button state
            self._update_export_button_state()
            
            # Show success message
            msg_parts = [f"Configuration imported from:\n{fn}"]
            if "selected_pixels" in metadata:
                msg_parts.append(f"Restored {len(metadata['selected_pixels'])} selected pixels")
            if "selected_classes" in metadata:
                msg_parts.append(f"Restored {len(metadata['selected_classes'])} selected classes")
            
            QMessageBox.information(self, "Success", "\n".join(msg_parts))
            self.console.appendPlainText(f"Configuration imported from: {fn}")
            
        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "JSON Parse Error", f"Could not parse JSON:\n{e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import configuration:\n{e}\n{traceback.format_exc()}")

    def _load_model_internal(self, model_path):
        """Internal method to load a model without user interaction"""
        try:
            model = onnx.load(model_path)
            
            # Build outline tree (simplified version)
            self.modelTree.clear()
            inputs_item = QTreeWidgetItem(["Inputs"])
            for inp in model.graph.input:
                inputs_item.addChild(QTreeWidgetItem([inp.name]))
            self.modelTree.addTopLevelItem(inputs_item)
            
            outputs_item = QTreeWidgetItem(["Outputs"])
            for out in model.graph.output:
                outputs_item.addChild(QTreeWidgetItem([out.name]))
            self.modelTree.addTopLevelItem(outputs_item)
            
            # Detect multiclass
            out_type = model.graph.output[0].type.tensor_type
            dims = [d.dim_value for d in out_type.shape.dim if d.dim_value > 0]
            num_classes = dims[-1] if dims else 0
            self.has_multiclass = (num_classes > 2)
            
        except Exception as e:
            self.console.appendPlainText(f"Warning: Could not fully load model: {e}")

    def _start(self):
        # 1) Must have JSON loaded
        if not self.feature_cfg:
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
            if not deltas or any(not np.isfinite(value) or value < 0 for value in deltas):
                raise ValueError("Delta values must be finite and nonnegative.")
            self._deltas = deltas
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid δ list")
            return

        # 4) Parse εs only for Jena; for ITD/TOF always none
        if pipeline == "Jena":
            try:
                epsilons = [float(x) for x in self.leEpsilons.text().split(",") if x.strip()]
                if not epsilons or any(not np.isfinite(value) or value <= 0 for value in epsilons):
                    raise ValueError("Output deviations must be finite and positive.")
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
                    if any(value < 0 for value in timeouts):
                        raise ValueError("Timeouts must be nonnegative.")
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
        self.engineSelector.setEnabled(False)
        self.resultTree.clear()
        
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
            self.btnRun.setEnabled(True)
            self.btnStop.setEnabled(False)
            self.engineSelector.setEnabled(True)
            self._elapsed_timer.stop()
            return

        # 10) Spawn the background worker
        try:
            worker = create_marabou_worker(
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
        except (ImportError, RuntimeError) as exc:
            self.btnRun.setEnabled(True)
            self.btnStop.setEnabled(False)
            self.engineSelector.setEnabled(True)
            self._elapsed_timer.stop()
            QMessageBox.warning(self, "Marabou engine", str(exc))
            return
        worker.log.connect(self._handle_log, Qt.QueuedConnection)
        worker.result.connect(self._handle_engine_result, Qt.QueuedConnection)
        worker.progress.connect(self._update_progress, Qt.QueuedConnection)
        worker.finished.connect(lambda: self._on_finished(worker), Qt.QueuedConnection)
        worker.timeout_dialog_requested.connect(self._handle_timeout_dialog, Qt.QueuedConnection)
        self.threads.append(worker)
        worker.start()

    def _handle_engine_result(self, result):
        details = result.get("details", {})
        QTreeWidgetItem(self.resultTree, [result["sample_id"], result["status"],
                                        str(details.get("delta", "")), result.get("reason", "")])
        self.console.appendPlainText("%s: %s" % (result["sample_id"], result["status"]))


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
    
        # 4) Draw your stacked‐bar segments with modern gradients and styling...
        if total == 0:
            sat_frac = unsat_frac = timeout_frac = 0
        else:
            sat_frac     = sat   / total
            unsat_frac   = unsat / total
            timeout_frac = timeout / total
    
        # Modern color scheme with gradients (using solid for PyQtGraph compatibility)
        widths      = [sat_frac, unsat_frac, timeout_frac]
        # Rich, modern colors: SAT (red gradient), UNSAT (green gradient), TIMEOUT (amber gradient)
        colors      = ['#d62728', '#1f77b4', '#ff7f0e']
        labels      = [f"SAT: {sat}", f"UNSAT: {unsat}", f"INCONCLUSIVE: {timeout}"]
        text_colors = ['white', 'white', 'white']
    
        # Position bar lower to make room for info above
        bar_y_center = 0.35
        bar_height = 0.5
        
        x0 = 0
        for w, c, lbl, tc in zip(widths, colors, labels, text_colors):
            if w > 0:
                # Create bar with increased height for better visibility
                bar = pg.BarGraphItem(x0=x0, y=bar_y_center, height=bar_height, width=w, brush=c)
                self.pgPlot.addItem(bar)
                
                # Add subtle separator line between segments for better definition
                if x0 > 0:
                    separator = pg.PlotDataItem([x0, x0], [bar_y_center - bar_height/2, bar_y_center + bar_height/2], 
                                              pen=pg.mkPen(color='#2a2a3e', width=2))
                    self.pgPlot.addItem(separator)
                
                # Add text label with better styling
                ti = pg.TextItem(lbl, color=tc, anchor=(0.5,0.5))
                ti.setFont(QFont("Arial", 12, QFont.Bold))
                ti.setPos(x0 + w/2, bar_y_center)
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
        
        # Style the bottom axis with modern look
        bottom_axis = self.pgPlot.getAxis('bottom')
        bottom_axis.setStyle(
            tickFont=QFont("Arial", 20, QFont.Bold),
            tickLength=8  # Longer tick marks for better visibility
        )
        # Dark color for white background
        bottom_axis.setPen(pg.mkPen(color='#333333', width=2))
        bottom_axis.setTextPen(pg.mkPen(color='#000000'))
        
        # Add sleek vertical ruler lines from labels up to progress bar
        ruler_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        bar_bottom = 0.35 - 0.25  # bar_y_center - bar_height/2
        for x_pos in ruler_positions:
            # Create gradient-like effect with darker lines at key positions
            line_color = '#333333' if x_pos in [0.0, 0.5, 1.0] else '#999999'
            line_width = 2 if x_pos in [0.0, 0.5, 1.0] else 1
            
            ruler_line = pg.PlotDataItem([x_pos, x_pos], [-0.05, bar_bottom], 
                                       pen=pg.mkPen(color=line_color, width=line_width))
            self.pgPlot.addItem(ruler_line)
        
        # Add a subtle bottom border line for polish
        border_line = pg.PlotDataItem([0, 1], [bar_bottom, bar_bottom], 
                                     pen=pg.mkPen(color='#999999', width=1, style=Qt.DotLine))
        self.pgPlot.addItem(border_line)
        
        # 5) "Run Overview" section with table layout
        if hasattr(self, "_deltas"):
            try:
                idx = self._deltas.index(delta) + 1
                total_d = len(self._deltas)
                run_info = f"run {idx}/{total_d}"
            except ValueError:
                run_info = "run 1/1"
        else:
            run_info = "run 1/1"
    
        # Left side - Run Overview section (aligned with bar start)
        overview_title = pg.TextItem("Run Overview", color='#1976d2', anchor=(0, 1))
        overview_title.setFont(QFont("Arial", 36, QFont.Bold))
        overview_title.setPos(0.0, 0.93)
        self.pgPlot.addItem(overview_title)
        
        delta_label = pg.TextItem(f"δ = {delta}", color='#424242', anchor=(0, 1))
        delta_label.setFont(QFont("Arial", 20, QFont.Bold))
        delta_label.setPos(0.0, 0.85)
        self.pgPlot.addItem(delta_label)
        
        run_label = pg.TextItem(run_info, color='#424242', anchor=(0, 1))
        run_label.setFont(QFont("Arial", 20, QFont.Bold))
        run_label.setPos(0.0, 0.78)
        self.pgPlot.addItem(run_label)
        
        # Right side - Progress section
        progress_title = pg.TextItem("Progress", color='#2e7d32', anchor=(0.5, 1))
        progress_title.setFont(QFont("Arial", 20, QFont.Bold))
        progress_title.setPos(0.80, 0.95)
        self.pgPlot.addItem(progress_title)
        
        progress_value = pg.TextItem(f"{done}/{total}", color='#2e7d32', anchor=(0.5, 1))
        progress_value.setFont(QFont("Arial", 20, QFont.Bold))
        progress_value.setPos(0.80, 0.87)
        self.pgPlot.addItem(progress_value)
        
        # Far right - Time elapsed section
        time_title = pg.TextItem("Time elapsed", color='#d32f2f', anchor=(0.5, 1))
        time_title.setFont(QFont("Arial", 20, QFont.Bold))
        time_title.setPos(0.95, 0.95)
        self.pgPlot.addItem(time_title)
        
        time_value = pg.TextItem(f"{elapsed_str}", color='#d32f2f', anchor=(0.5, 1))
        time_value.setFont(QFont("Arial", 20, QFont.Bold))
        time_value.setPos(0.97, 0.87)
        self.pgPlot.addItem(time_value)
        
        # Add estimate if available (below time)
        if est_text:
            est_label = pg.TextItem(est_text.strip(), color='#9e9e9e', anchor=(0.5, 1))
            est_label.setFont(QFont("Arial", 10, QFont.Italic))
            est_label.setPos(0.97, 0.81)
            self.pgPlot.addItem(est_label)

        


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
        self.engineSelector.setEnabled(True)

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
            thr.result.disconnect()
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
        self.transformerPanel.controller.shutdown()
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

