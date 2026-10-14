#!/bin/bash

set -e -o pipefail

ENV_NAME="verify_gui"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# conda activate needs the shell function, even when conda is on PATH.
if ! declare -F conda > /dev/null; then
    if command -v conda > /dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    else
        CONDA_BASE=""
        for candidate in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" /opt/conda; do
            if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
                CONDA_BASE="$candidate"
                break
            fi
        done
    fi
    if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        echo "Conda not found. Install Conda or add it to PATH before launching StabilEdge." >&2
        exit 1
    fi
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda activate "$ENV_NAME"

# Keep inherited ROS/OpenCV/system Qt paths out of this application.
# Mixing system Qt libraries with PyQt5's bundled libraries breaks xcb.
unset LD_LIBRARY_PATH QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH
QT_LIB_DIR="$(python -c 'from PyQt5.QtCore import QLibraryInfo; print(QLibraryInfo.location(QLibraryInfo.LibrariesPath))')"
QT_PLUGINS_DIR="$(python -c 'from PyQt5.QtCore import QLibraryInfo; print(QLibraryInfo.location(QLibraryInfo.PluginsPath))')"
export LD_LIBRARY_PATH="$QT_LIB_DIR"
export QT_PLUGIN_PATH="$QT_PLUGINS_DIR"
export QT_QPA_PLATFORM_PLUGIN_PATH="$QT_PLUGINS_DIR/platforms"
export PYTHONPATH="$SCRIPT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"

# Force Qt to use software rendering to prevent OpenGL issues
export QT_OPENGL=software
export QT_QUICK_BACKEND=software
export QT_XCB_GL_INTEGRATION=none

# Use the repository directory regardless of where the launcher was invoked.
cd "$SCRIPT_DIR"
exec python "$SCRIPT_DIR/StabilEdge_GUI.py"

