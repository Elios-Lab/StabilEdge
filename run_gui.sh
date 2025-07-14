#!/bin/bash

# Adjust this to your actual conda environment name
ENV_NAME="verify_gui"
PYTHON_SCRIPT="StableNN_GUI.py"  # Change to your actual script if different

# Source conda setup (if not already available)
if ! command -v conda &> /dev/null; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi

# Activate the environment
conda activate "$ENV_NAME"

# Compute PyQt5's Qt path inside the environment
PYQT5_QT_LIB="$CONDA_PREFIX/lib/python3.8/site-packages/PyQt5/Qt5/lib"
PLUGIN_PATH="$PYQT5_QT_LIB/plugins/platforms"

# Isolate Qt plugins to avoid ROS/CUDA/Carla conflicts
export LD_LIBRARY_PATH="$PYQT5_QT_LIB"
export QT_QPA_PLATFORM_PLUGIN_PATH="$PLUGIN_PATH"

# Force Qt to use software rendering to prevent OpenGL issues
export QT_OPENGL=software
export QT_QUICK_BACKEND=software
export QT_XCB_GL_INTEGRATION=none

# Optional: Debug prints
echo "Using Python: $(which python)"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "QT_QPA_PLATFORM_PLUGIN_PATH=$QT_QPA_PLATFORM_PLUGIN_PATH"

# Run your GUI
python "$PYTHON_SCRIPT"

