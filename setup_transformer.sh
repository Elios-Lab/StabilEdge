#!/usr/bin/env bash
set -e -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${STABILEDGE_TRANSFORMER_REPO:-$(dirname -- "$SCRIPT_DIR")/Transformer-Position-Scheduling-Certification}"
BACKEND_REVISION="aed9f862ed02c7d5e12aa5eb6ddb66b7538cc121"
SETUP_PYTHON="${STABILEDGE_SETUP_PYTHON:-python3}"

"$SETUP_PYTHON" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 11), "Use Python 3.10 or 3.11 for this environment."'
if [[ ! -d "$BACKEND_DIR" ]]; then
    git clone https://github.com/Elios-Lab/Transformer-Position-Scheduling-Certification.git "$BACKEND_DIR"
    git -C "$BACKEND_DIR" checkout "$BACKEND_REVISION"
fi
if [[ ! -f "$BACKEND_DIR/experiments/cascade_verification.py" ]]; then
    echo "The backend folder does not contain the expected Transformer repository." >&2
    exit 1
fi

# Avoid inheriting ROS packages or the GUI's Qt library paths while installing.
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH
"$SETUP_PYTHON" -m venv "$SCRIPT_DIR/.venv-transformer"
"$SCRIPT_DIR/.venv-transformer/bin/python" -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
"$SCRIPT_DIR/.venv-transformer/bin/python" -m pip install -r "$SCRIPT_DIR/requirements-transformer.txt"
"$SCRIPT_DIR/.venv-transformer/bin/python" -m pip check
echo "Transformer environment ready: $SCRIPT_DIR/.venv-transformer/bin/python"
echo "Backend source: $BACKEND_DIR"
