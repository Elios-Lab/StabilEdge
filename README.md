# StabilEdge: Formal Verification of Deep Neural Networks

StabilEdge is a verification framework that extends the Marabou solver with a high‑level workflow and graphical interface. It simplifies the specification of verification problems, monitors solver progress, and visualises results, thereby lowering the entry barrier for rigorous analysis of neural networks in safety‑critical applications.

---


## Features

| Category                   | Description                                                                                                                                     |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| **Verification Modes**    | Four distinct verification procedures (see accompanying paper) covering local and targeted robustness analyses.                                |
| **GUI**                   | Cross‑platform graphical interface with integrated Netron viewer.                                                                              |
| **Flexible Input**        | Accepts ONNX models, data sets, and a Feature JSON file specifying task type, domain constraints, and perturbation settings.                   |
| **Interactive Configuration** | Users select pixel or time‑series features to perturb, set delta (δ) and epsilon (ε) thresholds, and choose single or multiple values.     |
| **Progress Monitoring**   | Real‑time progress bar tracking remaining SAT/UNSAT instances; optional visualisation of results on completion.                                |
| **Adaptive Timeout**      | Extended Marabou timeout that analyses solver statistics and recommends when to prolong or disable the limit.                                  |


---

## Installation

## 1  Obtain Marabou

```bash
git clone https://github.com/NeuralNetworkVerification/Marabou.git
# Follow Marabou documentation for build options
```

## 2  Clone StabilEdge

```bash
git clone https://github.com/your‑organisation/StabilEdge.git
```

## 3  Create & Activate the Conda Environment

You’ll need a Conda environment called **`verify_gui`** with all required dependencies.

```bash
# Using the provided environment file
cd /path/to/StabilEdge
conda env create -f environment.yml
conda activate verify_gui
```

<details>
<summary><strong>Manual install (alternative)</strong></summary>

```bash
conda create -n verify_gui python=3.8 pyqt matplotlib ...
conda activate verify_gui
```
</details>

## 4  Run the GUI

### 4.1 Method A — Command‑Line

```bash
cd StabilEdge
chmod +x run_gui.sh
./run_gui.sh
```

The script automatically

* activates the correct Conda environment  
* sets the necessary Qt plugin paths  
* forces software rendering to avoid OpenGL / ROS / CUDA conflicts  
* launches **`StabilEdge_GUI.py`**

Run this in an Ubuntu desktop session. The launcher initializes Conda if
needed and adds `scripts/` to Python's import path. It also isolates the
application's Qt libraries and plugins from inherited ROS, OpenCV, and
system Qt paths, which can otherwise cause an `xcb` plugin startup crash.
Qt paths are detected from the active environment rather than hard-coded.

If the `verify_gui` environment already exists, you can skip environment
creation and launch the application directly. A Marabou warning that the
TensorFlow parser is unavailable does not prevent use of ONNX models.

---

### 4.2 Method B — Double‑Click App Icon *(optional)*

1. **Right‑click** `run_gui.sh` → **Properties** → **Permissions**  
2. ✔️ Enable **“Allow executing file as program”**  
3. Double‑click `run_gui.sh` and choose **Run**

> 💡 Some desktop environments need a `.desktop` file for a true menu launcher (see below).

## 5  [Optional] Create a Desktop App Launcher

Create `StabilEdge.desktop`:

```desktop
[Desktop Entry]
Type=Application
Version=1.0
Name=StabilEdge
Comment=Formal verification GUI for ONNX models
Exec=sh -c "cd /path/to/StabilEdge && ./run_gui.sh"
Icon=/path/to/StabilEdge/icon.png
Terminal=false
Categories=Development;Science;
StartupWMClass=StabilEdge_GUI.py
```

Move it to:

```bash
mv StabilEdge.desktop ~/.local/share/applications/
```

## Usage Overview

The GUI supports **Marabou** and **Transformer Bounds** through the verification
engine selector. Marabou provides the ONNX workflows described below. Transformer
Bounds adds binary text-classifier certification, token previews, certified-radius
search, and position scheduling with validation/fallback in an isolated Python
process. See [Transformer engine setup and usage](docs/transformer-engine.md).

```bash
# Optional: install the CPU Transformer environment and backend source
./setup_transformer.sh
```

1. **Import the Model**: Load an ONNX network; inspect its architecture via the embedded Netron viewer.
2. **Load Metadata**: Supply the Feature JSON describing task type (classification/regression), number of classes, input domain, and perturbation options. Example files are provided in `examples/`.
3. **Configure Analysis**: Set δ values for input perturbations and ε thresholds for acceptable output deviation. For images, select specific pixels; for time‑series data, choose features and time steps via the grid.
4. **Run Verification**: Monitor progress in real time. When the timeout is near, the system evaluates solver statistics and issues recommendations.
5. **Inspect Results**: Visualise outcomes through interactive plots or export logs for further analysis.

---

## Input File Specification

- **ONNX Model** – Neural network to be verified.
- **Dataset** – Input samples used during verification (CSV/NPZ supported).
- **Feature JSON** – Contains:
  - `task`: `classification` | `regression`
  - `num_classes`: integer (for classification)
  - `perturbation`: domain constraints and pixel/feature selections
  - Additional optional fields (see examples).



## Citation

If you use StabilEdge in academic work, please cite:

```bibtex
@inproceedings{StabilEdge2025,
  title     = {StabilEdge:},
  author    = {Author Names},
  booktitle = {Proceedings of the XYZ Conference},
  year      = {2025}
}
```

---

## License

StabilEdge is released under the MIT License. See `LICENSE` for details.

---

## Acknowledgements

This project builds upon the Marabou verification engine and the Netron network visualiser. We thank their respective contributors for making their software available to the community.

