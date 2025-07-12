# StabilEdge: Formal Verification of Deep Neural Networks

StabilEdge is a verification framework that extends the Marabou solver with a high‑level workflow and graphical interface. It simplifies the specification of verification problems, monitors solver progress, and visualises results, thereby lowering the entry barrier for rigorous analysis of neural networks in safety‑critical applications.

---

## Features

| Category | Description |
| -------- | ----------- |
|          |             |

| **Verification Modes**        | Four distinct verification procedures (see accompanying paper) covering local and targeted robustness analyses.                        |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **GUI**                       | Cross‑platform graphical interface with integrated Netron viewer.                                                                      |
| **Flexible Input**            | Accepts ONNX models, data sets, and a Feature JSON file specifying task type, domain constraints, and perturbation settings.           |
| **Interactive Configuration** | Users select pixel or time‑series features to perturb, set delta (δ) and epsilon (ε) thresholds, and choose single or multiple values. |
| **Progress Monitoring**       | Real‑time progress bar tracking remaining SAT/UNSAT instances; optional visualisation of results on completion.                        |
| **Adaptive Timeout**          | Extended Marabou timeout that analyses solver statistics and recommends when to prolong or disable the limit.                          |

---

## Installation

1. **Obtain Marabou**

```bash
git clone https://github.com/NeuralNetworkVerification/Marabou.git
cd Marabou
./scripts/install.sh   # Follow Marabou documentation for build options
```

2. **Clone StabilEdge**

```bash
git clone https://github.com/your‑organisation/StabilEdge.git
```

3. **Launch the GUI**

```bash
cd StabilEdge
./run_gui.sh           # Starts the graphical interface
```

A pre‑compiled standalone executable is available for users who prefer not to build from source.

---

## Usage Overview

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

---

## Command‑Line Options

```bash
# Verify without the GUI
python3 stabil_edge_cli.py --model network.onnx --json feature.json \
                           --dataset inputs.csv --delta 0.03 --epsilon 0.05
```

---

## Citation

If you use StabilEdge in academic work, please cite:

```bibtex
@inproceedings{StabilEdge2025,
  title     = {StabilEdge: Formal Verification of Deep Neural Networks},
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

