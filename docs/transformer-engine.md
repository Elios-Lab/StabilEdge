# Transformer certification in StabilEdge

The GUI now has a **Verification engine** selector. Marabou uses the existing
ONNX workflows. **Transformer Bounds** uses the custom binary text-classifier
models and verifiers from
[Transformer-Position-Scheduling-Certification](https://github.com/Elios-Lab/Transformer-Position-Scheduling-Certification).
The integration was tested with source revision
`aed9f862ed02c7d5e12aa5eb6ddb66b7538cc121`.

## Launch and setup

On this workspace, the CPU environment has already been installed in
`.venv-transformer`, and the backend source is in the sibling repository folder.
Launch the GUI normally:

```bash
cd /home/vafali/StabilEdge
./run_gui.sh
```

Select **Transformer Bounds**. The setup fields automatically select the local
backend folder and `.venv-transformer/bin/python` when present.

For another Ubuntu machine with Python 3.10 or 3.11:

```bash
./setup_transformer.sh
```

This script clones the tested source revision if the backend folder is absent
and installs a separate CPU environment. It leaves an existing backend checkout
in place. `STABILEDGE_TRANSFORMER_REPO` can select a different backend location,
and `STABILEDGE_SETUP_PYTHON` can select the Python used to create the environment.
Set the GUI's Backend folder to the chosen location.

Alternatively, use the portable Ubuntu Conda environment supplied by StabilEdge:

```bash
conda env create -f environment-transformer.yml
```

Then choose the Python executable inside `pbverify` in the GUI. The original
upstream environment files mix macOS/Python 3.6 and Linux/Python 3.10 exports;
the integration does not use them.

For GPU execution, install a compatible CUDA build of PyTorch 2.5.1 in a separate
environment using the [official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/),
then install `requirements-transformer.txt` there. Select that interpreter and
enter a physical device such as `cuda:0`. The runner isolates the chosen GPU
before importing PyTorch. Only CPU execution has been tested in this workspace.

## Checkpoints and inputs

Choose a checkpoint folder containing:

```text
pytorch_model.bin
bert_config.json  (or config.json)
vocab.txt
```

A parent folder with an upstream `checkpoint` marker pointing to `ckpt-N` is
also accepted. Weight loading is strict: missing or mismatched parameters cause
an error instead of silently initializing part of the model. The model must use
the upstream custom BERT/Transformer architecture with two classifier outputs.
General ONNX models, arbitrary Hugging Face models, multiclass models, and
regression models are not supported by this backend.

The upstream repository excludes the paper checkpoints. Obtain those separately
following its `ARTIFACT_CHECKPOINTS.md`, or use a compatible checkpoint you have
trained. No paper checkpoint or dataset was downloaded by this integration.

Enter a sentence and its expected class (0 or 1), or choose a JSON batch:

```json
{
  "examples": [
    {"id": "review-1", "text": "a good movie", "label": 1}
  ]
}
```

`Json_examples/transformer_batch.json` illustrates the format; its labels are
examples, not a claim about a particular checkpoint's vocabulary or class mapping.
The runner loads these inputs directly. It does not load benchmark training
datasets, rebuild the vocabulary, train a model, or repeatedly sample until a
correctly classified example is found.

Use **Preview tokens** to inspect checkpoint tokenization, indices, eligibility,
prediction, and task count. Preview results are explicitly unverified. Long
sentences are returned as UNKNOWN without silently truncating the text.

## What the workflows certify

- **Check an input radius:** certify every task in the configured scope at the
  requested embedding radius.
- **Search certified radii:** search every task and return the minimum certified
  lower endpoint. Refinement can use a fixed number of iterations or a bracket
  tolerance. The search ceiling is a strict query limit.
- **Schedule positions with validation and fallback:** rank tasks, search the
  selected budget, validate every remaining task at the candidate radius, and
  search failed validations. The final radius is lowered when fallback requires
  it. Gradient ranking alone never establishes a certificate.

The supported bound methods are IBP, Forward, Backward, or the
IBP → Forward → Backward cascade. A method's numerical/assertion failure remains
UNKNOWN. The interface does not turn a failed certification into an adversarial
example. The upstream verifiers use ordinary floating-point arithmetic; these
are certificates conditional on the underlying verifier, not independently
checked proof objects.

Each verifier call receives an independent nominal embedding tensor because
the upstream IBP implementation modifies its input bounds in place. Experimental
BAF/alternative bound variants are not exposed by this initial integration.

The norm applies to perturbations in embeddings before layer normalization.
For one-token tasks, each eligible token is perturbed individually. For two-token
tasks, each eligible unordered pair is a task with a radius budget for each
token. These properties do not permit arbitrary simultaneous perturbations of
every token and do not directly certify discrete synonym replacements.

An empty position field means all eligible positions. An explicit list limits
the certified scope, and the exported result records that scope. Eligibility
follows the upstream complete-token rule and excludes special/subword positions.
Two-token task counts grow quadratically with sentence length.

## Results and cancellation

Results use a versioned common schema:

| Status | Meaning |
| --- | --- |
| CERTIFIED | Every task in the declared scope was certified at the reported radius. |
| COUNTEREXAMPLE | A violating example was established; for text, clean misclassification is detected. |
| UNKNOWN | The engine did not establish the requested certificate. |
| TIMEOUT | The configured run deadline was reached. |
| CANCELLED | The user stopped the run. |
| ERROR | Setup, compatibility, input, process, or transport failure. |

Only CERTIFIED results carry a certified radius. A search bracket's upper endpoint
is a certification-search boundary, not an established adversarial radius.
Batch results retain each example's status, token scope, bound statistics,
validation/fallback tasks, and certification evidence. A batch containing UNKNOWN
results is not presented as fully certified. GPU memory exhaustion remains UNKNOWN.

The GUI uses a separate `QProcess`, clears inherited ROS/Python/Qt library paths
for the runner, and receives JSONL events without blocking the main window.
Stop first requests termination, then forces exit if necessary. A watchdog handles
unresponsive deadline expiry. Incomplete output or a crashed process cannot count
as a completed certificate. Closing the window stops the child process.

Run artifacts are saved under `~/.local/share/StabilEdge/runs/`:

- `request.json`: inputs and verification settings;
- `events.jsonl`: structured progress and results;
- `engine.log`: backend output;
- `results.json`: final status and completed per-example results.

Use **Save config**, **Load config**, and **Export results** for reproducible runs.
Configurations embed the labelled inputs. Results include the source revision,
PyTorch/NumPy versions, and elapsed time. These run files contain the input text.

## Code and validation

`verification/registry.py` declares engines and capabilities.
`verification/marabou_worker.py` contains the extracted existing Marabou worker.
`verification/core.py` defines common statuses and conservative query aggregation.
`verification/transformer_runner.py` loads the upstream model/verifiers;
`verification/radius.py` owns search and scheduling. The upstream source is used
without modification, so installing a fresh copy does not lose local fixes.

The Marabou ITD workflows now preserve UNKNOWN and cancellation rather than
falling through to UNSAT. They apply the normalized delta and the selected/all-pixel
scope consistently, and reject input-size mismatches instead of padding or
truncating the sample. This can change results compared with the old behavior.
The GUI's explicit result table distinguishes statuses; legacy charts combine
inconclusive outcomes.

Run the tests from the StabilEdge root:

```bash
python3 -m unittest discover -s tests -p test_certification.py -v
python3 -m unittest discover -s tests -p test_transformer_integration.py -v

# In verify_gui, with the same Qt library isolation as run_gui.sh:
python -m unittest discover -s tests -p test_marabou.py -v
python -m unittest discover -s tests -p test_qt_engines.py -v
```

The optional integration tests create temporary synthetic checkpoints and ONNX
models. They exercise real verifiers and process transport, including successful
certification, failed certification, search, scheduling, cancellation, deadline
expiry, malformed inputs, and incomplete outputs. They do not reproduce the
paper's trained-model experiments. Set `STABILEDGE_TRANSFORMER_REPO` and
`STABILEDGE_TRANSFORMER_PYTHON` for the Transformer tests if using other locations.
