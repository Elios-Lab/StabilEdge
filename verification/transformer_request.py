"""Validate requests before importing the optional Transformer dependencies."""
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
import json
import math
import re


def checkpoint_directory(path):
    path = Path(path).expanduser().resolve()
    marker = path / "checkpoint"
    if marker.is_file():
        epoch = int(marker.read_text().strip())
        path = path / ("ckpt-%d" % epoch)
    if not path.is_dir():
        raise ValueError("Choose an existing Transformer checkpoint directory.")
    for name in ("pytorch_model.bin", "vocab.txt"):
        if not (path / name).is_file():
            raise ValueError("Checkpoint is missing %s." % name)
    configs = [path / "bert_config.json", path / "config.json"]
    config_path = next((item for item in configs if item.is_file()), None)
    if config_path is None:
        raise ValueError("Checkpoint needs bert_config.json or config.json.")
    config = json.loads(config_path.read_text())
    if config.get("num_labels", 2) != 2:
        raise ValueError("This engine currently supports binary classification only.")
    if config.get("hidden_act", "gelu") not in ("relu", "gelu", "tanh"):
        raise ValueError("Unsupported Transformer activation.")
    return path, config


def read_examples(path):
    payload = json.loads(Path(path).expanduser().read_text())
    if isinstance(payload, dict):
        payload = payload.get("request", payload)
        payload = payload.get("examples", payload.get("cases"))
    if not isinstance(payload, list) or not payload:
        raise ValueError('Batch JSON must contain a nonempty "examples" list.')
    examples = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError("Every batch example must be a JSON object.")
        text = row.get("text", row.get("sent_a", ""))
        if isinstance(text, list):
            if any(not isinstance(token, str) for token in text):
                raise ValueError("Token lists must contain strings.")
            text = " ".join(text)
        examples.append({"text": text, "label": row.get("label"), "id": str(row.get("id", index))})
    return examples


@dataclass
class TransformerRequest:
    repository: str
    checkpoint: str
    examples: list
    action: str = "check"
    methods: list = field(default_factory=lambda: ["ibp", "forward", "backward"])
    norm: str = "2"
    radius: float = 0.01
    maximum_radius: float = 2.0
    iterations: int = 4
    tolerance: object = None
    max_length: int = 32
    perturbed_tokens: int = 1
    positions: list = field(default_factory=list)
    budget: int = 1
    ranker: str = "gradient"
    device: str = "cpu"
    seed: int = 0

    @classmethod
    def from_dict(cls, payload, validate_paths=True):
        request = cls(**payload)
        request.validate(check_paths=validate_paths)
        return request

    def validate(self, check_paths=True):
        if check_paths:
            root = Path(self.repository).expanduser().resolve()
            for name in ("Parser.py", "Models/BERT.py", "Verifiers/VerifierForward.py", "experiments/cascade_verification.py"):
                if not (root / name).is_file():
                    raise ValueError("Choose the Transformer-Position-Scheduling-Certification repository folder.")
            checkpoint_directory(self.checkpoint)
        if self.action not in ("preview", "check", "search", "schedule"):
            raise ValueError("Unknown certification workflow.")
        if not self.methods or any(method not in ("ibp", "forward", "backward") for method in self.methods):
            raise ValueError("Choose a supported verifier method or cascade.")
        if self.norm not in ("1", "2", "inf"):
            raise ValueError("Norm must be L1, L2, or L-infinity.")
        if not re.fullmatch(r"cpu|cuda:\d+", self.device):
            raise ValueError("Device must be cpu or cuda:N.")
        if not (math.isfinite(self.radius) and math.isfinite(self.maximum_radius) and 0 < self.radius <= self.maximum_radius):
            raise ValueError("Require 0 < input radius <= search ceiling.")
        if self.tolerance is not None and (not math.isfinite(self.tolerance) or self.tolerance <= 0):
            raise ValueError("Search tolerance must be positive.")
        if not 1 <= self.iterations <= 100 or not 3 <= self.max_length <= 512:
            raise ValueError("Choose 1–100 search iterations and a token limit of 3–512.")
        if self.perturbed_tokens not in (1, 2) or self.budget < 1:
            raise ValueError("Choose one/two perturbed tokens and a positive scheduling budget.")
        if self.ranker not in ("gradient", "left_to_right", "random"):
            raise ValueError("Unsupported position ranker.")
        if any(type(pos) is not int or pos < 1 for pos in self.positions) or len(set(self.positions)) != len(self.positions):
            raise ValueError("Positions must be unique positive token indices.")
        if not isinstance(self.examples, list) or not self.examples:
            raise ValueError("Provide a labelled sentence or a nonempty batch.")
        ids = set()
        for index, example in enumerate(self.examples):
            if not isinstance(example, dict) or not isinstance(example.get("text"), str) or not example["text"].strip():
                raise ValueError("Every example needs nonempty text.")
            if type(example.get("label")) is not int or example["label"] not in (0, 1):
                raise ValueError("Every text example needs a label of 0 or 1.")
            sample_id = str(example.get("id", index))
            if sample_id in ids:
                raise ValueError("Example IDs must be unique.")
            ids.add(sample_id)

    def to_dict(self):
        return asdict(self)


def position_tasks(valid, selected, count):
    valid = list(valid)
    if any(pos not in valid for pos in selected):
        raise ValueError("Selected positions must be eligible complete tokens, excluding special tokens and subwords.")
    positions = selected or valid
    return [(pos,) for pos in positions] if count == 1 else list(combinations(positions, 2))
