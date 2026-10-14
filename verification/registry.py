"""Engine discovery and lazy creation of existing Marabou workers."""
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineCapabilities:
    id: str
    name: str
    model_format: str
    input_kind: str
    properties: tuple


ENGINES = (
    EngineCapabilities("marabou", "Marabou", "onnx", "numeric", ("classification", "regression")),
    EngineCapabilities("transformer", "Transformer Bounds", "checkpoint", "text", ("binary_classification",)),
)


def create_marabou_worker(**kwargs):
    from verification.marabou_worker import MarabouWorker
    return MarabouWorker(**kwargs)
