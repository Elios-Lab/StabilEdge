"""Backend-neutral outcomes. No GUI, PyTorch, or solver imports."""
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, Optional


class Status(str, Enum):
    CERTIFIED = "CERTIFIED"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"
    UNKNOWN = "UNKNOWN"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


def solver_status(code):
    return {
        "sat": Status.COUNTEREXAMPLE,
        "unsat": Status.CERTIFIED,
        "timeout": Status.TIMEOUT,
        "cancelled": Status.CANCELLED,
        "error": Status.ERROR,
    }.get(str(code).lower(), Status.UNKNOWN)


def aggregate_queries(codes, expected, cancelled=False):
    """A universal property needs every query to complete successfully."""
    codes = [str(code).lower() for code in codes]
    if "sat" in codes:
        return "sat"
    if cancelled:
        return "cancelled"
    if "timeout" in codes:
        return "timeout"
    if expected > 0 and len(codes) == expected and all(code == "unsat" for code in codes):
        return "unsat"
    return "unknown"


@dataclass
class VerificationResult:
    engine: str
    sample_id: str
    status: Status
    radius: Optional[float] = None
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        if self.radius is not None and (not math.isfinite(self.radius) or self.radius < 0):
            raise ValueError("A radius must be finite and nonnegative.")
        if self.status != Status.CERTIFIED and self.radius is not None:
            raise ValueError("Only certified results may carry a certified radius.")
        result = asdict(self)
        result["status"] = self.status.value
        result["schema_version"] = 1
        return result
