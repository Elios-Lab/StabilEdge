"""Search and scheduling over certification queries, independent of a solver.

A false query means unknown, never a counterexample. Only successfully checked
lower endpoints can be returned. The search ceiling is a strict query limit.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RadiusBound:
    radius: object
    certified: bool
    upper: float
    iterations: int
    at_cap: bool = False


def search_radius(certify, initial=0.01, maximum=2.0, iterations=4, tolerance=None):
    if not (math.isfinite(initial) and math.isfinite(maximum) and 0 < initial <= maximum):
        raise ValueError("Require 0 < initial radius <= maximum radius.")
    if iterations < 0 or (tolerance is not None and (not math.isfinite(tolerance) or tolerance <= 0)):
        raise ValueError("Search iterations and tolerance must be valid.")
    # Explicitly verify the base point; numerical or assertion failures remain unknown.
    if not certify(0.0):
        return RadiusBound(None, False, initial, 0)
    low, high = 0.0, initial
    if certify(high):
        low = high
        while low < maximum:
            high = min(maximum, low * 2)
            if not certify(high):
                break
            low = high
        if low == maximum:
            return RadiusBound(low, True, maximum, 0, True)
    count = 0
    while (count < iterations if tolerance is None else high - low > tolerance):
        middle = (low + high) / 2
        if middle == low or middle == high:
            break
        if certify(middle):
            low = middle
        else:
            high = middle
        count += 1
    return RadiusBound(low, True, high, count)


def schedule_tasks(tasks, ranked, budget, search, validate, notify=lambda *args: None):
    """Search selected tasks; certify all others or search the failed tasks."""
    tasks, ranked = list(tasks), list(ranked)
    if not tasks or budget < 1 or len(set(tasks)) != len(tasks) or set(tasks) != set(ranked) or len(tasks) != len(ranked):
        raise ValueError("Scheduling needs a nonempty task set, a permutation, and a positive budget.")
    selected = ranked[:budget]
    skipped = ranked[budget:]
    bounds = {}
    for task in selected:
        bounds[task] = search(task)
        notify("search", task, bounds[task])
    if any(not bound.certified for bound in bounds.values()):
        return {"radius": None, "certified": False, "selected": selected, "validated": [], "fallback": [], "bounds": bounds}
    candidate = min(bound.radius for bound in bounds.values())
    validated, fallback = [], []
    for task in skipped:
        if validate(task, candidate):
            validated.append(task)
            notify("validation", task, candidate)
        else:
            fallback.append(task)
            notify("validation_failed", task, candidate)
    for task in fallback:
        bounds[task] = search(task)
        notify("fallback", task, bounds[task])
    certified = all(bound.certified for bound in bounds.values())
    return {
        "radius": min([candidate] + [bound.radius for bound in bounds.values()]) if certified else None,
        "certified": certified,
        "selected": selected, "validated": validated, "fallback": fallback, "bounds": bounds,
    }
