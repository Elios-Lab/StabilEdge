from __future__ import annotations
import json
import numpy as np
import tempfile
import pandas as pd
import seaborn as sns
from maraboupy import Marabou
from matplotlib.figure import Figure
from typing import List, Tuple, Optional, Dict, Any, Callable


def run_jena(
    json_path: str,
    delta_orig: float,
    epsilon_orig: Optional[float],
    perturb_all: bool,
    selected_flat_idxs: List[int],
    plot: bool = False,
    marabou_opts: Dict[str, Any] = {},
    stop_requested: Callable[[], bool] = lambda: False,
    progress_callback: Optional[Callable[[int,int,int,int,int],None]] = None
) ->  List[Tuple[Optional[float], float, int, int, int]]:
    
    # 1) Load config
    with open(json_path, "r") as f:
        cfg = json.load(f)

    features = cfg.get("features", [])

    # 2) Build domain map and feature ranges
    domain_map: Dict[int, Tuple[float, float]] = {}
    feature_ranges: List[float] = []
    for feat in features:
        lo = feat["properties"]["input_domain"]["min"]
        hi = feat["properties"]["input_domain"]["max"]
        feature_ranges.append(hi - lo)
        for idx in feat["properties"]["starting_indexes"]:
            domain_map[idx] = (lo, hi)

    # 3) Build normalized deltas
    perturbed_index_to_delta: Dict[int, float] = {}
    if perturb_all:
        for idx, (lo, hi) in domain_map.items():
            perturbed_index_to_delta[idx] = delta_orig / (hi - lo)
    else:
        for idx in selected_flat_idxs:
            if idx in domain_map:
                lo, hi = domain_map[idx]
                perturbed_index_to_delta[idx] = delta_orig / (hi - lo)
            else:
                print(f"→ Warning: selected index {idx} not in domain_map")

    # 4) Normalize epsilon
    if epsilon_orig is not None:
        if cfg.get("output_domain"):
            od = cfg["output_domain"]
            eps_den = od["max"] - od["min"]
        else:
            eps_den = max(feature_ranges) if feature_ranges else 1.0
        epsilon_norm = epsilon_orig / eps_den
    else:
        epsilon_norm = None


    # 5) Load inputs
    inp_path = (
        cfg.get("file_paths", {}).get("input_file")
        or cfg.get("file_paths", {}).get("input_path")
        or cfg.get("input_path", "")
    )
    raw = np.load(inp_path)

    # figure out how many features we expect per sample
    flat_count = sum(
        len(feat["properties"]["starting_indexes"])
        for feat in features
    )

    # if raw.size == flat_count, it’s a single sample → wrap it
    if raw.size == flat_count:
        inputs = raw.reshape(1, flat_count)
    else:
        # otherwise assume raw is (n_samples, flat_count)
        inputs = raw

    # inputs = np.load(inp_path)

    sat_count = unsat_count = timeout_count = 0
    results: List[Tuple[Optional[float], float, int, int]] = []

    # 6) Verify each sample
    total = len(inputs)
    for sidx, sample in enumerate(inputs, start=1):
        if stop_requested():
            print(f"→ Stop requested, aborting Jena at sample {sidx}")
            break
        print(f"\nVerifying sample {sidx}…")
        try:
            net = Marabou.read_onnx(cfg["file_paths"]["onnx_model"])
            inVars, outVars = net.inputVars[0], net.outputVars[0]

            flat = sample.flatten()
            nom_val = float(net.evaluateWithoutMarabou([flat])[0].item())

            # set input bounds
            for i, var in enumerate(inVars.flatten()):
                base = flat[i]
                if perturb_all or i in perturbed_index_to_delta:
                    d = perturbed_index_to_delta.get(i, 0.0)
                    net.setLowerBound(var, base - d)
                    net.setUpperBound(var, base + d)
                else:
                    net.setLowerBound(var, base)
                    net.setUpperBound(var, base)

            # set output bound if epsilon given
            if epsilon_norm is not None:
                net.setUpperBound(outVars[0, 0], nom_val - epsilon_norm)

            opts = Marabou.createOptions(**marabou_opts)
            exitCode, vals, stats = net.solve(options=opts)

            # try other side if UNSAT
            if epsilon_norm is not None and exitCode == 'unsat':
                net.upperBounds.pop(outVars[0, 0], None)
                net.setLowerBound(outVars[0, 0], nom_val + epsilon_norm)
                exitCode, vals, stats = net.solve(options=opts)

            # record result
            if exitCode == 'sat':
                sat_flag, unsat_flag, timeout_flag = 1, 0, 0
                sat_count += 1
            elif exitCode == 'unsat':
                sat_flag, unsat_flag, timeout_flag = 0, 1, 0
                unsat_count += 1
            elif exitCode == 'TIMEOUT':
                sat_flag, unsat_flag, timeout_flag = 0, 0, 1
                timeout_count += 1
            else:
                sat_flag, unsat_flag, timeout_flag = 0, 0, 0
                print(f" → Unknown solver code: {exitCode}")

            # for display we keep the original epsilon, but for solving we use epsilon_norm
            display_eps = epsilon_orig if epsilon_orig is not None else 0.0
            results.append((display_eps, epsilon_norm, delta_orig, sat_flag, unsat_flag, timeout_flag))

            # emit progress after this sample
            if progress_callback is not None:
                progress_callback(sidx, total, sat_count, unsat_count, timeout_count)

        except Exception as e:
            print(f" Error on sample {sidx}: {e}")

    # 7) Emit heatmap if requested
    if plot and results:
        # 1) Aggregate results into a DataFrame
        df = pd.DataFrame(results, columns=[
            "Epsilon_raw","Epsilon_norm","Delta","SAT","UNSAT","TIMEOUT"
        ])
        grouped = df.groupby(["Epsilon","Delta"]).sum().reset_index()

        # 2) Pivot into two grids (SAT counts and TIMEOUT counts)
        sat_grid = (
            grouped
              .pivot(index="Epsilon", columns="Delta", values="SAT")
              .sort_index(ascending=False)[sorted(grouped["Delta"].unique())]
        )
        to_grid = (
            grouped
              .pivot(index="Epsilon", columns="Delta", values="TIMEOUT")
              .sort_index(ascending=False)[sorted(grouped["Delta"].unique())]
        )

        # 3) Mask-out so they never overlap
        sat_only = sat_grid.copy()
        sat_only[to_grid > 0] = np.nan

        to_only = to_grid.copy()
        to_only[to_only == 0] = np.nan

        # 4) Plot both layers on one set of axes
        fig = Figure(figsize=(10,6))
        ax  = fig.add_subplot(111)

        # — SAT layer (coolwarm), only where no TIMEOUT —
        if sat_only.notna().any().any():
            sns.heatmap(
                sat_only,
                annot=sat_only.fillna(0).astype(int),
                fmt="d",
                cmap="coolwarm",
                vmin=0,
                vmax=grouped["SAT"].max(),
                cbar_kws={"label":"# SAT"},
                mask=sat_only.isna(),
                ax=ax
            )

        # — TIMEOUT layer (gray), only where TIMEOUT occurred —
        if to_only.notna().any().any():
            max_to = int(np.nanmax(to_only.values))
            sns.heatmap(
                to_only,
                annot=to_only.fillna(0).astype(int),
                fmt="d",
                cmap="Greys",
                vmin=0,          # ensure gray scale runs from light→dark
                vmax=max_to,     # up to the largest timeout count
                cbar=False,
                mask=to_only.isna(),
                linewidths=0.5,
                linecolor="white",
                ax=ax
            )

        ax.set(xlabel="Δ", ylabel="ε", title="Jena Results (gray = TIMEOUT)")
        fig.tight_layout()

        # 5) Dump to PNG and emit
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, bbox_inches="tight")
        print(f"__PLOT__:{tmp.name}")

    return results