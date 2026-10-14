from __future__ import annotations
import json
import numpy as np
import tempfile
import pandas as pd
import seaborn as sns
from maraboupy import Marabou
from matplotlib.figure import Figure
from typing import List, Tuple, Optional, Dict, Any, Callable


def _get_input_indices(feature_props: Dict[str, Any]) -> List[int]:
    """Get input indices from feature properties, supporting both old and new naming."""
    # Try new naming first, fall back to old naming for backward compatibility
    return feature_props.get("input_indices") or feature_props.get("starting_indexes", [])


def run_jena(
    json_path: str,
    delta_orig: float,
    epsilon_orig: Optional[float],
    perturb_all: bool,
    selected_flat_idxs: List[int],
    plot: bool = False,
    marabou_opts: Dict[str, Any] = {},
    stop_requested: Callable[[], bool] = lambda: False,
    progress_callback: Optional[Callable[[int,int,int,int,int],None]] = None,
    result_callback: Optional[Callable] = None
) ->  List[Tuple[Optional[float], float, int, int, int]]:
    """
    Verify time-series forecasting model robustness using Marabou.
    
    This function handles multi-feature time-series inputs where each feature
    may have multiple timesteps. The 'input_indices' in the JSON config specify
    which positions in the FLATTENED input vector correspond to each feature,
    NOT temporal ordering. The code respects temporal structure through the
    feature grouping defined in the config.
    
    Args:
        json_path: Path to feature configuration JSON
        delta_orig: Perturbation magnitude in original feature space
        epsilon_orig: Output deviation threshold in original output space
        perturb_all: If True, perturb all features; if False, only user-selected ones
        selected_flat_idxs: User-selected positions in flattened input (when perturb_all=False)
        plot: Whether to generate result heatmap
        marabou_opts: Solver configuration options
        stop_requested: Callback to check for cancellation
        progress_callback: Callback for progress updates
    
    Returns:
        List of (epsilon, epsilon_norm, delta, sat, unsat, timeout) tuples
    """
    
    # 1) Load config
    with open(json_path, "r") as f:
        cfg = json.load(f)

    features = cfg.get("features", [])

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 2: Build Feature-to-Position Mapping
    # ═══════════════════════════════════════════════════════════════════════
    # Purpose: Map each position in the flattened input vector to its feature's
    #          valid domain (min, max). This preserves the semantic meaning of
    #          each input dimension.
    #
    # Example: If temperature readings occupy positions [0,1,2] and humidity
    #          occupies [3,4,5] in a 6-element input vector, this maps:
    #          {0: (temp_min, temp_max), 1: (temp_min, temp_max), ...,
    #           3: (humid_min, humid_max), ...}
    #
    # The 'input_indices' field in JSON specifies which vector positions belong
    # to each feature, NOT the temporal order of measurements.
    # ═══════════════════════════════════════════════════════════════════════
    domain_map: Dict[int, Tuple[float, float]] = {}  # position → (min, max)
    feature_ranges: List[float] = []  # range (max-min) for each feature
    
    for feat in features:
        lo = feat["properties"]["input_domain"]["min"]
        hi = feat["properties"]["input_domain"]["max"]
        feature_ranges.append(hi - lo)
        
        # Map each input position that belongs to this feature
        for idx in _get_input_indices(feat["properties"]):
            domain_map[idx] = (lo, hi)

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 3: Compute Feature-Specific Normalized Perturbations
    # ═══════════════════════════════════════════════════════════════════════
    # Purpose: Convert the user-specified perturbation (delta_orig) from
    #          original feature units to normalized [0,1] space.
    #
    #
    # Two modes:
    #   - perturb_all=True: Apply perturbation to ALL input positions
    #   - perturb_all=False: Apply only to user-selected positions (selected_flat_idxs)
    #
    # Note: selected_flat_idxs contains POSITIONS in the flattened array that
    #       the user chose to perturb, NOT timestep indices.
    # ═══════════════════════════════════════════════════════════════════════
    perturbed_index_to_delta: Dict[int, float] = {}  # position → normalized delta
    
    if perturb_all:
        # Apply perturbation to every input position
        for idx, (lo, hi) in domain_map.items():
            perturbed_index_to_delta[idx] = delta_orig / (hi - lo)
    else:
        # Apply perturbation only to user-selected positions
        for idx in selected_flat_idxs:
            if idx in domain_map:
                lo, hi = domain_map[idx]
                perturbed_index_to_delta[idx] = delta_orig / (hi - lo)
            else:
                print(f"→ Warning: User-selected position {idx} not found in feature mapping")
                print(f"   Valid positions are: {sorted(domain_map.keys())}")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 4: Normalize Output Threshold (Epsilon)
    # ═══════════════════════════════════════════════════════════════════════
    # Purpose: Convert output deviation threshold from original units to
    #          normalized space, matching the network's output range.
    #
    # Example: If forecasting temperature in range [10°C, 35°C] and user wants
    #          ε=2°C tolerance, normalized ε = 2/(35-10) ≈ 0.08
    # ═══════════════════════════════════════════════════════════════════════
    if epsilon_orig is not None:
        if cfg.get("output_domain"):
            od = cfg["output_domain"]
            eps_den = od["max"] - od["min"]
        else:
            # Fallback: use largest input feature range as proxy
            eps_den = max(feature_ranges) if feature_ranges else 1.0
        epsilon_norm = epsilon_orig / eps_den
    else:
        epsilon_norm = None


    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 5: Load and Prepare Input Samples
    # ═══════════════════════════════════════════════════════════════════════
    # Purpose: Load input data and ensure it's in (n_samples, features) format.
    #
    # The input file contains FLATTENED feature vectors. For time-series with
    # multiple features and timesteps, all are concatenated into a 1D array.
    #
    # Example: 3 features × 10 timesteps = 30-element flattened vector
    #          [temp_t0, temp_t1, ..., temp_t9, humid_t0, ..., humid_t9, press_t0, ...]
    # ═══════════════════════════════════════════════════════════════════════
    inp_path = (
        cfg.get("file_paths", {}).get("input_file")
        or cfg.get("file_paths", {}).get("input_path")
        or cfg.get("input_path", "")
    )
    raw = np.load(inp_path)

    # Calculate expected input dimensionality from feature config
    # This counts total positions across all features (including all timesteps)
    flat_count = sum(
        len(_get_input_indices(feat["properties"]))
        for feat in features
    )

    # Handle both single-sample and multi-sample inputs
    if raw.size == flat_count:
        # Single sample: reshape to (1, flat_count)
        inputs = raw.reshape(1, flat_count)
    else:
        # Multiple samples: assume already in (n_samples, flat_count) format
        inputs = raw

    # inputs = np.load(inp_path)

    sat_count = unsat_count = timeout_count = 0
    results: List[Tuple[Optional[float], float, int, int]] = []
    all_stats = []  # Collect all stats objects

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 6: Robustness Verification Loop
    # ═══════════════════════════════════════════════════════════════════════
    # Purpose: For each input sample, verify whether perturbed inputs within
    #          δ-ball can cause output deviation > ε.
    #
    # Process:
    #   1. Get nominal (unperturbed) output value
    #   2. Set input bounds: nominal ± δ for selected positions, exact for others
    #   3. Constrain output to be outside [nominal-ε, nominal+ε]
    #   4. Ask Marabou: SAT = adversarial example exists, UNSAT = robust
    # ═══════════════════════════════════════════════════════════════════════
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

            # Set bounds for each input position
            # - If position is selected for perturbation: [base-δ, base+δ]
            # - Otherwise: fix to exact nominal value [base, base]
            for i, var in enumerate(inVars.flatten()):
                base = flat[i]
                if perturb_all or i in perturbed_index_to_delta:
                    d = perturbed_index_to_delta.get(i, 0.0)
                    net.setLowerBound(var, base - d)
                    net.setUpperBound(var, base + d)
                else:
                    # Fix this position to nominal value (no perturbation)
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
            
            all_stats.append(stats)  # Store stats object
            exitCode = str(exitCode).lower()
            if result_callback is not None:
                result_callback(str(sidx), exitCode, "")

            # record result
            if exitCode == 'sat':
                sat_flag, unsat_flag, timeout_flag = 1, 0, 0
                sat_count += 1
            elif exitCode == 'unsat':
                sat_flag, unsat_flag, timeout_flag = 0, 1, 0
                unsat_count += 1
            elif exitCode == 'timeout':
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
            if result_callback is not None:
                result_callback(str(sidx), "error", str(e))

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

    return results, all_stats
