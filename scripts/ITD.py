import os
import json
import tempfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
import cv2  # for Otsu thresholding on gradients
from maraboupy import Marabou
import collections
import copy
from typing import Optional, List, Tuple, Dict, Any, Callable

import glob

# --------------------------------------------
# Utility functions
# --------------------------------------------


def normalize_difference(diff: float, tmin: float, tmax: float) -> float:
    return diff / (tmax - tmin)




def load_random_image(directory: str) -> Tuple[np.ndarray, str]:
    pngs = [f for f in os.listdir(directory) if f.lower().endswith(".png")]
    if not pngs:
        raise FileNotFoundError(f"No PNG images found in {directory}")
    fn = random.choice(pngs)
    print(f"Selected image: {fn}")
    img = Image.open(os.path.join(directory, fn))
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    return arr / 255.0, fn


def compute_meaningful_mask(image: np.ndarray) -> np.ndarray:
    gray = image.mean(axis=2) if image.ndim == 3 else image
    gx, gy = np.gradient(gray)
    mag = np.sqrt(gx**2 + gy**2)
    norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    scaled = (255 * norm).astype(np.uint8)
    _, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask > 0


def set_pixel_bounds_adaptive(
    network,
    input_vars: np.ndarray,
    image: np.ndarray,
    delta: float,
    perturb_all: bool = False,
    selected_idxs: Optional[List[int]] = None
) -> None:
    """
    If perturb_all is False, only perturbs pixels in selected_idxs or "meaningful" mask.
    Otherwise perturbs all pixels by ±delta.
    """
    # 1) build a per‐pixel mask matching image.shape
    if selected_idxs is not None:
        # selected_idxs are flat‐indices into image.flatten()
        flat_count = int(np.prod(image.shape))
        flat_mask = np.zeros(flat_count, dtype=bool)
        flat_mask[selected_idxs] = True
        mask = flat_mask.reshape(image.shape)
    elif not perturb_all:
        # compute 2D meaningful mask
        mask = compute_meaningful_mask(image)  # shape (H,W) for color or gray
        # if image has channels, replicate mask for each channel
        if image.ndim == 3 and mask.ndim == 2:
            mask = np.repeat(mask[:, :, None], image.shape[2], axis=2)
    else:
        mask = None

    # 2) flatten vars, image, and mask (if any)
    flat_vars = input_vars.flatten()
    flat_img  = image.flatten()
    flat_mask = None if mask is None else mask.flatten()

    # 3) set bounds per‐variable
    for i, var in enumerate(flat_vars):
        pix = flat_img[i]
        if perturb_all or (flat_mask is not None and flat_mask[i]):
            lo = max(pix - delta, 0.0)
            hi = min(pix + delta, 1.0)
        else:
            lo = hi = pix
        network.setLowerBound(var, lo)
        network.setUpperBound(var, hi)



def get_image_from_marabou(vals: Dict[int, float], input_vars: np.ndarray) -> np.ndarray:
    shp = input_vars.shape
    out = np.zeros(shp)
    for idx in np.ndindex(shp):
        out[idx] = vals[input_vars[idx]]
    return out

# --------------------------------------------
# Tracking stats
# --------------------------------------------

misclassification_stats = collections.Counter()
correct_counter = 0
detailed_report: List[Dict] = []

# --------------------------------------------
# Local Robustness / Sensitivity
# --------------------------------------------
def local_rob_sens(
    image: np.ndarray,
    filename: str,
    delta: float,
    onnx_path: str,
    tmin: float,
    tmax: float,
    selected_classes: Optional[List[int]] = None,
    verbosity: int = 0,
    marabou_opts: Dict[str, Any] = {},
    progress_callback: Optional[Callable] = None,
    stop_requested: Callable[[], bool] = lambda: False
) -> Tuple[Dict, str]:
    # 1. Compute normalized delta (and print raw vs. normalized)
    normalized_delta = normalize_difference(delta, tmin=tmin, tmax=tmax)
    # This print will now actually show you both values:
    print(f"Delta before normalization: {delta}, Delta normalized: {normalized_delta:.6f}")


    net0 = Marabou.read_onnx(onnx_path)
    inp0 = net0.inputVars[0][0]
    out0 = net0.outputVars[0][0]

    orig_out = net0.evaluateWithoutMarabou([image])[0]
    orig_pred = int(np.argmax(orig_out))
    print(f"Original predicted class: {orig_pred}")

    num_classes = out0.shape[0]
    targets = selected_classes or [c for c in range(num_classes) if c != orig_pred]
    adversarial_outcomes: List[Dict[str, Any]] = []
    statuses: List[str] = []

    for alt in targets:
        if stop_requested():
            print(f"→ Stop requested inside ITD at class {alt}")
            break

        net = Marabou.read_onnx(onnx_path)
        inp = net.inputVars[0][0]
        out = net.outputVars[0][0]

        # only perturb pixels in meaningful regions
        set_pixel_bounds_adaptive(net, inp, image, delta, perturb_all=False)

        # enforce that the logit for class `alt` ≥ logit(j) + ε for all j≠alt
        for j in range(num_classes):
            if j == alt:
                continue
            net.addInequality([out[j], out[alt]], [1.0, -1.0], -1e-6)

        opts = Marabou.createOptions(**marabou_opts)
        exit_code, vals, stats = net.solve(options=opts)
        lc = exit_code.lower()
        statuses.append(lc)

        if vals:
            adv = get_image_from_marabou(vals, inp)
            adv_out = net.evaluateWithoutMarabou([adv])[0]
            adv_pred = int(np.argmax(adv_out))
            if adv_pred == alt:
                misclassification_stats[(orig_pred, adv_pred)] += 1
                adversarial_outcomes.append({
                    "forced_class": alt,
                    "new_prediction": adv_pred,
                    "original_logits": orig_out,
                    "adv_logits": adv_out
                })

    # collapse all per-alt statuses into one final verdict:
    #  - if any SAT → overall SAT
    #  - else if any TIMEOUT → overall TIMEOUT
    #  - else → all were UNSAT → overall UNSAT
    if "sat" in statuses:
        overall = "sat"
    elif "timeout" in statuses:
        overall = "timeout"
    else:
        overall = "unsat"

    report = {
        "filename": filename,
        "original_prediction": orig_pred,
        "adversarial_results": adversarial_outcomes
    }
    detailed_report.append(report)

    return report, overall


# --------------------------------------------
# Targeted Robustness / Sensitivity
# --------------------------------------------
def targeted_rob_sens(
    image: np.ndarray,
    filename: str,
    delta: float,
    onnx_path: str,
    tmin: float,
    tmax: float,
    perturb_all_pixels: bool,
    selected_idxs: List[int],
    selected_classes: Optional[List[int]] = None,
    verbosity: int = 0,
    marabou_opts: Dict[str,Any] = {},
    progress_callback: Optional[Callable] = None,
    stop_requested: Callable[[], bool] = lambda: False
) -> Tuple[Dict, str]:
    
        # 1. Normalize (and print)
    normalized_delta = normalize_difference(delta, tmin=tmin, tmax=tmax)
    print(f"Delta before normalization: {delta}, Delta normalized: {normalized_delta:.4f}")

    base_net = Marabou.read_onnx(onnx_path)
    base_inp = base_net.inputVars[0][0]
    base_out = base_net.outputVars[0][0]

    orig_out = base_net.evaluateWithoutMarabou([image])[0]
    orig_pred = int(np.argmax(orig_out))
    print(f"Original pred: {orig_pred}")

    num_classes = base_out.shape[0]
    targets = selected_classes or [c for c in range(num_classes) if c != orig_pred]

    adversarial_outcomes: List[Dict[str, Any]] = []
    statuses: List[str] = []

    for alt in targets:
        net = copy.deepcopy(base_net)
        inp = net.inputVars[0][0]
        out = net.outputVars[0][0]

        # set adaptive bounds (either full‐image or selected pixels)
        set_pixel_bounds_adaptive(
            net, inp, image, delta,
            perturb_all=perturb_all_pixels,
            selected_idxs=selected_idxs
        )

        # require logit(alt) ≥ logit(j) + ε for all j ≠ alt
        for j in range(num_classes):
            if j == alt:
                continue
            net.addInequality([out[j], out[alt]], [1.0, -1.0], -1e-6)

        opts = Marabou.createOptions(**marabou_opts)
        exit_code, vals, stats = net.solve(options=opts)
        lc = exit_code.lower()
        statuses.append(lc)

        # if a counter‐example is found, record and stop iterating further alts
        if vals:
            adv = get_image_from_marabou(vals, inp)
            adv_out = net.evaluateWithoutMarabou([adv])[0]
            adv_pred = int(np.argmax(adv_out))
            if adv_pred == alt:
                adversarial_outcomes.append({
                    "forced_class": alt,
                    "new_prediction": adv_pred,
                    "original_logits": orig_out,
                    "adv_logits": adv_out
                })
                break

    # collapse per‐alt statuses into one overall result:
    #   - SAT if any alt check was SAT
    #   - else TIMEOUT if any alt timed out
    #   - else UNSAT (all attempted alts were UNSAT)
    if "sat" in statuses:
        overall = "sat"
    elif "timeout" in statuses:
        overall = "timeout"
    else:
        overall = "unsat"

    report = {
        "filename": filename,
        "original_prediction": orig_pred,
        "adversarial_results": adversarial_outcomes
    }

    return report, overall


import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def plot_aggregated_itd(
    itd_results: List[Tuple[float, int, int, int]],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Given [(delta, sat_count, unsat_count), …], returns a SAT/UNSAT stacked
    bar chart as a Figure, using navy/skyblue colors.
    """
    # Unpack
    deltas   = [d for d,_,_,_ in itd_results]
    sat      = np.array([s for _,s,_,_ in itd_results], dtype=float)
    unsat    = np.array([u for _,_,u,_ in itd_results], dtype=float)
    timeout       = np.array([t for _,_,_,t in itd_results], dtype=float)

    total     = sat + unsat + timeout
    # percentages
    sat_pct   = np.where(total>0, 100*sat/total, 0)
    unsat_pct = np.where(total>0, 100*unsat/total, 0)
    timeout_pct = np.where(total>0, 100*timeout/total, 0)

    # Build figure
    fig, ax = plt.subplots(figsize=figsize)
    x     = np.arange(len(deltas))
    width = 0.8

    # draw bars
    ax.bar(x, sat_pct,   width, color='navy',    label='SAT')
    ax.bar(x, unsat_pct, width, bottom=sat_pct,  color='skyblue', label='UNSAT')
    ax.bar(x, timeout_pct, width, bottom=sat_pct+unsat_pct, color='darkgrey', label='TIMEOUT')

    # labels & ticks
    ax.set_xticks(x)
    ax.set_xticklabels(deltas)
    ax.set_xlabel('Delta')
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 100)
    ax.legend()

    fig.tight_layout()
    return fig




# ── in ITD.py (or a new plotting_utils.py) ────────────────────────────

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from typing import Dict, List, Tuple

def plot_grouped_confusion(
    confusion_data: Dict[float, List[List[float]]],
    gt_labels: List[str],
    figsize: Tuple[int, int] = (18, 10)
) -> plt.Figure:
    """
    Given a dict mapping δ → (num_classes×num_classes %) matrices,
    produce a grouped‐bar chart for each δ, showing:
      • UNSAT (i==j) in blue
      • SAT   (i!=j) in red
      • TIMEOUT (the leftover to 100%) in dark grey
    """
    deltas     = sorted(confusion_data.keys())
    num_panels = len(deltas)
    num_classes = len(gt_labels)

    # one extra “predicted” category for TIMEOUT
    pred_labels = [f"C{j}" for j in range(num_classes)] + ["TO"]

    # layout: smallest square ≥ num_panels
    rows = cols = int(np.ceil(np.sqrt(num_panels)))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_list = np.array(axes).flatten()

    bar_w     = 0.2
    group_gap = 1

    for ax, delta in zip(axes_list, deltas):
        matrix = np.array(confusion_data[delta])  # shape (num_classes, num_classes)
        # compute timeout % per GT class
        row_sums = matrix.sum(axis=1)
        timeout_pct = np.maximum(0, 100 - row_sums)

        # extend each row with timeout % as last “pred” column
        ext_matrix = np.hstack([matrix, timeout_pct[:, None]])

        x_pos, x_lbls = [], []
        for i in range(num_classes):
            base = i * (num_classes + 1 + group_gap) * bar_w
            for j in range(num_classes + 1):
                xpos  = base + j * bar_w
                value = ext_matrix[i, j]
                # choose color
                if j == num_classes:
                    color = "darkgrey"       # TIMEOUT
                elif i == j:
                    color = "blue"           # UNSAT
                else:
                    color = "red"            # SAT
                ax.bar(xpos, value, bar_w, color=color, edgecolor="black")
                x_pos.append(xpos)
                x_lbls.append(pred_labels[j])

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_lbls, fontsize=10, rotation=90)
        # draw GT labels under each block group
        centers = [
            (i*(num_classes+1+group_gap) + (num_classes)/2) * bar_w
            for i in range(num_classes)
        ]
        for i, lbl in enumerate(gt_labels):
            ax.text(centers[i], -10, lbl, ha="center", fontsize=12)

        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_title(f"δ = {delta}", pad=20)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

    # hide any extra subplots
    for ax in axes_list[len(deltas):]:
        ax.set_visible(False)

    # global legend
    legend_elems = [
        Patch(color="blue",      label="UNSAT"),
        Patch(color="red",       label="SAT"),
        Patch(color="darkgrey",  label="TIMEOUT"),
    ]
    fig.legend(
        handles=legend_elems,
        loc="upper right",
        ncol=3,
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# ── Dataset‐wide runner for ITD (replaces your old run_itd) ─────────
def run_itd(
    json_path: str,
    delta_original: float,
    perturb_all_pixels: bool,
    stop_requested: Callable[[], bool] = lambda: False,
    progress_callback: Optional[Callable[[int,int,int,int,int],None]] = None,
    json_cfg: Optional[str] = None
) -> Tuple[int, int, int]:
    # load config
    with open(json_path, 'r') as f:
        cfg = json.load(f)
    onnx_p = cfg['file_paths']['onnx_model']
    inp_p  = cfg['file_paths'].get('input_file') \
            or cfg['file_paths'].get('image_directory')

    # build list of images / arrays
    if os.path.isdir(inp_p):
        exts = ('*.png','*.jpg','*.jpeg','*.npy','*.csv')
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(inp_p, ext)))
    else:
        paths = [inp_p]

    sat_cnt = unsat_cnt = timeout_cnt = 0

    total = len(paths)
    for idx, img_path in enumerate(paths, start=1):
        if stop_requested():
            print("Stopping ITD run as requested.")
            break
        # load image/array in [0,1]
        ext = os.path.splitext(img_path)[1].lower()
        if ext in ('.png', '.jpg', '.jpeg'):
            img = Image.open(img_path).convert('RGB')
            image = np.array(img, dtype=np.float32) / 255.0
        elif ext == '.csv':
            df   = pd.read_csv(img_path, header=None)
            flat = df.to_numpy(dtype=float).flatten()
            side = int(np.sqrt(flat.size))
            image = flat.reshape(side, side).astype(np.float32)
            image /= image.max() if image.max() > 1 else 1
        elif ext == '.npy':
            image = np.load(img_path).astype(np.float32)
            if image.max() > 1.0:
                image /= 255.0
        else:
            continue  # unsupported

        # run local robustness
        _, exitCode = local_rob_sens(
            image=image,
            filename=os.path.basename(img_path),
            delta=delta_original,
            onnx_path=onnx_p,
            selected_classes=None,
            verbosity=cfg.get('marabou', {}).get('verbosity', 0),
            marabou_opts=cfg.get('marabou', {}),
            stop_requested=stop_requested
        )

        code = exitCode
        if code == "sat":
            sat_cnt += 1
        elif code == "unsat":
            unsat_cnt += 1
        elif code == "timeout":
            timeout_cnt += 1
        else:
            # if Marabou ever returns something unexpected (e.g. "unknown")
            print(f"Warning: unrecognized exit code ‘{exitCode}’, counting as timeout")
            timeout_cnt += 1

        # emit progress after this image
        if progress_callback is not None:
            progress_callback(idx, total, sat_cnt, unsat_cnt, timeout_cnt)

    total = sat_cnt + unsat_cnt + timeout_cnt
    print(f"Results: {sat_cnt} SAT, {unsat_cnt} UNSAT, {timeout_cnt} TIMEOUT over {total} samples.")

    # now plot
    fig = plot_aggregated_itd([(delta_original, sat_cnt, unsat_cnt, timeout_cnt)])
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    out_png = tmp.name
    tmp.close()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"__PLOT__:{out_png}")

    return sat_cnt, unsat_cnt, timeout_cnt

# alias
run_ITD = run_itd

