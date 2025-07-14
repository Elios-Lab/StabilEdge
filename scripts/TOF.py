# TOF.py
from __future__ import annotations
import os
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from maraboupy import Marabou
from matplotlib import pyplot as plt
import tempfile


def load_config(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor from 0–255 to 0–1."""
    return tensor / 255.0


def normalize_delta(raw: float, factor: float, delta_type: str) -> float:
    if delta_type == "relative":
        return raw
    return raw / factor


# ── make CSVImageFolder accept a `transform` ───────────────────────────────────
class CSVImageFolder(Dataset):
    def __init__(
        self,
        root: str,
        loader: Optional[Callable[[str], torch.Tensor]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        self.root = root
        self.loader = loader or self.default_loader
        self.transform = transform

        self.classes = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        if len(self.classes) == 1 and self.classes[0].lower() == "robot":
            self.class_to_idx = {self.classes[0]: 1}
        else:
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[str, int, str]] = []
        for cls in self.classes:
            folder = os.path.join(root, cls)
            for fn in os.listdir(folder):
                if fn.lower().endswith(".csv"):
                    path = os.path.join(folder, fn)
                    self.samples.append((path, self.class_to_idx[cls], fn))

    def default_loader(self, path: str) -> torch.Tensor:
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        if arr.ndim == 1 and arr.size == 64:
            arr = arr.reshape(8, 8)
        tensor = torch.tensor(arr, dtype=torch.float32)
        return tensor.unsqueeze(0)   # still in [0,255]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label, fn = self.samples[idx]
        img = self.loader(path)           # raw [0,255]
        if self.transform is not None:
            img = self.transform(img)     # now in [0,1]
        return img, label, fn

    def __len__(self) -> int:
        return len(self.samples)


class Feature(ABC):
    def __init__(self, name: str, properties: dict):
        self.name = name
        self.props = properties

    @abstractmethod
    def apply_bounds(
        self,
        net: Marabou.Network,
        inp_var: np.ndarray,
        image: torch.Tensor,
        delta: float,
        mask: Optional[np.ndarray] = None
    ) -> None:
        pass


# ── PixelFeature.apply_bounds now normalizes into [0,1] before clamping ──────
class PixelFeature(Feature):
    def apply_bounds(
        self,
        net: Marabou.Network,
        inp_var: np.ndarray,
        image: torch.Tensor,
        delta: float,
        mask: Optional[np.ndarray] = None
    ) -> None:
        raw = self.props.get("delta_type", "absolute")
        raw_domain = self.props.get("input_domain", {"min": 0, "max": 255})

        # first convert the domain [0,255] → [0,1]:
        domain_min = raw_domain["min"] / 255.0      # = 0.0
        domain_max = raw_domain["max"] / 255.0      # = 1.0

        # `image` coming in here must already be normalized to [0,1],
        # because we applied normalize_tensor() at dataset‐load time.
        # So `image[0,i,j]` is ∈ [0,1].
        _, h, w = image.shape

        for i in range(h):
            for j in range(w):
                v = float(image[0, i, j])   # normalized pixel ∈ [0,1]
                if mask is not None and not mask[i, j]:
                    lo = hi = v
                else:
                    if raw == "absolute":
                        lo, hi = v - delta, v + delta
                    elif raw == "relative":
                        lo = v * (1 - delta)
                        hi = v * (1 + delta)
                    else:
                        raise ValueError(f"Unknown delta_type {raw}")

                    # clip into [domain_min, domain_max] = [0,1]
                    lo = max(lo, domain_min)
                    hi = min(hi, domain_max)

                net.setLowerBound(inp_var[i][j], lo)
                net.setUpperBound(inp_var[i][j], hi)


def get_mask_if_needed(mask_cfg: dict, image: torch.Tensor) -> Optional[np.ndarray]:
    if mask_cfg.get("compute_mask", False):
        from ITD import compute_meaningful_mask
        return compute_meaningful_mask(image.numpy().squeeze(0))
    return None


class SingleCSVImage(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.fn = os.path.basename(path)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        if idx != 0:
            raise IndexError("Index out of range for SingleCSVImage (len=1)")
        arr = np.loadtxt(self.path, delimiter=",", skiprows=1)
        if arr.ndim == 1 and arr.size == 64:
            arr = arr.reshape(8, 8)
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # [0,255]
        tensor = tensor / 255.0                                       # normalize→[0,1]
        return tensor, 0, self.fn


def verify_single_image(
    image: torch.Tensor,
    correct_class: int,
    delta: float,
    features: List[Feature],
    feature_norm: float,
    model_path: str,
    verbosity: int,
    marabou_opts: Dict[str, Any]
) -> str:
    net = Marabou.read_onnx(model_path)
    inp_var = net.inputVars[0][0]
    out_var = net.outputVars[0][0]

    # image is already ∈[0,1], because either CSVImageFolder applied normalize_tensor
    # or SingleCSVImage did `tensor = tensor/255.0` above.
    orig_out = net.evaluateWithoutMarabou([image.numpy()])[0][0][0]
    pred = 1 if orig_out >= 0 else 0

    mask_cfg = {f.name: f.props.get("mask", {}) for f in features}
    mask = get_mask_if_needed(mask_cfg.get("pixel", {}), image)

    for feat in features:
        feat_norm = normalize_delta(delta, feature_norm, feat.props.get("delta_type", "absolute"))
        print(f"Delta (mm): {delta}, Delta normalized: {feat_norm:.4f}")
        feat.apply_bounds(net, inp_var, image, feat_norm, mask)

    coeff = 1.0 if pred == 1 else -1.0
    net.addInequality([out_var[0]], [coeff], 0.0)

    opts = Marabou.createOptions(**marabou_opts)
    exitCode, vals, stats = net.solve(options=opts)



    if exitCode == "sat":
        return "sat"
    elif exitCode == "unsat":
        return "unsat"
    else:
        return "TIMEOUT"


def sample_perturbed_image(image: np.ndarray, delta: float) -> np.ndarray:
    pert = image.copy()
    noise = np.random.uniform(-delta, delta, size=image.shape)
    return pert + noise


def plot_results(sat: int, unsat: int, timeout: int, raw_delta: float) -> plt.Figure:
    total = sat + unsat + timeout
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    if total == 0:
        ax.text(0.5, 0.5, "No data to plot.", ha="center", va="center")
    else:
        s_pct = 100 * sat / total
        u_pct = 100 * unsat / total
        t_pct = 100 * timeout / total
        x = [raw_delta]
        ax.bar(x, [s_pct], 0.5, label="SAT")
        ax.bar(x, [u_pct], 0.5, bottom=[s_pct], label="UNSAT")
        ax.bar(x, [t_pct], 0.5, bottom=[s_pct + u_pct], label="TIMEOUT")
        ax.set_xlabel("Δ (raw)")
        ax.set_ylabel("% samples")
        ax.legend()

    fig.tight_layout()
    return fig


def run_tof(
    json_cfg: str,
    delta: float,
    perturb_all: bool,
    progress_callback: Optional[Callable[[int,int,int,int,int],None]] = None,
    verbosity: int = 0,
    marabou_opts: Optional[Dict[str, Any]] = None,
    stop_requested: Optional[Callable[[], bool]] = None
) -> Tuple[int, int, int]:
    # load config
    cfg = load_config(json_cfg)
    if marabou_opts is None:
        marabou_opts = cfg.get("marabou", {}).copy()
    
    # Force the GUI’s requested verbosity into Marabou’s options:
    marabou_opts["verbosity"] = verbosity

    # ── hard‐code normalization factor here, not in JSON ──
    norm = 500.0
    features_cfg = cfg.get("features", [])

    features: List[Feature] = []
    for fc in features_cfg:
        prop = fc.get("properties", {})
        if fc["name"] == "pixel":
            features.append(PixelFeature(fc["name"], prop))
        else:
            raise NotImplementedError(f"Feature type '{fc['name']}' not supported yet.")

    img_dir = cfg["file_paths"]["image_directory"]
    model_path = cfg["file_paths"]["onnx_model"]
    # verbosity = cfg.get("marabou", {}).get("verbosity", 0)
    plot_flag = cfg.get("plot", False)

    if stop_requested and stop_requested():
        return 0, 0, 0

    if os.path.isfile(img_dir) and img_dir.lower().endswith(".csv"):
        dataset: Dataset = SingleCSVImage(img_dir)
    else:
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        dataset = CSVImageFolder(
            root=img_dir,
            transform=normalize_tensor
        )

    sat_cnt = unsat_cnt = timeout_cnt = 0
    total = len(dataset)
    for idx, (img, lb, fn) in enumerate(dataset, start=1):
        if stop_requested and stop_requested():
            break
        print(f"Verifying {fn}...")
        res = verify_single_image(
            image=img,
            correct_class=lb,
            delta=delta,
            features=features,
            feature_norm=norm,
            model_path=model_path,
            verbosity=verbosity,
            marabou_opts=marabou_opts
        )
        if res == "sat":
            sat_cnt += 1
        elif res == "unsat":
            unsat_cnt += 1
        else:
            timeout_cnt += 1

        if progress_callback is not None:
            progress_callback(idx, total, sat_cnt, unsat_cnt, timeout_cnt)
        if stop_requested and stop_requested():
            break

    print(f"Results: {sat_cnt} SAT, {unsat_cnt} UNSAT, {timeout_cnt} TIMEOUT over {total} samples.")

    if plot_flag:
        fig = plot_results(sat_cnt, unsat_cnt, timeout_cnt, delta)
        out_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        out_png = out_tmp.name
        out_tmp.close()
        fig.savefig(out_png, format="png", bbox_inches="tight")
        plt.close(fig)
        print(f"__PLOT__:{out_png}")

    return sat_cnt, unsat_cnt, timeout_cnt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TOF verifier runner")
    parser.add_argument("config", type=str, help="Path to JSON config file")
    parser.add_argument("delta", type=float, help="Raw delta value to test")
    parser.add_argument(
        "--perturb_all_pixels",
        action="store_true",
        help="Apply perturbation to all pixels, ignoring mask"
    )
    parser.add_argument(
       "-v", "--verbosity",
       type=int, default=2,
       help="Marabou verbosity level (0=mute, >0=more stats)"
   )
    args = parser.parse_args()
    run_tof(
     args.config,
     args.delta,
     args.perturb_all_pixels,
     verbosity=args.verbosity
   )
