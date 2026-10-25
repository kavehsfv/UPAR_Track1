"""Post-training: per-attribute bias calibration + weight export.

Track 1 decision rule is fixed (p > 0.5), so we calibrate LOGITS with a
per-attribute bias b_a: p' = sigmoid(logit(p) + b_a), chosen to maximize
Challenge_Avg on val via coordinate ascent over a bias grid.

Track 2 uses probabilities in an attribute-weighted L1 distance; we select
(raw vs calibrated probs) x (weight exponent) by mADM on val.

Outputs:
  export/par_model.pt   (visual + head + t1_bias + t2_weights, fp32 state dict)
  export/calib.json     (biases, weights, mode, val metrics)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (  # noqa: E402
    load_gt,
    load_ids,
    load_queries,
)
from metrics import (  # noqa: E402
    adm_weighted_distances,
    l1_distances,
    track1_metrics,
    track2_metrics,
)
from train import infer_probs  # noqa: E402
from vit import PARModel, VisionTransformer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    ap = argparse.ArgumentParser()
    _HERE = Path(__file__).resolve().parent
    ap.add_argument("--run", type=str, default=str(_HERE / "runs" / "clip_vitb16"))
    ap.add_argument("--out", type=str, default=str(_HERE / "runs" / "clip_vitb16" / "export"))
    ap.add_argument("--ckpt", type=str, default=None,
                    help="checkpoint to calibrate (default: <run>/best.pt)")
    ap.add_argument("--track", type=int, choices=[1, 2], default=1,
                    help="1: export T1-optimal bias; 2: export bias consistent "
                         "with the best T2 distance mode (zeros if mode=raw)")
    args = ap.parse_args()
    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt) if args.ckpt else run / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"loaded {ckpt_path} (epoch {ckpt['epoch']})")

    device = torch.device("cuda")
    vit = VisionTransformer()
    vit.load_state_dict(ckpt["visual"], strict=True)
    model = PARModel(vit)
    model.head.load_state_dict(ckpt["head"], strict=True)
    model = model.to(device).eval()

    val_paths, val_gt = load_gt(DATA / "annotations/task1/val/gt.csv")
    probs = infer_probs(model, val_paths, device)
    np.save(out / "val_probs.npy", probs)

    base = track1_metrics(val_gt, probs)
    print("raw:", {k: round(v, 4) for k, v in base.items()})

    # ---------------- per-attribute bias calibration (maximize Challenge_Avg)
    logits = logit(probs)
    best_b = np.zeros(40, dtype=np.float32)
    best_t1: dict[str, float] = dict(base)
    best_score = float(base["Challenge_Avg"])
    for T in (1.0, 2.0):
        lg = logits / T
        b = np.zeros(40, dtype=np.float32)
        grid = np.linspace(-3, 3, 17)
        for _ in range(2):  # coordinate ascent sweeps
            for a in range(40):
                scores = np.empty(len(grid))
                for gi, g in enumerate(grid):
                    b_try = b.copy()
                    b_try[a] = g
                    p_try = sigmoid(lg + b_try)
                    scores[gi] = track1_metrics(val_gt, p_try)["Challenge_Avg"]
                b[a] = float(grid[int(np.argmax(scores))])
        t1 = track1_metrics(val_gt, sigmoid(lg + b))
        print(f"T={T}: Challenge_Avg={t1['Challenge_Avg']:.4f}")
        if t1["Challenge_Avg"] > best_score:
            best_t1, best_score, best_b = t1, float(t1["Challenge_Avg"]), b
    print("calibrated:", {k: round(v, 4) for k, v in best_t1.items()})
    p_cal = sigmoid(logits + best_b)

    # ---------------- Track 2 distance selection (maximize mADM)
    ids = load_ids(DATA / "annotations/task2/val/ids.csv")
    Q = load_queries(DATA / "annotations/task2/val/queries.csv")
    qids = np.asarray([ids[p] for p in val_paths], dtype=np.int64)
    gt8 = val_gt.astype(np.int8)

    tr_paths, tr_gt = load_gt(DATA / "annotations/task1/train/gt.csv")
    pos_rate = tr_gt.mean(0).astype(np.float64)

    best = {"mode": None, "madm": -1.0, "weights": None, "metrics": None}
    for mode, p_use in (("raw", probs), ("cal", p_cal)):
        D = l1_distances(Q.astype(np.float32), p_use)
        t2 = track2_metrics(D, qids, gt8)
        print(f"T2[l1_{mode}]: mADM={t2['mADM']:.4f} mAP={t2['mAP']:.4f} R1={t2['Rank-1']:.4f}")
        if t2["mADM"] > best["madm"]:
            best = {"mode": mode, "madm": float(t2["mADM"]), "weights": None, "metrics": t2}
        for w_exp in (0.5, 1.0):
            w = ((1.0 / np.maximum(pos_rate, 1e-4)) ** w_exp).astype(np.float32)
            w /= w.mean()
            D = adm_weighted_distances(Q.astype(np.float32), p_use, w)
            t2 = track2_metrics(D, qids, gt8)
            print(f"T2[{mode} w_exp={w_exp}]: mADM={t2['mADM']:.4f} mAP={t2['mAP']:.4f} R1={t2['Rank-1']:.4f}")
            if t2["mADM"] > best["madm"]:
                best = {"mode": mode, "madm": float(t2["mADM"]), "weights": w, "metrics": t2}
    print("T2 best:", best["mode"], "w=" + ("none" if best["weights"] is None else "custom"),
          f"mADM={best['madm']:.4f}")

    # ---------------- export
    # The T2 wrapper applies the same bias as T1 inside predict_attributes;
    # for a T2 export whose best distance mode is "raw", the bias must be
    # zeros so the exported model reproduces the selected configuration.
    export_bias = best_b if (args.track == 1 or best["mode"] == "cal") \
        else np.zeros(40, dtype=np.float32)
    torch.save({
        "visual": ckpt["visual"],
        "head": ckpt["head"],
        "epoch": ckpt["epoch"],
        "t1_bias": torch.from_numpy(export_bias),
        "t2_weights": (torch.ones(40) if best["weights"] is None
                       else torch.from_numpy(best["weights"].astype(np.float32))),
    }, out / "par_model.pt")
    (out / "calib.json").write_text(json.dumps({
        "checkpoint": str(ckpt_path),
        "epoch": int(ckpt["epoch"]),
        "t1_bias": best_b.tolist(),
        "t2_mode": best["mode"],
        "t2_weights": ((np.ones(40) if best["weights"] is None else best["weights"]).tolist()),
        "raw_t1": base,
        "calibrated_t1": best_t1,
        "t2": {k: v for k, v in best["metrics"].items() if k != "n_eval"},
    }, indent=2))
    print(f"exported {out}/par_model.pt + calib.json")


if __name__ == "__main__":
    main()
