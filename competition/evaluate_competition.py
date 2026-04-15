#!/usr/bin/env python3
"""
evaluate_competition.py — Yarışma formatında tam değerlendirme

Metrikler:
  RMSE_x, RMSE_y, RMSE_z, RMSE_2D, RMSE_3D
  Final drift
  Max drift (tüm test boyunca)
  Recovery time (health=0 başladıktan sonra hatanın 1m altına düşme frame sayısı)
  Avg confidence (health=0 bölümü)

Kullanım:
    python3 competition/evaluate_competition.py
    python3 competition/evaluate_competition.py --est results/est_droid.csv --gt evaluation/ground_truth_figure8_v3.csv
"""

import argparse
import csv
import os
import sys
import numpy as np

_REPO = os.path.expanduser("~/code/uav-visual-odometry")
sys.path.insert(0, _REPO)

RESULTS_DIR = os.path.join(_REPO, "competition/results")
GT_PATH     = os.path.join(_REPO, "evaluation/ground_truth_figure8_v3.csv")


# ── Yükleyiciler ──────────────────────────────────────────────────────────────

def load_est(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k not in ("frame",) else int(v))
                         for k, v in r.items()})
    return rows


def load_gt(path: str) -> dict[int, tuple]:
    gt = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            gt[int(r["frame"])] = (float(r["dx"]), float(r["dy"]), float(r.get("dz", 0)))
    return gt


# ── Metrik hesaplamaları ──────────────────────────────────────────────────────

def compute_all_metrics(rows: list[dict], gt: dict[int, tuple]) -> dict:
    dead   = [r for r in rows if r.get("health", 1) == 0]
    alive  = [r for r in rows if r.get("health", 1) == 1]

    def err3d(row):
        gx, gy, gz = gt.get(row["frame"], (row["gt_x"], row["gt_y"], 0.0))
        ex = row["est_x"] - gx
        ey = row["est_y"] - gy
        ez = row.get("est_z", 0.0) - gz
        return ex, ey, ez

    # RMSE (health=0)
    if dead:
        errs = np.array([err3d(r) for r in dead])
        rmse_x  = float(np.sqrt(np.mean(errs[:, 0]**2)))
        rmse_y  = float(np.sqrt(np.mean(errs[:, 1]**2)))
        rmse_z  = float(np.sqrt(np.mean(errs[:, 2]**2)))
        rmse_2d = float(np.sqrt(np.mean(errs[:, 0]**2 + errs[:, 1]**2)))
        rmse_3d = float(np.sqrt(np.mean(np.sum(errs**2, axis=1))))
    else:
        rmse_x = rmse_y = rmse_z = rmse_2d = rmse_3d = float("nan")

    # Final drift
    last = rows[-1]
    gx, gy, gz = gt.get(last["frame"], (last.get("gt_x", 0), last.get("gt_y", 0), 0))
    final_drift = float(np.sqrt(
        (last["est_x"] - gx)**2 + (last["est_y"] - gy)**2
    ))

    # Max drift (tüm health=0 bölümünde)
    if dead:
        drifts = [np.sqrt((r["est_x"] - gt.get(r["frame"], (r.get("gt_x",0), 0, 0))[0])**2 +
                          (r["est_y"] - gt.get(r["frame"], (0, r.get("gt_y",0), 0))[1])**2)
                  for r in dead]
        max_drift = float(max(drifts))
    else:
        max_drift = float("nan")

    # Recovery time: health=0 başladıktan sonra hatanın < 1m olduğu ilk frame
    recovery_frames = []
    in_dead = False
    dead_start = None
    RECOVERY_THRESHOLD = 1.0  # metre

    for i, r in enumerate(rows):
        h = r.get("health", 1)
        if not in_dead and h == 0:
            in_dead = True
            dead_start = i
        elif in_dead and h == 0:
            gx2, gy2, _ = gt.get(r["frame"], (r.get("gt_x", 0), r.get("gt_y", 0), 0))
            err_2d = float(np.sqrt((r["est_x"] - gx2)**2 + (r["est_y"] - gy2)**2))
            # Recovery: hata eşiğin altına düştüğünde
            if err_2d < RECOVERY_THRESHOLD:
                frames_since_dead = i - dead_start
                recovery_frames.append(frames_since_dead)
                in_dead = False
        elif in_dead and h == 1:
            in_dead = False

    avg_recovery = float(np.mean(recovery_frames)) if recovery_frames else float("nan")
    min_recovery = int(min(recovery_frames)) if recovery_frames else -1
    max_recovery = int(max(recovery_frames)) if recovery_frames else -1

    # Confidence (health=0, eğer sütun varsa)
    conf_vals = [r.get("confidence", float("nan")) for r in dead
                 if not np.isnan(r.get("confidence", float("nan")))]
    avg_conf = float(np.mean(conf_vals)) if conf_vals else float("nan")

    # Drift rejection count
    total_rejected = sum(int(r.get("drift_rejected", 0)) for r in rows[-1:])

    return {
        "n_total":      len(rows),
        "n_dead":       len(dead),
        "n_alive":      len(alive),
        "rmse_x":       rmse_x,
        "rmse_y":       rmse_y,
        "rmse_z":       rmse_z,
        "rmse_2d":      rmse_2d,
        "rmse_3d":      rmse_3d,
        "final_drift":  final_drift,
        "max_drift":    max_drift,
        "recovery_avg": avg_recovery,
        "recovery_min": min_recovery,
        "recovery_max": max_recovery,
        "avg_confidence": avg_conf,
    }


def fmt(v) -> str:
    if isinstance(v, float) and not np.isnan(v):
        return f"{v:.4f}"
    elif isinstance(v, float):
        return "N/A"
    return str(v)


def format_report(metrics: dict, model_label: str = "DROID") -> str:
    m = metrics
    lines = [
        "=" * 55,
        f"Yarışma Değerlendirme — {model_label}",
        "=" * 55,
        "",
        f"Toplam frame     : {m['n_total']}",
        f"Health=0 frame   : {m['n_dead']}  ({100*m['n_dead']/max(m['n_total'],1):.1f}%)",
        "",
        "── RMSE (health=0 bölümü) ──────────────────────────",
        f"  RMSE_x          : {fmt(m['rmse_x'])} m",
        f"  RMSE_y          : {fmt(m['rmse_y'])} m",
        f"  RMSE_z          : {fmt(m['rmse_z'])} m",
        f"  RMSE_2D         : {fmt(m['rmse_2d'])} m",
        f"  RMSE_3D         : {fmt(m['rmse_3d'])} m",
        "",
        "── Drift ────────────────────────────────────────────",
        f"  Final drift     : {fmt(m['final_drift'])} m",
        f"  Max drift       : {fmt(m['max_drift'])} m",
        "",
        "── Recovery (hata < 1.0m olana kadar frame) ────────",
        f"  Avg recovery    : {fmt(m['recovery_avg'])} frame",
        f"  Min recovery    : {m['recovery_min']} frame",
        f"  Max recovery    : {m['recovery_max']} frame",
        "",
        "── Kalite ───────────────────────────────────────────",
        f"  Avg confidence  : {fmt(m['avg_confidence'])}",
        "=" * 55,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--est",   default=os.path.join(RESULTS_DIR, "est_droid.csv"))
    parser.add_argument("--gt",    default=GT_PATH)
    parser.add_argument("--label", default="DROID")
    parser.add_argument("--out",   default=None)
    args = parser.parse_args()

    if not os.path.exists(args.est):
        print(f"HATA: {args.est} bulunamadı", file=sys.stderr)
        sys.exit(1)

    rows    = load_est(args.est)
    gt      = load_gt(args.gt)
    metrics = compute_all_metrics(rows, gt)
    report  = format_report(metrics, args.label)

    print(report)

    out = args.out or os.path.join(RESULTS_DIR, f"eval_{args.label.lower()}.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(report + "\n")
    print(f"\nKaydedildi: {out}")


if __name__ == "__main__":
    main()
