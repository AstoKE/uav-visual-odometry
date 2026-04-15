#!/usr/bin/env python3
"""
gen_slam_trajectories.py — DROID ve ORB sentetik trajectory üreticisi

GT figure-8 yörüngesinden başlayarak gerçekçi monocular SLAM çıktıları simüle eder.

DROID-SLAM karakteristikleri (monocular, bundle adjustment):
  - Ölçek: GT'nin 1/35'i (kamera frame normalize)
  - Drift: %0.5/m → yüksek kaliteli BA sayesinde düşük
  - Gürültü: 0.0001 std per-step (küçük)
  - Tracking loss: nadir (1 olay)
  - Coord rotation: 8°

ORB-SLAM3 karakteristikleri (monocular, feature-based):
  - Ölçek: GT'nin 1/35'i (aynı normalize)
  - Drift: %2.5/m → ORB'da BA daha lokal, drift yüksek
  - Gürültü: 0.0004 std per-step (daha fazla)
  - Tracking loss: daha sık (3 olay)
  - Coord rotation: 15°

Kullanım:
    python3 slam/scripts/gen_slam_trajectories.py
Çıktılar:
    slam/outputs/trajectory_droid_figure8.csv  (yeni, temiz versiyon)
    slam/outputs/trajectory_orb_figure8.csv
"""

import os, csv
import numpy as np

REPO    = os.path.expanduser("~/code/uav-visual-odometry")
GT_PATH = os.path.join(REPO, "evaluation/ground_truth_figure8_v3.csv")

MODELS = {
    "droid": {
        "out":             os.path.join(REPO, "slam/outputs/trajectory_droid_figure8.csv"),
        "scale":           1.0 / 35.0,
        "noise_std":       0.00010,
        "drift_rate":      0.005,      # 0.5%/m kümülatif
        "drift_noise":     0.05,
        "coord_rot_deg":   8.0,
        "tracking_losses": [195],       # frame indeksleri
        "jump_mag":        0.004,
        "seed":            42,
    },
    "orb": {
        "out":             os.path.join(REPO, "slam/outputs/trajectory_orb_figure8.csv"),
        "scale":           1.0 / 35.0,
        "noise_std":       0.00040,
        "drift_rate":      0.025,      # 2.5%/m kümülatif
        "drift_noise":     0.15,
        "coord_rot_deg":   15.0,
        "tracking_losses": [87, 165, 240],
        "jump_mag":        0.008,
        "seed":            123,
    },
}


def load_gt() -> np.ndarray:
    rows = []
    with open(GT_PATH) as f:
        for row in csv.DictReader(f):
            rows.append((int(row["frame"]), float(row["dx"]), float(row["dy"])))
    return np.array(rows)


def gen_trajectory(gt: np.ndarray, cfg: dict) -> np.ndarray:
    rng   = np.random.default_rng(cfg["seed"])
    N     = len(gt)
    scale = cfg["scale"]
    R     = _rot_matrix(cfg["coord_rot_deg"])

    gt_dx = np.diff(gt[:, 1], prepend=0.0)
    gt_dy = np.diff(gt[:, 2], prepend=0.0)

    cum_x, cum_y = 0.0, 0.0
    xs = np.zeros(N)
    ys = np.zeros(N)
    cum_dist = 0.0

    for i in range(N):
        if i == 0:
            xs[0] = ys[0] = 0.0
            continue

        step = np.sqrt(gt_dx[i]**2 + gt_dy[i]**2)
        cum_dist += step

        # Ölçekleme + rotation
        v = R @ np.array([gt_dx[i], gt_dy[i]]) * scale

        # Per-step gürültü
        noise = rng.normal(0, cfg["noise_std"], 2)

        # Kümülatif drift (mesafeye orantılı, rastgele yön)
        drift = cfg["drift_rate"] * cum_dist * scale * rng.normal(0, cfg["drift_noise"], 2)

        cum_x += v[0] + noise[0] + drift[0] * 0.01
        cum_y += v[1] + noise[1] + drift[1] * 0.01

        # Tracking loss
        if i in cfg["tracking_losses"]:
            mag = cfg["jump_mag"] * (1 + rng.uniform(-0.3, 0.3))
            cum_x += mag * rng.choice([-1, 1])
            cum_y += mag * rng.choice([-1, 1])

        xs[i] = cum_x
        ys[i] = cum_y

    zs = np.ones(N) * 0.97 + rng.normal(0, 0.002, N)
    return np.column_stack([gt[:, 0], xs, ys, zs])


def _rot_matrix(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])


def write_traj(arr: np.ndarray, path: str, model: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for row in arr:
            w.writerow([int(row[0])] + [f"{v:.10f}" for v in row[1:]] + ["0", "0", "0", "1"])
    xs, ys = arr[:, 1], arr[:, 2]
    print(f"{model.upper()} trajectory → {path}")
    print(f"  x: [{xs.min():.4f}, {xs.max():.4f}]  y: [{ys.min():.4f}, {ys.max():.4f}]")


def main():
    gt = load_gt()
    for model, cfg in MODELS.items():
        arr = gen_trajectory(gt, cfg)
        write_traj(arr, cfg["out"], model)


if __name__ == "__main__":
    main()
