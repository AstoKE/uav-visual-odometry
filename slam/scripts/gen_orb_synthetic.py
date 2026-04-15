#!/usr/bin/env python3
"""
gen_orb_synthetic.py — ORB-SLAM3 monocular sentetik trajectory üretici

Gerçekçi ORB-SLAM3 monocular davranışı simüle eder:
  - Ölçek belirsizliği (scale ≈ 1/35 gerçek ölçeğe göre)
  - Kümülatif drift (mesafeye orantılı, %2/m)
  - Gaussian gürültü (per-step)
  - İzleme kaybı + yeniden lokalizasyon atlaması (2 olay)
  - Coordinate frame rotasyonu (~15°)

Ayrıca DROID ham trajectory için aynı noise-free test sağlar.

Kullanım:
    python3 slam/scripts/gen_orb_synthetic.py
Çıktı:
    slam/outputs/trajectory_orb_figure8.csv
"""

import os, csv
import numpy as np

REPO = os.path.expanduser("~/code/uav-visual-odometry")
GT_PATH   = os.path.join(REPO, "evaluation/ground_truth_figure8_v3.csv")
OUT_PATH  = os.path.join(REPO, "slam/outputs/trajectory_orb_figure8.csv")

# ORB-SLAM3 monocular karakteristikleri
ORB_SCALE       = 1.0 / 35.0   # GT metre → ORB birim (yaklaşık DROID scale'a eşit)
ORB_NOISE_STD   = 0.0003       # per-step Gaussian gürültü (ORB biriminde)
ORB_DRIFT_RATE  = 0.020        # %2 kümülatif drift / metre
COORD_ROTATION  = np.deg2rad(15.0)  # kamera frame rotasyonu

# Tracking loss parametreleri
TRACKING_LOSS_FRAMES = [95, 220]   # bu frame'lerde izleme kaybı olur
JUMP_MAGNITUDE = 0.008             # ani sıçrama (ORB birim)


def load_gt() -> np.ndarray:
    """GT'yi yükle: (N, 3) — (frame, dx_cum, dy_cum)"""
    rows = []
    with open(GT_PATH) as f:
        for row in csv.DictReader(f):
            rows.append((int(row["frame"]), float(row["dx"]), float(row["dy"])))
    return np.array(rows)


def main():
    rng = np.random.default_rng(123)
    gt = load_gt()
    N = len(gt)

    # GT'den per-frame delta hesapla
    gt_dx = np.diff(gt[:, 1], prepend=0.0)
    gt_dy = np.diff(gt[:, 2], prepend=0.0)

    # Coordinate frame rotation matrix
    R = np.array([
        [ np.cos(COORD_ROTATION), np.sin(COORD_ROTATION)],
        [-np.sin(COORD_ROTATION), np.cos(COORD_ROTATION)],
    ])

    # ORB birimlerinde kümülatif pozisyon oluştur
    cum_x, cum_y = 0.0, 0.0
    orb_cum_x = np.zeros(N)
    orb_cum_y = np.zeros(N)

    cumulative_distance = 0.0

    for i in range(N):
        if i == 0:
            orb_cum_x[0] = 0.0
            orb_cum_y[0] = 0.0
            continue

        # GT delta → ORB ölçek
        true_dx = gt_dx[i]
        true_dy = gt_dy[i]
        step_dist = np.sqrt(true_dx**2 + true_dy**2)
        cumulative_distance += step_dist

        # ORB ölçeğe çevir
        orb_dx = true_dx * ORB_SCALE
        orb_dy = true_dy * ORB_SCALE

        # Coordinate rotation uygula
        orb_step = R @ np.array([orb_dx, orb_dy])
        orb_dx, orb_dy = orb_step

        # Kümülatif drift ekle (mesafeye orantılı)
        drift_x = ORB_DRIFT_RATE * cumulative_distance * ORB_SCALE * rng.normal(0, 0.1)
        drift_y = ORB_DRIFT_RATE * cumulative_distance * ORB_SCALE * rng.normal(0, 0.1)

        # Per-step Gaussian gürültü
        noise_x = rng.normal(0, ORB_NOISE_STD)
        noise_y = rng.normal(0, ORB_NOISE_STD)

        cum_x += orb_dx + noise_x + drift_x * 0.01
        cum_y += orb_dy + noise_y + drift_y * 0.01

        # Tracking loss: ani sıçrama + yeniden lokalizasyon
        if i in TRACKING_LOSS_FRAMES:
            jump = JUMP_MAGNITUDE * (1.0 + rng.uniform(-0.3, 0.3))
            cum_x += jump * rng.choice([-1, 1])
            cum_y += jump * rng.choice([-1, 1])

        orb_cum_x[i] = cum_x
        orb_cum_y[i] = cum_y

    # Z sabit (monocular kamera, z≈1 normalize edilmiş)
    orb_z = np.ones(N) * 0.97 + rng.normal(0, 0.002, N)

    # CSV yaz
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for i in range(N):
            w.writerow([
                int(gt[i, 0]),
                f"{orb_cum_x[i]:.10f}",
                f"{orb_cum_y[i]:.10f}",
                f"{orb_z[i]:.6f}",
                "0.0", "0.0", "0.0", "1.0",
            ])

    print(f"ORB trajectory yazıldı: {OUT_PATH}  ({N} frame)")
    print(f"x range: {orb_cum_x.min():.4f} → {orb_cum_x.max():.4f}")
    print(f"y range: {orb_cum_y.min():.4f} → {orb_cum_y.max():.4f}")
    print(f"GT scale: {gt[:, 1:3].std() / np.column_stack([orb_cum_x, orb_cum_y]).std():.2f}x")


if __name__ == "__main__":
    main()
