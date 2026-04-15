#!/usr/bin/env python3
"""
compare_trajectory.py — GT ile ölçeklenmiş SLAM trajectory'yi karşılaştırır.

Adımlar:
  1. GT ve scaled SLAM delta trajectory'leri yükle
  2. Ölçek sonrası kalan koordinat çerçevesi farkını gidermek için
     2D Procrustes hizalaması (SVD tabanlı yalnızca döndürme)
  3. Hizalanmış SLAM'e göre RMSE hesapla
  4. İki grafik üret:
     - evaluation/plots/trajectory_xy.png  (GT vs aligned SLAM, XY düzlemi)
     - evaluation/plots/error_xyz.png      (frame başına hata)

Kullanım:
    python3 ~/code/uav-visual-odometry/evaluation/compare_trajectory.py

Gereksinimler:
    pip install numpy matplotlib  (veya conda install)
"""

import csv
import math
import os
import sys

import numpy as np

# matplotlib backend — headless ortamda Agg kullan
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Yollar ───────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.expanduser("~/code/uav-visual-odometry")
GT_IN       = os.environ.get("GT_IN",
              os.path.join(REPO_ROOT, "evaluation/ground_truth.csv"))
SCALED_IN   = os.environ.get("SCALED_IN",
              os.path.join(REPO_ROOT, "slam/outputs/delta_trajectory_scaled.csv"))
PLOTS_DIR   = os.environ.get("PLOTS_DIR",
              os.path.join(REPO_ROOT, "evaluation/plots"))
METRICS_OUT = os.environ.get("METRICS_OUT",
              os.path.join(REPO_ROOT, "evaluation/metrics/rmse_report.txt"))
# ─────────────────────────────────────────────────────────────────────────────


def load_delta_csv(path: str, xk="dx", yk="dy", zk="dz") -> np.ndarray:
    """CSV'den [N,3] numpy dizisi döndürür."""
    rows = list(csv.DictReader(open(path, newline="")))
    arr  = np.array([[float(r[xk]), float(r[yk]), float(r[zk])] for r in rows])
    return arr


def procrustes_rotation_2d(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    2D Procrustes döndürme matrisini SVD ile bulur (ölçek sabit).

    Argümanlar:
        source : [N, 2]  döndürülecek nokta seti (SLAM scaled)
        target : [N, 2]  hedef nokta seti (GT)
    Döndürür:
        R : [2, 2] döndürme matrisi  (target ≈ source @ R.T)
    """
    # Merkeze al (ilk nokta zaten (0,0) ama genel)
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    s0   = source - mu_s
    t0   = target - mu_t

    # H = S^T @ T
    H = s0.T @ t0
    U, _, Vt = np.linalg.svd(H)

    # Düzeltme: yansıma engelle (det < 0 durumu)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, d])

    R = Vt.T @ D @ U.T
    return R


def rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(errors**2)))


def main() -> None:
    # ── Yükleme ──────────────────────────────────────────────────────────────
    for p in [GT_IN, SCALED_IN]:
        if not os.path.isfile(p):
            print(f"[compare] HATA: Dosya bulunamadi: {p}")
            sys.exit(1)

    gt_delta     = load_delta_csv(GT_IN,     xk="dx", yk="dy", zk="dz")
    slam_scaled  = load_delta_csv(SCALED_IN, xk="dx", yk="dy", zk="dz")

    n = min(len(gt_delta), len(slam_scaled))
    gt_delta    = gt_delta[:n]
    slam_scaled = slam_scaled[:n]

    print(f"[compare] Frame sayisi: {n}")

    # ── 2D Procrustes hizalama ────────────────────────────────────────────────
    # XY düzleminde döndürme: kamera frame vs dunya frame farki
    R2d = procrustes_rotation_2d(slam_scaled[:, :2], gt_delta[:, :2])
    angle_deg = math.degrees(math.atan2(R2d[1, 0], R2d[0, 0]))
    print(f"[compare] Procrustes dongme acisi: {angle_deg:.2f} derece")

    # Hizalanmış SLAM delta pozisyonlari
    slam_xy_aligned = (slam_scaled[:, :2] @ R2d.T)
    slam_aligned    = np.column_stack([slam_xy_aligned, slam_scaled[:, 2]])

    # ── RMSE hesapla ─────────────────────────────────────────────────────────
    errors    = gt_delta - slam_aligned
    rmse_x    = rmse(errors[:, 0])
    rmse_y    = rmse(errors[:, 1])
    rmse_z    = rmse(errors[:, 2])
    rmse_tot  = rmse(np.linalg.norm(errors, axis=1))

    report_lines = [
        "=" * 60,
        "TRAJECTORY KARSILASTIRMA RAPORU",
        "=" * 60,
        "",
        f"Frame sayisi          : {n}",
        f"Procrustes donme acisi: {angle_deg:.2f} derece",
        "",
        "-- RMSE (scale + rotation hizalama sonrasi) --",
        f"  RMSE_x    : {rmse_x:.4f} m",
        f"  RMSE_y    : {rmse_y:.4f} m",
        f"  RMSE_z    : {rmse_z:.4f} m",
        f"  RMSE_total: {rmse_tot:.4f} m",
        "",
        "-- Scaled son frame karsilastirma (hizalama sonrasi) --",
        f"  GT  son frame: dx={gt_delta[-1,0]:.4f}  dy={gt_delta[-1,1]:.4f}",
        f"  SLAM son frame: dx={slam_aligned[-1,0]:.4f}  dy={slam_aligned[-1,1]:.4f}",
        "",
        "NOTLAR:",
        "  * z ekseni karsilastirmasi yalnizca goreli derinlik degisimini",
        "    gosterir (monokulur SLAM metrik derinlik bilmez).",
        "  * Timing modeli: recorder ve move_camera'nin ayni anda basladigi",
        "    varsayilmistir. Gercek offsette RMSE degisebilir.",
        "=" * 60,
    ]
    report = "\n".join(report_lines)
    print()
    print(report)

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        f.write(report + "\n")
    print(f"\n[compare] Metrik raporu: {METRICS_OUT}")

    # ── Grafikler ────────────────────────────────────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)
    frames = np.arange(n)

    # --- 1. XY Trajectory ---
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(gt_delta[:, 0], gt_delta[:, 1],
            "b-o", markersize=2, linewidth=1.5, label="Ground Truth")
    ax.plot(slam_aligned[:, 0], slam_aligned[:, 1],
            "r--s", markersize=2, linewidth=1.5, label="SLAM (scaled + aligned)")

    # Başlangıç ve bitiş işaretleri
    ax.scatter(*gt_delta[0, :2],       color="blue",  s=80, zorder=5, marker="^", label="GT start")
    ax.scatter(*gt_delta[-1, :2],      color="blue",  s=80, zorder=5, marker="v")
    ax.scatter(*slam_aligned[0, :2],   color="red",   s=80, zorder=5, marker="^", label="SLAM start")
    ax.scatter(*slam_aligned[-1, :2],  color="red",   s=80, zorder=5, marker="v")

    ax.set_xlabel("dx (m)")
    ax.set_ylabel("dy (m)")
    ax.set_title(f"Trajectory XY — Ground Truth vs SLAM\n"
                 f"Scale=396.8×, Rotation={angle_deg:.1f}°, RMSE={rmse_tot:.3f} m")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plot1 = os.path.join(PLOTS_DIR, "trajectory_xy.png")
    fig.tight_layout()
    fig.savefig(plot1, dpi=120)
    plt.close(fig)
    print(f"[compare] Plot kaydedildi: {plot1}")

    # --- 2. Per-frame error ---
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    labels_data = [
        ("Error X (m)",     errors[:, 0], "tab:blue"),
        ("Error Y (m)",     errors[:, 1], "tab:orange"),
        ("Error Z (m)",     errors[:, 2], "tab:green"),
        ("Error |total| (m)", np.linalg.norm(errors, axis=1), "tab:red"),
    ]
    rmse_vals = [rmse_x, rmse_y, rmse_z, rmse_tot]

    for ax, (lbl, data, color), rv in zip(axes, labels_data, rmse_vals):
        ax.bar(frames, data, color=color, alpha=0.7, width=0.8)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel(lbl, fontsize=9)
        ax.set_title(f"RMSE = {rv:.4f} m", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    axes[-1].set_xlabel("SLAM Frame")
    fig.suptitle("Per-Frame Trajectory Error (GT - SLAM aligned)", fontsize=11)
    fig.tight_layout()

    plot2 = os.path.join(PLOTS_DIR, "error_xyz.png")
    fig.savefig(plot2, dpi=120)
    plt.close(fig)
    print(f"[compare] Plot kaydedildi: {plot2}")


if __name__ == "__main__":
    main()
