#!/usr/bin/env python3
"""
run_demo.py — 2000-frame yarışma demo senaryosu

Senaryo:
  - 2000 frame, figure-8 + drift yörüngesi
  - İlk %20 (400 frame) → health=1 (kalibrasyon penceresi)
  - Kalan %80 → competition health senaryosu (burst + blackout)
  - SLAMPoseEstimator ile DROID taklit pozisyon kestirimi
  - Çıktı: trajectory plot + error graph → competition/results/demo_*.png

Kullanım:
    python3 competition/run_demo.py
    python3 competition/run_demo.py --frames 2000 --seed 7 --out competition/results
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

_REPO = os.path.expanduser("~/code/uav-visual-odometry")
sys.path.insert(0, _REPO)

from competition.slam_pose_estimator import SLAMPoseEstimator
from competition.simulate_health     import make_health_flags
from competition.score_official      import score_summary

RESULTS_DIR = os.path.join(_REPO, "competition/results")


# ── Yörünge üreticileri ───────────────────────────────────────────────────────

def gen_gt_figure8(n: int, ax: float = 4.5, ay: float = 3.5,
                   period: float = 25.0, step_dt: float = 0.08) -> np.ndarray:
    """
    Gerçek dünya figure-8 yörüngesi (metre).
    Döndürür: (n, 3) array [x, y, z=0]
    """
    pos = np.zeros((n, 3))
    for i in range(n):
        t = i * step_dt
        theta = 2.0 * math.pi * t / period
        pos[i, 0] = ax * math.sin(theta)
        pos[i, 1] = ay * math.sin(2.0 * theta)
    return pos


def gen_slam_deltas(gt_pos: np.ndarray, rng,
                    noise_std: float = 0.00010,
                    drift_rate: float = 0.005,
                    coord_rot_deg: float = 8.0,
                    scale: float = 1 / 35.0) -> np.ndarray:
    """
    GT pozisyonlarından DROID benzeri SLAM deltaları üret.
    Döndürür: (n, 2) ham SLAM delta dizisi (ölçeksiz)
    """
    n = len(gt_pos)
    # GT → SLAM ölçek + koordinat rotasyonu
    angle = math.radians(coord_rot_deg)
    R = np.array([[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle),  math.cos(angle)]])

    # Ham SLAM pozisyonları
    slam_pos = np.zeros((n, 2))
    drift_x = drift_y = 0.0
    for i in range(n):
        raw = scale * R @ gt_pos[i, :2]
        drift_x += rng.normal(0, drift_rate * 0.01)
        drift_y += rng.normal(0, drift_rate * 0.01)
        noise = rng.normal(0, noise_std, 2)
        slam_pos[i] = raw + noise + np.array([drift_x, drift_y])

    # Artımlı delta
    deltas = np.zeros((n, 2))
    deltas[1:] = np.diff(slam_pos, axis=0)
    return deltas


def gen_competition_health(n: int, seed: int) -> np.ndarray:
    """
    İlk %20 = health=1 (kalibrasyon), geri kalan = competition senaryosu.
    """
    calib_end = int(n * 0.20)
    flags = np.zeros(n, dtype=np.int8)
    flags[:calib_end] = 1

    # Geri kalan kısım için competition senaryosu üret
    rest = n - calib_end
    rest_flags, _ = make_health_flags(rest, seed=seed, scenario="competition")
    flags[calib_end:] = rest_flags
    return flags


# ── Demo çalıştırma ───────────────────────────────────────────────────────────

def run_demo(n_frames: int = 2000, seed: int = 42,
             out_dir: str = RESULTS_DIR) -> dict:
    rng = np.random.default_rng(seed)

    # GT yörüngesi
    gt_pos = gen_gt_figure8(n_frames)

    # SLAM deltaları (DROID benzeri)
    deltas = gen_slam_deltas(gt_pos, rng)

    # Health flagleri
    health = gen_competition_health(n_frames, seed=seed)

    # Estimator
    est = SLAMPoseEstimator(calib_min_frames=30, calib_update_every=50)

    rows = []
    est_pos = np.zeros((n_frames, 3))

    for i in range(n_frames):
        h = int(health[i])
        ref = (float(gt_pos[i, 0]), float(gt_pos[i, 1]), 0.0) if h == 1 else None
        wx, wy, wz = est.update(float(deltas[i, 0]), float(deltas[i, 1]),
                                ref_pos=ref, health=h)
        est_pos[i] = [wx, wy, wz]

        rows.append({
            "frame":   i,
            "health":  h,
            "gt_x":    round(float(gt_pos[i, 0]), 4),
            "gt_y":    round(float(gt_pos[i, 1]), 4),
            "est_x":   round(wx, 4),
            "est_y":   round(wy, 4),
            "est_z":   round(wz, 4),
        })

    # CSV kaydet
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "demo_2000frame.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Metrikler
    dead_idx = [i for i in range(n_frames) if health[i] == 0]
    errs_2d = np.sqrt(
        (est_pos[:, 0] - gt_pos[:, 0]) ** 2 +
        (est_pos[:, 1] - gt_pos[:, 1]) ** 2
    )
    rmse_2d_dead = float(np.sqrt(np.mean(errs_2d[dead_idx] ** 2))) if dead_idx else float("nan")
    final_drift  = float(errs_2d[-1])
    max_drift    = float(np.max(errs_2d[dead_idx])) if dead_idx else float("nan")

    # Resmi yarışma skoru — §9.2 Denklem 2 (MAE_3D)
    # health=1'de referans gönderilir (sıfır hata) → payda = N_toplam
    est_tuples = [(float(est_pos[i, 0]), float(est_pos[i, 1]), float(est_pos[i, 2]))
                  for i in range(n_frames)]
    ref_tuples = [(float(gt_pos[i, 0]),  float(gt_pos[i, 1]),  0.0)
                  for i in range(n_frames)]
    sc = score_summary(est_tuples, ref_tuples, health.tolist())
    # Optimal strateji hesabı: health=1'de sıfır hata
    dead_err_sum = sum(
        math.sqrt((est_pos[i, 0] - gt_pos[i, 0])**2 +
                  (est_pos[i, 1] - gt_pos[i, 1])**2 +
                  est_pos[i, 2]**2)
        for i in dead_idx
    )
    mae_3d_official = dead_err_sum / n_frames  # payda = toplam frame

    metrics = {
        "n_frames":         n_frames,
        "n_dead":           len(dead_idx),
        "calibrated":       est.calibrated,
        "calib_n":          est.calib_data_count,
        # ── Resmi yarışma skoru ───────────────────────────────────────────────
        "mae_3d_official":  round(mae_3d_official, 4),  # §9.2 Denklem 2
        "mae_3d_dead_only": round(sc["mae_3d_dead"], 4),
        # ── İç metrikler ─────────────────────────────────────────────────────
        "rmse_2d_dead":     rmse_2d_dead,
        "final_drift":      final_drift,
        "max_drift":        max_drift,
    }

    # Grafikler
    _plot_trajectory(gt_pos, est_pos, health, n_frames, out_dir)
    _plot_error(errs_2d, health, n_frames, out_dir)

    print(f"\n{'='*55}")
    print(f"Demo Sonuçları ({n_frames} frame)")
    print(f"{'='*55}")
    print(f"Health=0 frame   : {len(dead_idx)} ({100*len(dead_idx)/n_frames:.1f}%)")
    print(f"Kalibre          : {'Evet' if est.calibrated else 'Hayır'}  (n={est.calib_data_count})")
    print(f"MAE_3D★ (resmi)  : {mae_3d_official:.4f} m  ← §9.2 yarışma skoru")
    print(f"MAE_3D (dead)    : {sc['mae_3d_dead']:.4f} m")
    print(f"RMSE_2D (dead)   : {rmse_2d_dead:.4f} m")
    print(f"Final drift      : {final_drift:.4f} m")
    print(f"Max drift        : {max_drift:.4f} m")
    print(f"\nDosyalar:")
    print(f"  CSV  : {csv_path}")
    print(f"  Traj : {os.path.join(out_dir, 'demo_trajectory.png')}")
    print(f"  Error: {os.path.join(out_dir, 'demo_error.png')}")

    return metrics


# ── Grafik fonksiyonları ──────────────────────────────────────────────────────

def _plot_trajectory(gt_pos, est_pos, health, n_frames, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("UYARI: matplotlib yok, grafikler atlandı.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # GT yörüngesi
    ax.plot(gt_pos[:, 0], gt_pos[:, 1],
            color="#2196F3", linewidth=1.5, alpha=0.7, label="GT Yörüngesi", zorder=2)

    # Estimator — health=1 ve health=0 bölümlerini ayrı renkte çiz
    h1_x, h1_y = [], []
    h0_x, h0_y = [], []
    for i in range(n_frames):
        if health[i] == 1:
            h1_x.append(est_pos[i, 0]); h1_y.append(est_pos[i, 1])
        else:
            h0_x.append(est_pos[i, 0]); h0_y.append(est_pos[i, 1])

    if h1_x:
        ax.scatter(h1_x, h1_y, s=3, color="#4CAF50", alpha=0.6,
                   label="Tahmin (health=1)", zorder=3)
    if h0_x:
        ax.scatter(h0_x, h0_y, s=3, color="#F44336", alpha=0.5,
                   label="Tahmin (health=0)", zorder=3)

    # Başlangıç / bitiş işaretleri
    ax.plot(gt_pos[0, 0], gt_pos[0, 1], "go", ms=10, label="Başlangıç", zorder=5)
    ax.plot(gt_pos[-1, 0], gt_pos[-1, 1], "rs", ms=10, label="Bitiş", zorder=5)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(f"Yörünge Karşılaştırma — {n_frames} Frame\n"
                 f"(İlk %20 = health=1 kalibrasyon, geri kalan = competition senaryosu)",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    out = os.path.join(out_dir, "demo_trajectory.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_error(errs_2d, health, n_frames, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    frames = np.arange(n_frames)
    calib_end = int(n_frames * 0.20)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Üst panel: 2D hata
    ax1 = axes[0]
    # health=0 bölgelerini gölgele
    in_dead = False
    dead_start = 0
    for i in range(n_frames):
        if not in_dead and health[i] == 0:
            in_dead = True
            dead_start = i
        elif in_dead and (health[i] == 1 or i == n_frames - 1):
            ax1.axvspan(dead_start, i, color="#FFCDD2", alpha=0.5)
            in_dead = False
    if in_dead:
        ax1.axvspan(dead_start, n_frames, color="#FFCDD2", alpha=0.5)

    ax1.axvline(calib_end, color="#9C27B0", linestyle="--", linewidth=1.5,
                label=f"Kalibrasyon sonu (frame {calib_end})")
    ax1.axhline(1.0, color="#FF5722", linestyle=":", linewidth=1.2,
                label="1.0 m eşiği", alpha=0.8)
    ax1.plot(frames, errs_2d, color="#2196F3", linewidth=0.8, alpha=0.9,
             label="2D Hata (m)")

    ax1.set_ylabel("2D Hata (m)", fontsize=11)
    ax1.set_title(f"Konum Hatası — {n_frames} Frame Demo\n"
                  f"(Kırmızı gölge = health=0, mor kesik = kalibrasyon sonu)",
                  fontsize=11)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Alt panel: health flag
    ax2 = axes[1]
    ax2.fill_between(frames, health.astype(float), 0,
                     where=health == 1, color="#4CAF50", alpha=0.6, label="health=1")
    ax2.fill_between(frames, 1, health.astype(float),
                     where=health == 0, color="#F44336", alpha=0.4, label="health=0")
    ax2.axvline(calib_end, color="#9C27B0", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Frame", fontsize=11)
    ax2.set_ylabel("Health", fontsize=11)
    ax2.set_yticks([0, 1])
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "demo_error.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


# ── Giriş noktası ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="2000-frame yarışma demo senaryosu")
    parser.add_argument("--frames", type=int,  default=2000)
    parser.add_argument("--seed",   type=int,  default=42)
    parser.add_argument("--out",    default=RESULTS_DIR)
    args = parser.parse_args()

    run_demo(n_frames=args.frames, seed=args.seed, out_dir=args.out)


if __name__ == "__main__":
    main()
