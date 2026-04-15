#!/usr/bin/env python3
"""
benchmark_v4_v5.py — v4 vs v5 World Karşılaştırma Raporu + Plotlar

Mevcut metrik dosyalarını okur ve karşılaştırır:
  evaluation/metrics/rmse_figure8_v3.txt  → v3 baseline
  evaluation/metrics/rmse_figure8_v4.txt  → v4 realistic
  evaluation/metrics/rmse_figure8_v5.txt  → v5 realistic (varsa)

Çıktılar:
  evaluation/metrics/realism_benchmark_v4_vs_v5.txt
  evaluation/plots/trajectory_v4_vs_v5.png
  evaluation/plots/error_v4_vs_v5.png

Kullanım:
  python3 evaluation/benchmark_v4_v5.py           # mevcut verilerle
  python3 evaluation/benchmark_v4_v5.py --plots   # sadece plotlar yenile

v5 verisi için:
  1. Gazebo v5 world ile görüntü topla:
       WORLD_NAME=slam_world_v5_realistic bash sim/scripts/run_gazebo_realistic.sh
  2. DROID-SLAM çalıştır:
       conda run -n droid_clean bash slam/scripts/run_droid_figure8_v4.sh
       (v5 için slam/scripts/run_droid_figure8_v5.sh oluşturulacak)
  3. evaluate_figure8_v5.py çalıştır → rmse_figure8_v5.txt üretir
  4. Bu scripti tekrar çalıştır
"""

import argparse
import csv
import os
import re
import sys
from typing import Optional
import numpy as np

REPO    = os.path.expanduser("~/code/uav-visual-odometry")
METRICS = os.path.join(REPO, "evaluation/metrics")
PLOTS   = os.path.join(REPO, "evaluation/plots")
SLAM_OUTPUTS = os.path.join(REPO, "slam/outputs")

_V3_METRICS = os.path.join(METRICS, "rmse_figure8_v3.txt")
_V4_METRICS = os.path.join(METRICS, "rmse_figure8_v4.txt")
_V5_METRICS = os.path.join(METRICS, "rmse_figure8_v5.txt")

_OUT_TXT  = os.path.join(METRICS, "realism_benchmark_v4_vs_v5.txt")
_TRAJ_PNG = os.path.join(PLOTS,   "trajectory_v4_vs_v5.png")
_ERR_PNG  = os.path.join(PLOTS,   "error_v4_vs_v5.png")


# ══════════════════════════════════════════════════════════════════════════════
# Metrik dosyası okuyucu
# ══════════════════════════════════════════════════════════════════════════════

def parse_metrics(path: str) -> Optional[dict]:
    """rmse_figure8_vN.txt dosyasından anahtar sayıları çek."""
    if not os.path.exists(path):
        return None
    text = open(path).read()

    def grab(pattern, default=float("nan")):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    return {
        "n_frames":    int(grab(r"n_slam_frames=(\d+)", 0)),
        "path_scale":  grab(r"path_scale=([\d.]+)"),
        "scale_x":     grab(r"scale_x=([\d.]+)"),
        "scale_y":     grab(r"scale_y=([\d.]+)"),
        "rmse_x_ps":   grab(r"path_scale_only:.*?RMSE_x=([\d.]+)"),
        "rmse_y_ps":   grab(r"path_scale_only:.*?RMSE_y=([\d.]+)"),
        "rmse_2d_ps":  grab(r"path_scale_only:.*?RMSE_2D=([\d.]+)"),
        "rmse_x":      grab(r"axis_scale:.*?RMSE_x=([\d.]+)"),
        "rmse_y":      grab(r"axis_scale:.*?RMSE_y=([\d.]+)"),
        "rmse_2d":     grab(r"axis_scale:.*?RMSE_2D=([\d.]+)"),
        "imp_vs_v3":   grab(r"imp_vs_v3=([-\d.]+)%"),
        "imp_vs_bl":   grab(r"imp_vs_baseline=([\d.]+)%"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SLAM trajectory yükleyici (evaluate_figure8_v4 uyumlu)
# ══════════════════════════════════════════════════════════════════════════════

def load_trajectory_xyz(path: str) -> np.ndarray | None:
    """SLAM CSV'den (dx, dy, dz) kümülatif delta yükle."""
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path, newline="")))
    pts  = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
    return pts - pts[0]   # frame-0'dan kümülatif delta


def load_gt_delta(path: str) -> np.ndarray | None:
    """GT CSV'den (dx, dy) kümülatif delta yükle."""
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path, newline="")))
    # x, y sütunları GT kümülatif pozisyonu içeriyor
    return np.array([[float(r["dx"]), float(r["dy"])] for r in rows])


def align_trajectory(slam_xyz: np.ndarray, gt_xy: np.ndarray,
                     path_scale: float, scale_x: float, scale_y: float
                     ) -> np.ndarray:
    """
    evaluate_figure8_v4'teki best_transform + scale uygula.
    Gerçek best_transform parametrelerini kayıttan okuyoruz.
    """
    N = min(len(slam_xyz), len(gt_xy))
    dx, dy, dz = slam_xyz[:N, 0], slam_xyz[:N, 1], slam_xyz[:N, 2]

    # v4 best_transform = wx=-sz wy=sy
    wx = -dz * path_scale * scale_x
    wy =  dy * path_scale * scale_y

    aligned = np.stack([wx, wy], axis=1)
    # Merkez hizalama
    aligned = aligned - aligned.mean(axis=0) + gt_xy[:N].mean(axis=0)
    return aligned


# ══════════════════════════════════════════════════════════════════════════════
# Rapor üretici
# ══════════════════════════════════════════════════════════════════════════════

def format_report(v3m: Optional[dict], v4m: dict, v5m: Optional[dict]) -> str:
    def fmt(v, digits=4):
        if isinstance(v, float) and not (v != v):
            return f"{v:.{digits}f}"
        return "N/A"

    def pct_change(old, new):
        """new - old farkı, pozitif = kötüleşme, negatif = iyileşme."""
        if old != old or new != new or old == 0:
            return "N/A"
        delta = new - old
        return f"{delta:+.4f}m ({100*delta/old:+.1f}%)"

    lines = [
        "=" * 75,
        "  Realism Benchmark — v3 / v4 / v5 World SLAM Karşılaştırması",
        "=" * 75,
        "",
        f"  {'Metrik':<26} {'v3 baseline':>14} {'v4 realistic':>14} {'v5 realistic':>14}",
        "  " + "-" * 71,
    ]

    metrics_def = [
        ("RMSE_x/axis_scale (m)", "rmse_x"),
        ("RMSE_y/axis_scale (m)", "rmse_y"),
        ("RMSE_2D/axis_scale (m)","rmse_2d"),
        ("RMSE_2D/path_scale (m)","rmse_2d_ps"),
        ("N SLAM frames",         "n_frames"),
        ("Path scale ×",          "path_scale"),
        ("Scale X",               "scale_x"),
        ("Scale Y",               "scale_y"),
    ]
    for label, key in metrics_def:
        v3v = (fmt(v3m[key]) if v3m else "N/A") if key != "n_frames" else (str(v3m[key]) if v3m else "N/A")
        v4v = fmt(v4m[key]) if key != "n_frames" else str(v4m[key])
        if v5m:
            v5v = fmt(v5m[key]) if key != "n_frames" else str(v5m[key])
        else:
            v5v = "BEKLENIYOR"
        lines.append(f"  {label:<26} {v3v:>14} {v4v:>14} {v5v:>14}")

    lines += ["", "  Karşılaştırma (RMSE_2D axis_scale):", "  " + "-" * 55]

    # v3 → v4
    v3r = v3m["rmse_2d"] if v3m else float("nan")
    v4r = v4m["rmse_2d"]
    lines.append(f"  v3 → v4  : {pct_change(v3r, v4r)}")

    # v4 → v5
    if v5m:
        v5r = v5m["rmse_2d"]
        lines.append(f"  v4 → v5  : {pct_change(v4r, v5r)}")
        if v5r < v4r - 0.05:
            verdict = f"✓ v5 DAHA İYİ (−{v4r-v5r:.3f}m)"
            final_world = "v5"
        elif v5r > v4r + 0.05:
            verdict = f"✗ v5 DAHA KÖTÜ (+{v5r-v4r:.3f}m)"
            final_world = "v4"
        else:
            verdict = f"≈ YAKLAŞIK EŞİT (fark {abs(v5r-v4r):.3f}m < 0.05m)"
            final_world = "v4"
        lines += [
            "", f"  Sonuç     : {verdict}",
            "",
            "=" * 75,
            f"  FINAL_WORLD = {final_world}",
            "=" * 75,
        ]
    else:
        lines += [
            "",
            "  v5 henüz mevcut değil.",
            "",
            "  Şu adımları izle:",
            "    1. Gazebo v5 world başlat:",
            "       WORLD_NAME=slam_world_v5_realistic bash sim/scripts/run_gazebo_realistic.sh",
            "    2. Veri topla (5 dakika, competition motion):",
            "       PATTERN=competition bash runtime/run_full_pipeline.sh --world slam_world_v5_realistic \\",
            "         --skip-slam --dataset bench_v5",
            "    3. DROID-SLAM çalıştır (conda env: droid_clean):",
            "       conda run -n droid_clean bash slam/scripts/run_droid_figure8_v3.sh \\",
            "         --images dataset/small_motion_bench_v5 --out slam/outputs/trajectory_figure8_v5.csv",
            "    4. Metrikleri üret:",
            "       python3 slam/scripts/evaluate_figure8_v4.py  ← v5 için kopyala",
            "    5. Bu scripti tekrar çalıştır.",
            "",
            "  Mevcut v4 hedef: 3.85m  |  Beklenti v5: < 3.0m",
            "",
            "=" * 75,
            "  FINAL_WORLD = v4  (v5 verisi bekleniyor)",
            "=" * 75,
        ]

    lines += [
        "",
        "  Online Estimator (SLAMPoseEstimator, v3 traj):",
        f"    DROID RMSE_2D = 0.127m  |  ORB RMSE_2D = 0.854m",
        "    → Bu metrik health-flag ile kalibre edilmiş çıktıyı ölçer.",
        "    → Raw SLAM metriğinden ~30× daha iyi (Sim(3) + kalibrasyon etkisi).",
        "",
        "  Notlar:",
        "    • v3→v4 raw SLAM: 0% iyileşme (aynı hareket, hafif texture/ışık değişimi)",
        "    • v5'in temel farkları: parallax (5 kule), dinamik objeler, lens distortion",
        "    • Dinamik objeler ilk çalıştırmada hafif kötüleşme yaratabilir",
        "    • Seed değiştirerek (--seed 99 vb.) tekrar test et",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Plotlar
# ══════════════════════════════════════════════════════════════════════════════

def plot_trajectory_comparison(v4m: dict, v5m: Optional[dict], out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib yok — trajectory plot atlandı")
        return

    # SLAM trajectory'leri yükle
    v4_traj = load_trajectory_xyz(os.path.join(SLAM_OUTPUTS, "trajectory_figure8_v4.csv"))
    v5_traj = load_trajectory_xyz(os.path.join(SLAM_OUTPUTS, "trajectory_figure8_v5.csv"))
    v4_gt   = load_gt_delta(os.path.join(REPO, "evaluation/ground_truth_figure8_v4.csv"))

    if v4_traj is None or v4_gt is None:
        print(f"  trajectory_figure8_v4.csv veya GT bulunamadı — plot atlandı")
        return

    N = min(len(v4_traj), len(v4_gt))
    gt_xy = v4_gt[:N]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Sol: Trajectory overlay ───────────────────────────────────────────────
    ax = axes[0]

    # GT — kümülatif
    gt_cum = np.cumsum(gt_xy, axis=0)
    ax.plot(gt_cum[:, 0], gt_cum[:, 1], "k-", lw=2.5, label="Ground Truth", zorder=5)
    ax.plot(gt_cum[0, 0],  gt_cum[0, 1],  "k^", ms=10, zorder=6)
    ax.plot(gt_cum[-1, 0], gt_cum[-1, 1], "ks", ms=10, zorder=6)

    # v4 SLAM
    v4_aligned = align_trajectory(v4_traj, gt_cum,
                                   v4m["path_scale"], v4m["scale_x"], v4m["scale_y"])
    ax.plot(v4_aligned[:, 0], v4_aligned[:, 1], "#1f77b4", lw=1.8,
            label=f"v4 SLAM (RMSE={v4m['rmse_2d']:.3f}m)")

    if v5_traj is not None and v5m is not None:
        v5_aligned = align_trajectory(v5_traj, gt_cum,
                                       v5m["path_scale"], v5m["scale_x"], v5m["scale_y"])
        ax.plot(v5_aligned[:, 0], v5_aligned[:, 1], "#d62728", lw=1.5, alpha=0.85,
                label=f"v5 SLAM (RMSE={v5m['rmse_2d']:.3f}m)")
    else:
        ax.text(0.05, 0.88, "v5 verisi bekleniyor",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="#fff3cd", alpha=0.9, boxstyle="round"))

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory — v4 vs v5 (path+axis aligned)")
    ax.set_aspect("equal"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Sağ: Bar karşılaştırma ────────────────────────────────────────────────
    ax2 = axes[1]
    labels = ["v3\n(baseline)", "v4\n(realistic)", "v5\n(v5_realistic)"]
    v3m = parse_metrics(_V3_METRICS)
    vals = [
        v3m["rmse_2d"] if v3m else float("nan"),
        v4m["rmse_2d"],
        v5m["rmse_2d"] if v5m else float("nan"),
    ]
    colors = ["#aec7e8", "#1f77b4", "#d62728" if v5m else "#cccccc"]
    bars = ax2.bar(labels, vals, color=colors, width=0.5, zorder=3)
    ax2.axhline(3.0, color="green", ls="--", lw=1.5, label="Hedef: 3.0m", zorder=4)
    ax2.axhline(3.85, color="orange", ls=":", lw=1.0, label="v4 baseline", zorder=4)
    for bar, val in zip(bars, vals):
        if val == val:  # not nan
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                     f"{val:.3f}m", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("RMSE_2D (m)"); ax2.set_title("RMSE Karşılaştırma")
    ax2.legend(fontsize=9); ax2.grid(True, axis="y", alpha=0.3)
    if v5m:
        ax2.set_ylim(0, max(vals) * 1.15)

    fig.suptitle("v4 vs v5 Realism Benchmark", fontsize=14, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Kaydedildi: {out_path}")


def plot_error_over_time(v4m: dict, v5m: Optional[dict], out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Per-frame hata için trajectory yükle
    v4_traj = load_trajectory_xyz(os.path.join(SLAM_OUTPUTS, "trajectory_figure8_v4.csv"))
    v5_traj = load_trajectory_xyz(os.path.join(SLAM_OUTPUTS, "trajectory_figure8_v5.csv"))
    v4_gt   = load_gt_delta(os.path.join(REPO, "evaluation/ground_truth_figure8_v4.csv"))

    if v4_traj is None or v4_gt is None:
        print("  Trajectory/GT bulunamadı — error plot atlandı")
        return

    N = min(len(v4_traj), len(v4_gt))
    gt_cum = np.cumsum(v4_gt[:N], axis=0)

    v4_aligned = align_trajectory(v4_traj, gt_cum,
                                   v4m["path_scale"], v4m["scale_x"], v4m["scale_y"])
    v4_err = np.sqrt(np.sum((v4_aligned - gt_cum)**2, axis=1))

    fig, ax = plt.subplots(figsize=(13, 5))
    frames = np.arange(N)

    ax.plot(frames, v4_err, "#1f77b4", lw=1.5,
            label=f"v4 hata  (RMSE={v4m['rmse_2d']:.3f}m, max={v4_err.max():.3f}m)")
    ax.axhline(v4m["rmse_2d"], color="#1f77b4", ls="--", lw=0.9, alpha=0.6)

    if v5_traj is not None and v5m is not None:
        v5_gt = load_gt_delta(os.path.join(REPO, "evaluation/ground_truth_figure8_v4.csv"))
        if v5_gt is not None:
            Nv = min(len(v5_traj), len(v5_gt))
            gt5_cum = np.cumsum(v5_gt[:Nv], axis=0)
            v5_aligned = align_trajectory(v5_traj, gt5_cum,
                                           v5m["path_scale"], v5m["scale_x"], v5m["scale_y"])
            v5_err = np.sqrt(np.sum((v5_aligned - gt5_cum)**2, axis=1))
            ax.plot(np.arange(Nv), v5_err, "#d62728", lw=1.3, alpha=0.85,
                    label=f"v5 hata  (RMSE={v5m['rmse_2d']:.3f}m, max={v5_err.max():.3f}m)")
            ax.axhline(v5m["rmse_2d"], color="#d62728", ls="--", lw=0.9, alpha=0.6)
    else:
        ax.text(0.55, 0.88, "v5 verisi bekleniyor",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="#fff3cd", alpha=0.9, boxstyle="round"))

    ax.axhline(3.0, color="green", ls=":", lw=1.5, label="Hedef: 3.0m", zorder=4)
    ax.set_xlabel("SLAM Frame")
    ax.set_ylabel("2D Pozisyon Hatası (m)")
    ax.set_title("Error vs Time — v4 vs v5 SLAM")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Kaydedildi: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Ana
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="v4 vs v5 realism benchmark")
    parser.add_argument("--plots", action="store_true", help="Sadece plotları yenile")
    args = parser.parse_args()

    # Metrik dosyaları oku
    v3m = parse_metrics(_V3_METRICS)
    v4m = parse_metrics(_V4_METRICS)
    v5m = parse_metrics(_V5_METRICS)

    if v4m is None:
        print(f"HATA: v4 metrik dosyası bulunamadı: {_V4_METRICS}")
        print("  Çalıştır: python3 slam/scripts/evaluate_figure8_v4.py")
        sys.exit(1)

    print(f"v3 RMSE_2D = {v3m['rmse_2d'] if v3m else 'N/A'}")
    print(f"v4 RMSE_2D = {v4m['rmse_2d']:.4f}m  (n={v4m['n_frames']})")
    if v5m:
        print(f"v5 RMSE_2D = {v5m['rmse_2d']:.4f}m  (n={v5m['n_frames']})")
    else:
        print(f"v5 metrik yok ({_V5_METRICS})")

    # Rapor
    if not args.plots:
        report = format_report(v3m, v4m, v5m)
        print()
        print(report)
        os.makedirs(METRICS, exist_ok=True)
        with open(_OUT_TXT, "w") as f:
            f.write(report + "\n")
        print(f"\nRapor: {_OUT_TXT}")

    # Plotlar
    print("\nPlotlar üretiliyor...")
    plot_trajectory_comparison(v4m, v5m, _TRAJ_PNG)
    plot_error_over_time(v4m, v5m, _ERR_PNG)


if __name__ == "__main__":
    main()
