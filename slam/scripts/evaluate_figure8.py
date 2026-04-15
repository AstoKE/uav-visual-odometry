#!/usr/bin/env python3
"""
evaluate_figure8.py — Figure-8 SLAM sonucunu tam pipeline ile değerlendirir.

Adımlar:
  1. trajectory_figure8.csv → delta → delta_trajectory_figure8.csv
  2. Ground truth üret        → evaluation/ground_truth_figure8.csv
  3. Scale hesapla (path-length ratio)
  4. Koordinat dönüşümü test et (10 aday, best seçilir)
  5. Axis-wise scale uygula
  6. RMSE hesapla (Procrustes hizalaması ile)
  7. Karşılaştırma tablosu + trajectory + scale dağılım plotları

Kullanım:
    python3 ~/code/uav-visual-odometry/slam/scripts/evaluate_figure8.py

Ortam değişkenleri:
    N_SLAM_FRAMES : SLAM frame sayısı (varsayılan: trajectory satır sayısından otomatik)
    STRIDE        : kayıt stride'ı    (varsayılan: 3)
"""

import csv, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO    = os.path.expanduser("~/code/uav-visual-odometry")
STRIDE  = int(os.environ.get("STRIDE", "3"))

TRAJ_IN   = os.path.join(REPO, "slam/outputs/trajectory_figure8.csv")
DELTA_OUT = os.path.join(REPO, "slam/outputs/delta_trajectory_figure8.csv")
GT_OUT    = os.path.join(REPO, "evaluation/ground_truth_figure8.csv")
WORLD_OUT = os.path.join(REPO, "slam/outputs/delta_figure8_world.csv")
PLOTS_DIR = os.path.join(REPO, "evaluation/plots")
METRICS   = os.path.join(REPO, "evaluation/metrics/rmse_figure8.txt")

# Lawnmower karşılaştırma referansları (önceki run)
PREV_RMSE = {"baseline": 12.983, "world_v1": 7.850, "axis_scale": 5.691}


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def load_csv(path, *cols):
    rows = list(csv.DictReader(open(path, newline="")))
    return np.array([[float(r[c]) for c in cols] for r in rows])


def path_length_2d(arr):
    """arr: [N,2] — ardışık noktalar arası 2D mesafelerin toplamı."""
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def procrustes_rmse(src2d, tgt2d):
    H = src2d.T @ tgt2d
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, d]) @ U.T
    angle = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    aligned = src2d @ R.T
    err = tgt2d - aligned
    rmse_x  = float(np.sqrt(np.mean(err[:, 0]**2)))
    rmse_y  = float(np.sqrt(np.mean(err[:, 1]**2)))
    rmse_2d = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
    return rmse_x, rmse_y, rmse_2d, angle, aligned


# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Trajectory yükle ──────────────────────────────────────────────────
    if not os.path.isfile(TRAJ_IN):
        print(f"HATA: trajectory bulunamadı: {TRAJ_IN}")
        print("  Önce çalıştır: bash slam/scripts/run_droid_figure8.sh")
        sys.exit(1)

    traj = load_csv(TRAJ_IN, "x", "y", "z")
    n_slam = len(traj)
    print(f"[eval] SLAM frame sayısı  : {n_slam}")

    # ── 2. Delta hesapla ─────────────────────────────────────────────────────
    x0, y0, z0 = traj[0]
    delta = traj - np.array([x0, y0, z0])   # [N,3]
    os.makedirs(os.path.dirname(DELTA_OUT), exist_ok=True)
    with open(DELTA_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "dx", "dy", "dz"])
        for i, (dx, dy, dz) in enumerate(delta):
            w.writerow([i, f"{dx:.8f}", f"{dy:.8f}", f"{dz:.8f}"])
    print(f"[eval] Delta      : {DELTA_OUT}")

    # ── 3. Ground truth üret ─────────────────────────────────────────────────
    os.environ["PATTERN"]       = "figure8"
    os.environ["N_SLAM_FRAMES"] = str(n_slam)
    os.environ["STRIDE"]        = str(STRIDE)
    os.environ["GT_OUT"]        = GT_OUT
    sys.path.insert(0, os.path.join(REPO, "sim/scripts"))
    import importlib
    import export_ground_truth as egt
    importlib.reload(egt)
    egt.main()

    gt = load_csv(GT_OUT, "dx", "dy")   # [N,2] metres

    n = min(n_slam, len(gt))
    delta = delta[:n];  gt = gt[:n]
    print(f"[eval] Kullanılan frame    : {n}")

    # ── 4. Scale hesapla (path-length ratio) ─────────────────────────────────
    slam_path = path_length_2d(delta[:, :2])
    gt_path   = path_length_2d(gt)
    path_scale = gt_path / slam_path if slam_path > 1e-9 else 1.0
    print(f"[eval] SLAM 2D yol         : {slam_path:.6f}")
    print(f"[eval] GT   2D yol         : {gt_path:.4f} m")
    print(f"[eval] Path-length scale   : {path_scale:.4f}×")

    # ── 5. Koordinat dönüşümü (10 aday) ──────────────────────────────────────
    candidates = [
        ("wx=sz  wy=sy",   lambda dx,dy,dz: (dz, dy)),
        ("wx=-sz wy=sy",   lambda dx,dy,dz: (-dz, dy)),
        ("wx=sz  wy=-sy",  lambda dx,dy,dz: (dz, -dy)),
        ("wx=-sz wy=-sy",  lambda dx,dy,dz: (-dz, -dy)),
        ("wx=sx  wy=sy",   lambda dx,dy,dz: (dx, dy)),
        ("wx=-sx wy=sy",   lambda dx,dy,dz: (-dx, dy)),
        ("wx=sx  wy=-sy",  lambda dx,dy,dz: (dx, -dy)),
        ("wx=sx  wy=sz",   lambda dx,dy,dz: (dx, dz)),
        ("wx=sy  wy=sx",   lambda dx,dy,dz: (dy, dx)),
        ("wx=sz  wy=sx",   lambda dx,dy,dz: (dz, dx)),
    ]
    best_r2d = float("inf");  best_fn = None;  best_name = ""
    for name, fn in candidates:
        slam_xy = np.array([fn(dx*path_scale, dy*path_scale, dz*path_scale)
                            for dx, dy, dz in delta])
        rx, ry, r2d, ang, _ = procrustes_rmse(slam_xy, gt)
        if r2d < best_r2d:
            best_r2d = r2d;  best_fn = fn;  best_name = name

    print(f"[eval] En iyi dönüşüm      : {best_name}  RMSE_2D={best_r2d:.4f}m")

    # World-frame koordinatlar (path scale ile)
    slam_world = np.array([best_fn(dx*path_scale, dy*path_scale, dz*path_scale)
                           for dx, dy, dz in delta])

    # ── 6. Axis-wise scale ────────────────────────────────────────────────────
    THRESH = 1e-4
    sx_vals, sy_vals = [], []
    for i in range(n):
        if abs(slam_world[i, 0]) > THRESH:
            sx_vals.append(gt[i, 0] / slam_world[i, 0])
        if abs(slam_world[i, 1]) > THRESH:
            sy_vals.append(gt[i, 1] / slam_world[i, 1])

    scale_x = float(np.median(sx_vals)) if sx_vals else 1.0
    scale_y = float(np.median(sy_vals)) if sy_vals else 1.0
    print(f"[eval] Axis-wise scale_x   : {scale_x:.4f}  (n={len(sx_vals)}, std={np.std(sx_vals):.2f})")
    print(f"[eval] Axis-wise scale_y   : {scale_y:.4f}  (n={len(sy_vals)}, std={np.std(sy_vals):.2f})")

    slam_axis = slam_world.copy()
    slam_axis[:, 0] *= scale_x
    slam_axis[:, 1] *= scale_y

    # RMSE hesapla (sadece Procrustes, axis-wise scale sonrası)
    rx_pw, ry_pw, r2d_pw, ang_pw, aligned_pw = procrustes_rmse(slam_world, gt)
    rx_ax, ry_ax, r2d_ax, ang_ax, aligned_ax = procrustes_rmse(slam_axis,  gt)

    # ── 7. Çıktı CSV ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(WORLD_OUT), exist_ok=True)
    with open(WORLD_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y"])
        for i, (x, y) in enumerate(slam_axis):
            w.writerow([i, f"{x:.6f}", f"{y:.6f}"])

    # ── 8. Karşılaştırma tablosu ──────────────────────────────────────────────
    rows = [
        ("Lawnmower baseline",          PREV_RMSE["baseline"],  "—",            "—"),
        ("Lawnmower world+path scale",   PREV_RMSE["world_v1"], "—",            "—"),
        ("Lawnmower axis scale",         PREV_RMSE["axis_scale"],"—",           "—"),
        ("Figure-8 path scale only",     r2d_pw,  f"rx={rx_pw:.3f}", f"ry={ry_pw:.3f}"),
        ("Figure-8 axis-wise scale ◄",   r2d_ax,  f"rx={rx_ax:.3f}", f"ry={ry_ax:.3f}"),
    ]

    imp_vs_lawn = (PREV_RMSE["axis_scale"] - r2d_ax) / PREV_RMSE["axis_scale"] * 100
    imp_vs_base = (PREV_RMSE["baseline"]   - r2d_ax) / PREV_RMSE["baseline"]   * 100

    print()
    print("=" * 70)
    print("KARŞILAŞTIRMA TABLOSU")
    print("=" * 70)
    print(f"{'Yöntem':<40} {'RMSE_2D':>9}  {'RMSE_x':>10}  {'RMSE_y':>10}")
    print("-" * 70)
    for name, r2d, info_x, info_y in rows:
        r2d_str = f"{r2d:.3f} m" if isinstance(r2d, float) else str(r2d)
        print(f"{name:<40} {r2d_str:>9}  {info_x:>10}  {info_y:>10}")
    print("=" * 70)
    print(f"\nFigure-8 axis scale vs Lawnmower axis scale : {imp_vs_lawn:+.1f}%")
    print(f"Figure-8 axis scale vs Baseline             : {imp_vs_base:+.1f}%")

    # ── 9. Rapor ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(METRICS), exist_ok=True)
    with open(METRICS, "w") as f:
        f.write(f"n_slam_frames={n}\n")
        f.write(f"path_scale={path_scale:.4f}\n")
        f.write(f"best_transform={best_name}\n")
        f.write(f"scale_x={scale_x:.4f}\nscale_y={scale_y:.4f}\n")
        f.write(f"path_scale_only: RMSE_x={rx_pw:.4f} RMSE_y={ry_pw:.4f} RMSE_2D={r2d_pw:.4f}\n")
        f.write(f"axis_scale:      RMSE_x={rx_ax:.4f} RMSE_y={ry_ax:.4f} RMSE_2D={r2d_ax:.4f}\n")
        f.write(f"imp_vs_lawnmower={imp_vs_lawn:.1f}%\nimp_vs_baseline={imp_vs_base:.1f}%\n")
    print(f"\n[eval] Rapor: {METRICS}")

    # ── 10. Plotlar ───────────────────────────────────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --- Trajectory XY ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, slam_xy, title_suffix in [
        (axes[0], aligned_pw, f"Path scale ({path_scale:.1f}×)\nRMSE_2D={r2d_pw:.3f}m"),
        (axes[1], aligned_ax, f"Axis scale (sx={scale_x:.1f}×, sy={scale_y:.1f}×)\nRMSE_2D={r2d_ax:.3f}m"),
    ]:
        ax.plot(gt[:, 0],      gt[:, 1],      "b-o",  ms=2, lw=1.5, label="Ground Truth (figure-8)")
        ax.plot(slam_xy[:, 0], slam_xy[:, 1], "r--s", ms=2, lw=1.5, label="SLAM aligned")
        ax.scatter(*gt[0],         color="blue", s=60, zorder=5, marker="^")
        ax.scatter(*gt[-1],        color="blue", s=60, zorder=5, marker="v")
        ax.scatter(*slam_xy[0],    color="red",  s=60, zorder=5, marker="^")
        ax.scatter(*slam_xy[-1],   color="red",  s=60, zorder=5, marker="v")
        ax.set_xlabel("dx (m)"); ax.set_ylabel("dy (m)")
        ax.set_title(f"Figure-8 Trajectory — {title_suffix}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_aspect("equal")

    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, "trajectory_figure8.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"[eval] Trajectory plot: {p}")

    # --- GT figure-8 shape ---
    fig, ax = plt.subplots(figsize=(8, 6))
    gt_abs = load_csv(GT_OUT, "x", "y")
    ax.plot(gt_abs[:, 0], gt_abs[:, 1], "b-", lw=1.5)
    ax.scatter(*gt_abs[0], color="green", s=80, zorder=5, marker="^", label="start")
    ax.scatter(*gt_abs[-1], color="red",  s=80, zorder=5, marker="v", label="end")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(f"GT Figure-8 Trajectory  ({n} SLAM frames)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    fig.tight_layout()
    p2 = os.path.join(PLOTS_DIR, "gt_figure8_shape.png")
    fig.savefig(p2, dpi=120); plt.close(fig)
    print(f"[eval] GT shape plot  : {p2}")

    # --- Scale dağılımı (axis-wise) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, vals, med, label, color in [
        (axes[0], sx_vals, scale_x, f"scale_x  median={scale_x:.2f}", "steelblue"),
        (axes[1], sy_vals, scale_y, f"scale_y  median={scale_y:.2f}", "darkorange"),
    ]:
        clipped = np.clip(vals, -300, 300)
        ax.scatter(range(len(clipped)), clipped, s=10, alpha=0.6, color=color)
        ax.axhline(med, color="red", lw=1.5, linestyle="--", label=f"median={med:.2f}")
        ax.set_title(label); ax.set_xlabel("sample"); ax.set_ylabel("gt/slam")
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle("Figure-8 Per-Frame Axis Scale Distribution")
    fig.tight_layout()
    p3 = os.path.join(PLOTS_DIR, "scale_figure8_axis.png")
    fig.savefig(p3, dpi=120); plt.close(fig)
    print(f"[eval] Scale dist plot: {p3}")

    # --- Karşılaştırma bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = ["Lawnmower\nbaseline", "Lawnmower\nworld+path", "Lawnmower\naxis scale",
               "Figure-8\npath scale", "Figure-8\naxis scale"]
    rmses   = [PREV_RMSE["baseline"], PREV_RMSE["world_v1"], PREV_RMSE["axis_scale"],
               r2d_pw, r2d_ax]
    colors  = ["#d62728", "#ff7f0e", "#9467bd", "#1f77b4", "#2ca02c"]
    bars = ax.bar(methods, rmses, color=colors, alpha=0.8, edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}m", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("RMSE_2D (m)")
    ax.set_title("RMSE_2D Karşılaştırması: Lawnmower vs Figure-8")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p4 = os.path.join(PLOTS_DIR, "rmse_comparison_figure8.png")
    fig.savefig(p4, dpi=120); plt.close(fig)
    print(f"[eval] Karşılaştırma  : {p4}")

    print()
    print("=" * 55)
    print("ÖZET")
    print("=" * 55)
    print(f"  Figure-8 path scale   RMSE_2D = {r2d_pw:.4f} m")
    print(f"  Figure-8 axis scale   RMSE_2D = {r2d_ax:.4f} m")
    print(f"  Lawnmower axis scale  RMSE_2D = {PREV_RMSE['axis_scale']:.4f} m")
    print(f"  İyileşme (figure8 vs lawnmower): {imp_vs_lawn:+.1f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
