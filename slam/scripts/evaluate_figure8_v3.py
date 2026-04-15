#!/usr/bin/env python3
"""
evaluate_figure8_v3.py — Figure-8 v3 SLAM sonucunu tam pipeline ile değerlendirir.

v3 iyileştirmeleri (v2'ye göre):
  - Kamera yüksekliği 10m → 4m  (parallax ~2.5× artar)
  - Zengin Gazebo sahnesi: 77 obje + renkli zemin döşemeleri
  - Kalibrasyon: 452 → 457 (fx=fy)
  - Figure-8: Ax=4.5m, Ay=3.5m, T=25s, 3.5 tur

Pipeline:
  1. trajectory_figure8_v3.csv → delta
  2. GT üret (figure-8, 4m yükseklik)
  3. Path-length scale hesapla
  4. En iyi koordinat dönüşümünü bul (10 aday)
  5. Axis-wise scale uygula
  6. RMSE hesapla (Procrustes ile)
  7. Karşılaştırma tablosu + plotlar

Kullanım:
    python3 ~/code/uav-visual-odometry/slam/scripts/evaluate_figure8_v3.py

Ortam değişkenleri:
    STRIDE  : kayıt stride'ı (varsayılan: 3)
"""

import csv, importlib, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO   = os.path.expanduser("~/code/uav-visual-odometry")
STRIDE = int(os.environ.get("STRIDE", "3"))

TRAJ_IN   = os.path.join(REPO, "slam/outputs/trajectory_figure8_v3.csv")
DELTA_OUT = os.path.join(REPO, "slam/outputs/delta_trajectory_figure8_v3.csv")
GT_OUT    = os.path.join(REPO, "evaluation/ground_truth_figure8_v3.csv")
WORLD_OUT = os.path.join(REPO, "slam/outputs/delta_figure8_v3_world.csv")
PLOTS_DIR = os.path.join(REPO, "evaluation/plots")
METRICS   = os.path.join(REPO, "evaluation/metrics/rmse_figure8_v3.txt")

# Önceki referans sonuçlar
PREV = {
    "Lawnmower baseline":          12.983,
    "Lawnmower world+path scale":   7.850,
    "Lawnmower axis scale":         5.691,
    "Figure-8 v2 axis scale":       5.520,  # yaklaşık değer
}


# ── Yardımcılar ───────────────────────────────────────────────────────────────

def load_csv(path, *cols):
    rows = list(csv.DictReader(open(path, newline="")))
    return np.array([[float(r[c]) for c in cols] for r in rows])

def path_length_2d(arr):
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))

def procrustes_rmse(src2d, tgt2d):
    H = src2d.T @ tgt2d
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, d]) @ U.T
    angle = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    aligned = src2d @ R.T
    err = tgt2d - aligned
    rx  = float(np.sqrt(np.mean(err[:, 0]**2)))
    ry  = float(np.sqrt(np.mean(err[:, 1]**2)))
    r2d = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
    return rx, ry, r2d, angle, aligned


# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Trajectory yükle ──────────────────────────────────────────────────
    if not os.path.isfile(TRAJ_IN):
        print(f"HATA: trajectory bulunamadı: {TRAJ_IN}")
        print("  Önce çalıştır: bash slam/scripts/run_droid_figure8_v3.sh")
        sys.exit(1)

    traj = load_csv(TRAJ_IN, "x", "y", "z")
    n_slam = len(traj)
    print(f"[eval_v3] SLAM frame sayısı : {n_slam}")

    # ── 2. Delta hesapla ─────────────────────────────────────────────────────
    x0, y0, z0 = traj[0]
    delta = traj - np.array([x0, y0, z0])
    os.makedirs(os.path.dirname(DELTA_OUT), exist_ok=True)
    with open(DELTA_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "dx", "dy", "dz"])
        for i, (dx, dy, dz) in enumerate(delta):
            w.writerow([i, f"{dx:.8f}", f"{dy:.8f}", f"{dz:.8f}"])

    # ── 3. Ground truth üret ─────────────────────────────────────────────────
    os.environ.update({
        "PATTERN": "figure8",
        "N_SLAM_FRAMES": str(n_slam),
        "STRIDE": str(STRIDE),
        "GT_OUT": GT_OUT,
        # v3 parametreleri (move_camera.py v3 ile eşleşmeli)
        "FIGURE8_AX":     "4.5",
        "FIGURE8_AY":     "3.5",
        "FIGURE8_PERIOD": "25.0",
    })
    sys.path.insert(0, os.path.join(REPO, "sim/scripts"))
    import export_ground_truth as egt
    importlib.reload(egt)
    egt.main()

    gt = load_csv(GT_OUT, "dx", "dy")
    n  = min(n_slam, len(gt))
    delta = delta[:n];  gt = gt[:n]
    print(f"[eval_v3] Kullanılan frame  : {n}")

    # ── 4. Scale (path-length) ────────────────────────────────────────────────
    slam_path  = path_length_2d(delta[:, :2])
    gt_path    = path_length_2d(gt)
    path_scale = gt_path / slam_path if slam_path > 1e-9 else 1.0
    print(f"[eval_v3] SLAM 2D yol       : {slam_path:.6f}")
    print(f"[eval_v3] GT   2D yol       : {gt_path:.4f} m")
    print(f"[eval_v3] Path-length scale : {path_scale:.4f}×")

    # ── 5. En iyi koordinat dönüşümü ─────────────────────────────────────────
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
        _, _, r2d, _, _ = procrustes_rmse(slam_xy, gt)
        if r2d < best_r2d:
            best_r2d = r2d;  best_fn = fn;  best_name = name

    slam_world = np.array([best_fn(dx*path_scale, dy*path_scale, dz*path_scale)
                           for dx, dy, dz in delta])
    print(f"[eval_v3] En iyi dönüşüm   : {best_name}  RMSE_2D={best_r2d:.4f}m")

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
    print(f"[eval_v3] scale_x           : {scale_x:.4f}  (n={len(sx_vals)}, std={np.std(sx_vals):.2f})")
    print(f"[eval_v3] scale_y           : {scale_y:.4f}  (n={len(sy_vals)}, std={np.std(sy_vals):.2f})")

    slam_axis = slam_world.copy()
    slam_axis[:, 0] *= scale_x
    slam_axis[:, 1] *= scale_y

    # ── 7. RMSE ───────────────────────────────────────────────────────────────
    rx_pw, ry_pw, r2d_pw, ang_pw, aligned_pw = procrustes_rmse(slam_world, gt)
    rx_ax, ry_ax, r2d_ax, ang_ax, aligned_ax = procrustes_rmse(slam_axis,  gt)

    # ── 8. Çıktı CSV ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(WORLD_OUT), exist_ok=True)
    with open(WORLD_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y"])
        for i, (x, y) in enumerate(slam_axis):
            w.writerow([i, f"{x:.6f}", f"{y:.6f}"])

    # ── 9. Karşılaştırma tablosu ──────────────────────────────────────────────
    v2_axis = PREV["Figure-8 v2 axis scale"]
    imp_vs_v2   = (v2_axis  - r2d_ax) / v2_axis  * 100
    imp_vs_base = (PREV["Lawnmower baseline"] - r2d_ax) / PREV["Lawnmower baseline"] * 100

    all_rows = list(PREV.items()) + [
        ("Figure-8 v3 path scale",   r2d_pw),
        ("Figure-8 v3 axis scale ◄", r2d_ax),
    ]

    print()
    print("=" * 65)
    print("KARŞILAŞTIRMA TABLOSU")
    print("=" * 65)
    print(f"{'Yöntem':<40} {'RMSE_2D':>9}")
    print("-" * 55)
    for name, val in all_rows:
        marker = " ◄" if "◄" in name else ""
        val_str = f"{val:.3f} m"
        print(f"{name:<40} {val_str:>9}")
    print("=" * 65)
    print(f"\nFigure-8 v3 axis scale  vs  v2 axis scale : {imp_vs_v2:+.1f}%")
    print(f"Figure-8 v3 axis scale  vs  baseline      : {imp_vs_base:+.1f}%")
    print(f"RMSE_x = {rx_ax:.4f} m")
    print(f"RMSE_y = {ry_ax:.4f} m")

    # ── 10. Rapor ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(METRICS), exist_ok=True)
    with open(METRICS, "w") as f:
        f.write(f"n_slam_frames={n}\n")
        f.write(f"path_scale={path_scale:.4f}\n")
        f.write(f"best_transform={best_name}\n")
        f.write(f"scale_x={scale_x:.4f}\nscale_y={scale_y:.4f}\n\n")
        f.write(f"path_scale_only: RMSE_x={rx_pw:.4f} RMSE_y={ry_pw:.4f} RMSE_2D={r2d_pw:.4f}\n")
        f.write(f"axis_scale:      RMSE_x={rx_ax:.4f} RMSE_y={ry_ax:.4f} RMSE_2D={r2d_ax:.4f}\n\n")
        f.write(f"imp_vs_v2_axis={imp_vs_v2:.1f}%\n")
        f.write(f"imp_vs_baseline={imp_vs_base:.1f}%\n")
    print(f"\n[eval_v3] Rapor: {METRICS}")

    # ── 11. Plotlar ───────────────────────────────────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --- Trajectory XY ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, slam_xy, title_sfx in [
        (axes[0], aligned_pw, f"Path scale ({path_scale:.1f}×)\nRMSE_2D={r2d_pw:.3f}m"),
        (axes[1], aligned_ax, f"Axis scale  sx={scale_x:.1f}×  sy={scale_y:.1f}×\nRMSE_2D={r2d_ax:.3f}m"),
    ]:
        ax.plot(gt[:, 0],      gt[:, 1],      "b-o",  ms=2, lw=1.5, label="GT (figure-8)")
        ax.plot(slam_xy[:, 0], slam_xy[:, 1], "r--s", ms=2, lw=1.5, label="SLAM aligned")
        ax.scatter(*gt[0],         color="blue", s=60, zorder=5, marker="^")
        ax.scatter(*gt[-1],        color="blue", s=60, zorder=5, marker="v")
        ax.scatter(*slam_xy[0],    color="red",  s=60, zorder=5, marker="^")
        ax.scatter(*slam_xy[-1],   color="red",  s=60, zorder=5, marker="v")
        ax.set_xlabel("dx (m)"); ax.set_ylabel("dy (m)")
        ax.set_title(f"Figure-8 v3 — {title_sfx}\n(4m kamera, zengin sahne)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, "trajectory_figure8_v3.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"[eval_v3] Trajectory plot : {p}")

    # --- Karşılaştırma bar chart ---
    fig, ax = plt.subplots(figsize=(12, 5))
    names  = [n for n,_ in all_rows]
    rmses  = [v for _,v in all_rows]
    colors = ["#d62728","#ff7f0e","#9467bd","#aec7e8","#1f77b4","#2ca02c"]
    bars = ax.bar(names, rmses, color=colors[:len(names)], alpha=0.85,
                  edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}m", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("RMSE_2D (m)")
    ax.set_title("RMSE_2D Karşılaştırması — v3 (4m, zengin sahne) vs önceki")
    ax.tick_params(axis="x", labelrotation=15, labelsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p2 = os.path.join(PLOTS_DIR, "rmse_comparison_v3.png")
    fig.savefig(p2, dpi=120); plt.close(fig)
    print(f"[eval_v3] Karşılaştırma  : {p2}")

    # --- GT figure-8 v3 şekli ---
    gt_abs = load_csv(GT_OUT, "x", "y")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(gt_abs[:, 0], gt_abs[:, 1], "b-", lw=1.5)
    ax.scatter(*gt_abs[0],  color="green", s=80, zorder=5, marker="^", label="start")
    ax.scatter(*gt_abs[-1], color="red",   s=80, zorder=5, marker="v", label="end")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(f"GT Figure-8 v3  (Ax=4.5m, Ay=3.5m, T=25s × 3.5 tur)\n{n} SLAM frames")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    fig.tight_layout()
    p3 = os.path.join(PLOTS_DIR, "gt_figure8_v3_shape.png")
    fig.savefig(p3, dpi=120); plt.close(fig)
    print(f"[eval_v3] GT shape plot  : {p3}")

    print()
    print("=" * 55)
    print("ÖZET (v3)")
    print("=" * 55)
    print(f"  SLAM frame            : {n}")
    print(f"  Kamera yüksekliği     : 4 m  (eskisi: 10 m)")
    print(f"  Path-length scale     : {path_scale:.1f}×")
    print(f"  scale_x / scale_y     : {scale_x:.2f}×  /  {scale_y:.2f}×")
    print(f"  RMSE_x                : {rx_ax:.4f} m")
    print(f"  RMSE_y                : {ry_ax:.4f} m")
    print(f"  RMSE_2D               : {r2d_ax:.4f} m")
    print(f"  İyileşme vs v2        : {imp_vs_v2:+.1f}%")
    print(f"  İyileşme vs baseline  : {imp_vs_base:+.1f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
