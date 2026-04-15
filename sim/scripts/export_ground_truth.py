#!/usr/bin/env python3
"""
export_ground_truth.py — move_camera.py hareket modelinden ground truth üretir.

İki desen desteklenir (PATTERN env var ile):
  figure8   (varsayılan) — figure-8 (∞) deseni
  lawnmower             — orijinal biçerdöver deseni

Timing modeli (varsayım: recorder ve move_camera aynı anda başladı):
  - recording_frame = slam_frame * STRIDE
  - t = recording_frame / FPS
  - t < INITIAL_SLEEP: kamera başlangıç noktasında durur
  - t >= INITIAL_SLEEP: hareket başlar

Figure-8 için başlangıç noktası: (0, 0) — parametrik eğri t=0'da (0,0)'dadır.
Lawnmower için başlangıç noktası: (-8, -8) — grid'in sol-alt köşesi.

Çıktı: evaluation/ground_truth_<pattern>.csv
Format: frame, x, y, z, dx, dy, dz
  x/y/z  : mutlak dünya konumu (metre)
  dx/dy/dz: ilk frame'e göre göreli hareket (metre)

Kullanım:
    python3 ~/code/uav-visual-odometry/sim/scripts/export_ground_truth.py

Özelleştirme (ortam değişkeni):
    PATTERN       : figure8 | lawnmower  (varsayılan: figure8)
    STRIDE        : DROID-SLAM stride    (varsayılan: 3)
    N_SLAM_FRAMES : SLAM frame sayısı    (varsayılan: 100)
    GT_OUT        : çıktı yolu           (varsayılan: pattern'e göre otomatik)

  Figure-8 parametreleri (move_camera.py ile eşleşmeli):
    FIGURE8_AX     : X genliği metre    (varsayılan: 6.0)
    FIGURE8_AY     : Y genliği metre    (varsayılan: 5.0)
    FIGURE8_PERIOD : Bir tur süresi sn  (varsayılan: 30.0)

  Lawnmower parametreleri:
    STEP_SIZE     : m/adım              (varsayılan: 0.30)
    STEP_DELAY    : s/adım              (varsayılan: 0.12)
"""

import csv
import math
import os

# ── Ortak parametreler ────────────────────────────────────────────────────────
PATTERN       = os.environ.get("PATTERN",       "figure8")
CAMERA_FPS    = 15.0
INITIAL_SLEEP = 1.0   # s (move_camera.py başlangıç bekleme)
HEIGHT        = 4.0   # m — 10m'den 4m'ye düşürüldü (parallax artışı)

STRIDE        = int(os.environ.get("STRIDE",        "3"))
N_SLAM_FRAMES = int(os.environ.get("N_SLAM_FRAMES", "100"))

REPO_ROOT = os.path.expanduser("~/code/uav-visual-odometry")

# ── Figure-8 parametreleri (move_camera.py ile eşleşmeli) ────────────────────
FIGURE8_AX     = float(os.environ.get("FIGURE8_AX",     "4.5"))
FIGURE8_AY     = float(os.environ.get("FIGURE8_AY",     "3.5"))
FIGURE8_PERIOD = float(os.environ.get("FIGURE8_PERIOD", "25.0"))

# ── Lawnmower parametreleri ───────────────────────────────────────────────────
STEP_SIZE  = float(os.environ.get("STEP_SIZE",  "0.30"))
STEP_DELAY = float(os.environ.get("STEP_DELAY", "0.12"))
X_MIN, X_MAX = -8.0, 8.0
Y_MIN, Y_MAX = -8.0, 8.0

# ── Çıktı yolu ────────────────────────────────────────────────────────────────
_default_gt_out = os.path.join(
    REPO_ROOT, f"evaluation/ground_truth_{PATTERN}.csv"
)
GT_OUT = os.environ.get("GT_OUT", _default_gt_out)


# ─────────────────────────────────────────────────────────────────────────────
# Figure-8 GT
# ─────────────────────────────────────────────────────────────────────────────

def figure8_pos(t_motion: float) -> tuple[float, float]:
    """
    Hareketi başladıktan t_motion saniye sonraki (x, y) konumu.
    t_motion=0 → (0, 0) — eğri başlangıç noktası.
    """
    theta = 2.0 * math.pi * t_motion / FIGURE8_PERIOD
    x = FIGURE8_AX * math.sin(theta)
    y = FIGURE8_AY * math.sin(2.0 * theta)
    return x, y


def slam_frame_to_world_figure8(slam_frame: int) -> tuple[float, float, float]:
    rec_frame = slam_frame * STRIDE
    t = rec_frame / CAMERA_FPS
    if t < INITIAL_SLEEP:
        return 0.0, 0.0, HEIGHT
    x, y = figure8_pos(t - INITIAL_SLEEP)
    return x, y, HEIGHT


# ─────────────────────────────────────────────────────────────────────────────
# Lawnmower GT (orijinal)
# ─────────────────────────────────────────────────────────────────────────────

def generate_lawnmower() -> list[tuple[float, float]]:
    waypoints = []
    row = 0
    y = Y_MIN
    while y <= Y_MAX + 1e-6:
        xs = []
        x = X_MIN
        while x <= X_MAX + 1e-6:
            xs.append(round(x, 4))
            x += STEP_SIZE
        if row % 2 == 1:
            xs = xs[::-1]
        for px in xs:
            waypoints.append((px, round(y, 4)))
        y += STEP_SIZE * 4
        row += 1
    return waypoints


def slam_frame_to_world_lawnmower(
    slam_frame: int,
    waypoints: list[tuple[float, float]],
) -> tuple[float, float, float]:
    rec_frame = slam_frame * STRIDE
    t = rec_frame / CAMERA_FPS
    if t < INITIAL_SLEEP:
        wx, wy = waypoints[0]
    else:
        idx = int((t - INITIAL_SLEEP) / STEP_DELAY)
        idx = min(idx, len(waypoints) - 1)
        wx, wy = waypoints[idx]
    return wx, wy, HEIGHT


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[export_gt] Pattern       : {PATTERN}")
    print(f"[export_gt] STRIDE        : {STRIDE}")
    print(f"[export_gt] N_SLAM_FRAMES : {N_SLAM_FRAMES}")
    print(f"[export_gt] FPS           : {CAMERA_FPS}")
    print(f"[export_gt] Çıktı         : {GT_OUT}")

    # ── Pozisyon dizisi oluştur ───────────────────────────────────────────────
    if PATTERN == "figure8":
        print(f"[export_gt] Figure-8  A_x={FIGURE8_AX}  A_y={FIGURE8_AY}  T={FIGURE8_PERIOD}s")
        positions = [slam_frame_to_world_figure8(i) for i in range(N_SLAM_FRAMES)]

    elif PATTERN == "lawnmower":
        waypoints = generate_lawnmower()
        print(f"[export_gt] Lawnmower  waypoint={len(waypoints)}  step={STEP_SIZE}m  delay={STEP_DELAY}s")
        positions = [slam_frame_to_world_lawnmower(i, waypoints) for i in range(N_SLAM_FRAMES)]

    else:
        raise ValueError(f"Bilinmeyen PATTERN: '{PATTERN}'")

    # ── Referans (frame 0) ────────────────────────────────────────────────────
    x0, y0, z0 = positions[0]
    print(f"[export_gt] Referans (frame 0) : x={x0:.4f}  y={y0:.4f}  z={z0:.4f}")
    print(f"[export_gt] Son frame ({N_SLAM_FRAMES-1:3d})    : "
          f"x={positions[-1][0]:.4f}  y={positions[-1][1]:.4f}")

    # Toplam yol uzunluğu
    total_path = sum(
        math.sqrt(
            (positions[i][0] - positions[i-1][0])**2 +
            (positions[i][1] - positions[i-1][1])**2
        )
        for i in range(1, len(positions))
    )
    x_range = max(p[0] for p in positions) - min(p[0] for p in positions)
    y_range = max(p[1] for p in positions) - min(p[1] for p in positions)
    print(f"[export_gt] GT toplam yol      : {total_path:.4f} m")
    print(f"[export_gt] X aralığı          : {x_range:.4f} m")
    print(f"[export_gt] Y aralığı          : {y_range:.4f} m")

    os.makedirs(os.path.dirname(GT_OUT), exist_ok=True)
    with open(GT_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "z", "dx", "dy", "dz"])
        for i, (wx, wy, wz) in enumerate(positions):
            writer.writerow([
                i,
                f"{wx:.6f}", f"{wy:.6f}", f"{wz:.6f}",
                f"{wx - x0:.6f}",
                f"{wy - y0:.6f}",
                f"{wz - z0:.6f}",
            ])

    print(f"[export_gt] Tamamlandı. {N_SLAM_FRAMES} satır yazıldı.")
    print()
    print("[export_gt] --- İlk 8 satır ---")
    print("frame, x, y, z, dx, dy, dz")
    for i in range(min(8, len(positions))):
        wx, wy, wz = positions[i]
        print(f"  {i:>3},  {wx:8.4f},  {wy:8.4f},  {wz:.1f},  "
              f"{wx-x0:9.4f},  {wy-y0:9.4f},  {wz-z0:.4f}")


if __name__ == "__main__":
    main()
