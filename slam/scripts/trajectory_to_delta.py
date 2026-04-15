#!/usr/bin/env python3
"""
trajectory_to_delta.py — trajectory.csv'den delta (göreli hareket) hesaplar.

Okur : slam/outputs/trajectory.csv   (frame, x, y, z, qx, qy, qz, qw)
Yazar: slam/outputs/delta_trajectory.csv  (frame, dx, dy, dz, dist_2d, dist_3d)

Hesaplama:
    - İlk frame referans alınır (x0, y0, z0)
    - dx = x - x0,  dy = y - y0,  dz = z - z0
    - dist_2d = sqrt(dx^2 + dy^2)          (yatay mesafe)
    - dist_3d = sqrt(dx^2 + dy^2 + dz^2)  (toplam mesafe)

Not: DROID-SLAM monoküler çalıştığında çıktı normalize edilmiş birimler
içindedir (metrik değil). Gerçek metrik ölçek için ground-truth kıyaslaması
ya da stereo/depth bilgisi gerekir. Bu script göreli hareketin yönünü ve
deseni analiz etmek için kullanılır.

Kullanım:
    python3 ~/code/uav-visual-odometry/slam/scripts/trajectory_to_delta.py

Ortam değişkeni ile özelleştirme:
    TRAJ_IN   : giriş CSV yolu
    DELTA_OUT : çıkış CSV yolu
"""

import csv
import math
import os

# ── Ayarlar ──────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.expanduser("~/code/uav-visual-odometry")
TRAJ_IN   = os.environ.get(
    "TRAJ_IN",
    os.path.join(REPO_ROOT, "slam/outputs/trajectory.csv"),
)
DELTA_OUT = os.environ.get(
    "DELTA_OUT",
    os.path.join(REPO_ROOT, "slam/outputs/delta_trajectory.csv"),
)
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    if not os.path.isfile(TRAJ_IN):
        print(f"[trajectory_to_delta] HATA: Girdi dosyasi bulunamadi: {TRAJ_IN}")
        raise SystemExit(1)

    with open(TRAJ_IN, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("[trajectory_to_delta] HATA: trajectory.csv bos.")
        raise SystemExit(1)

    # Referans: ilk frame
    x0 = float(rows[0]["x"])
    y0 = float(rows[0]["y"])
    z0 = float(rows[0]["z"])

    print(f"[trajectory_to_delta] Girdi  : {TRAJ_IN}")
    print(f"[trajectory_to_delta] Cikti  : {DELTA_OUT}")
    print(f"[trajectory_to_delta] Frame sayisi: {len(rows)}")
    print(f"[trajectory_to_delta] Referans (frame 0): x0={x0:.6f}  y0={y0:.6f}  z0={z0:.6f}")

    os.makedirs(os.path.dirname(DELTA_OUT), exist_ok=True)

    deltas = []
    for row in rows:
        dx = float(row["x"]) - x0
        dy = float(row["y"]) - y0
        dz = float(row["z"]) - z0
        dist_2d = math.sqrt(dx**2 + dy**2)
        dist_3d = math.sqrt(dx**2 + dy**2 + dz**2)
        deltas.append({
            "frame":   int(row["frame"]),
            "dx":      dx,
            "dy":      dy,
            "dz":      dz,
            "dist_2d": dist_2d,
            "dist_3d": dist_3d,
        })

    with open(DELTA_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "dx", "dy", "dz", "dist_2d", "dist_3d"])
        for d in deltas:
            writer.writerow([
                d["frame"],
                f"{d['dx']:.8f}",
                f"{d['dy']:.8f}",
                f"{d['dz']:.8f}",
                f"{d['dist_2d']:.8f}",
                f"{d['dist_3d']:.8f}",
            ])

    # Özet istatistikler
    max_dist_2d = max(d["dist_2d"] for d in deltas)
    max_dist_3d = max(d["dist_3d"] for d in deltas)
    final       = deltas[-1]

    print()
    print("[trajectory_to_delta] --- Ozet ---")
    print(f"  Son frame delta     : dx={final['dx']:.6f}  dy={final['dy']:.6f}  dz={final['dz']:.6f}")
    print(f"  Maks 2D mesafe      : {max_dist_2d:.6f}")
    print(f"  Maks 3D mesafe      : {max_dist_3d:.6f}")
    print(f"  Kaydedilen          : {DELTA_OUT}")
    print()
    print("[trajectory_to_delta] --- Ilk 5 satir ---")
    print("frame, dx, dy, dz, dist_2d, dist_3d")
    for d in deltas[:5]:
        print(f"  {d['frame']:>3},  {d['dx']:>10.6f},  {d['dy']:>10.6f},  "
              f"{d['dz']:>10.6f},  {d['dist_2d']:>10.6f},  {d['dist_3d']:>10.6f}")


if __name__ == "__main__":
    main()
