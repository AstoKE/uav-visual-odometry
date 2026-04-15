#!/usr/bin/env python3
"""
apply_scale.py — SLAM trajectory'e metrik ölçek uygular.

Scale faktörünü iki yöntemle hesaplar ve raporlar:
  1. Toplam yol uzunluğu (path length) oranı  [birincil]
  2. Son frame mesafesi oranı (net displacement) [kontrol]

Sonra scale'i delta_trajectory.csv'e uygular ve
delta_trajectory_scaled.csv üretir.

GİRDİ:
  slam/outputs/trajectory.csv        (absolut SLAM pozisyonları)
  slam/outputs/delta_trajectory.csv  (SLAM fark vektörleri)
  evaluation/ground_truth.csv        (GT dünya pozisyonları ve farkları)

ÇIKTI:
  slam/outputs/delta_trajectory_scaled.csv
  slam/outputs/scale_report.txt

Kullanım:
    python3 ~/code/uav-visual-odometry/slam/scripts/apply_scale.py
"""

import csv
import math
import os

REPO_ROOT   = os.path.expanduser("~/code/uav-visual-odometry")
TRAJ_IN     = os.path.join(REPO_ROOT, "slam/outputs/trajectory.csv")
DELTA_IN    = os.path.join(REPO_ROOT, "slam/outputs/delta_trajectory.csv")
GT_IN       = os.path.join(REPO_ROOT, "evaluation/ground_truth.csv")
DELTA_OUT   = os.path.join(REPO_ROOT, "slam/outputs/delta_trajectory_scaled.csv")
REPORT_OUT  = os.path.join(REPO_ROOT, "slam/outputs/scale_report.txt")


def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def path_length(rows: list[dict], xk: str, yk: str, zk: str = None) -> float:
    total = 0.0
    for i in range(1, len(rows)):
        dx = float(rows[i][xk]) - float(rows[i-1][xk])
        dy = float(rows[i][yk]) - float(rows[i-1][yk])
        dz = (float(rows[i][zk]) - float(rows[i-1][zk])) if zk else 0.0
        total += math.sqrt(dx**2 + dy**2 + dz**2)
    return total


def net_distance_2d(rows: list[dict], xk: str, yk: str) -> float:
    dx = float(rows[-1][xk]) - float(rows[0][xk])
    dy = float(rows[-1][yk]) - float(rows[0][yk])
    return math.sqrt(dx**2 + dy**2)


def main() -> None:
    # ── Dosyaları yükle ───────────────────────────────────────────────────────
    for p in [TRAJ_IN, DELTA_IN, GT_IN]:
        if not os.path.isfile(p):
            print(f"[apply_scale] HATA: Dosya bulunamadi: {p}")
            raise SystemExit(1)

    traj_rows  = load_csv(TRAJ_IN)
    delta_rows = load_csv(DELTA_IN)
    gt_rows    = load_csv(GT_IN)

    if len(traj_rows) != len(gt_rows):
        print(f"[apply_scale] UYARI: Frame sayisi eslesmiyor: "
              f"SLAM={len(traj_rows)}  GT={len(gt_rows)}")
        n = min(len(traj_rows), len(gt_rows))
        traj_rows  = traj_rows[:n]
        delta_rows = delta_rows[:n]
        gt_rows    = gt_rows[:n]

    # ── Yol uzunlukları ───────────────────────────────────────────────────────
    slam_path_3d = path_length(traj_rows,  "x", "y", "z")
    slam_path_2d = path_length(traj_rows,  "x", "y")
    gt_path_2d   = path_length(gt_rows,    "x", "y")

    # Net yerleşim
    slam_net_2d  = net_distance_2d(traj_rows, "x", "y")
    gt_net_2d    = net_distance_2d(gt_rows,   "x", "y")

    # ── Scale faktörü (birincil: 2D yol uzunluğu oranı) ──────────────────────
    if slam_path_2d < 1e-9:
        print("[apply_scale] HATA: SLAM 2D yol uzunlugu sifira yakin. Hareket algilaniyor mu?")
        raise SystemExit(1)

    scale_path   = gt_path_2d  / slam_path_2d
    scale_net    = (gt_net_2d  / slam_net_2d) if slam_net_2d > 1e-9 else float("nan")

    # ── Rapor ─────────────────────────────────────────────────────────────────
    lines = [
        "=" * 60,
        "SCALE FACTOR RAPORU",
        "=" * 60,
        "",
        f"Frame sayisi        : {len(traj_rows)}",
        "",
        "-- SLAM (normalize birim) --",
        f"  2D yol uzunlugu   : {slam_path_2d:.8f}",
        f"  3D yol uzunlugu   : {slam_path_3d:.8f}",
        f"  Net 2D yerlesim   : {slam_net_2d:.8f}",
        "",
        "-- Ground Truth (metre) --",
        f"  2D yol uzunlugu   : {gt_path_2d:.4f} m",
        f"  Net 2D yerlesim   : {gt_net_2d:.4f} m",
        "",
        "-- Scale Faktorleri --",
        f"  Yol uzunlugu orani: {scale_path:.4f}  [birincil]",
        f"  Net mesafe orani  : {scale_net:.4f}  [kontrol]",
        "",
        f"  Kullanilan scale  : {scale_path:.4f} m/birim",
        "",
        "NOT: SLAM monokulur calistiginda scale gercek metrige",
        "esit degildir. Bu deger, GT yol uzunluguna gore normalize",
        "edilmis bir olcek tahminidir. Koordinat cercevesi farki",
        "(kamera frame vs dunya frame) apply_scale ici olmadan",
        "compare_trajectory.py tarafindan hizalanir.",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print(report)

    with open(REPORT_OUT, "w") as f:
        f.write(report + "\n")
    print(f"\n[apply_scale] Rapor: {REPORT_OUT}")

    # ── Scale uygula ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(DELTA_OUT), exist_ok=True)

    with open(DELTA_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "dx", "dy", "dz", "dist_2d", "dist_3d"])
        for row in delta_rows:
            dx = float(row["dx"]) * scale_path
            dy = float(row["dy"]) * scale_path
            dz = float(row["dz"]) * scale_path
            dist_2d = math.sqrt(dx**2 + dy**2)
            dist_3d = math.sqrt(dx**2 + dy**2 + dz**2)
            writer.writerow([
                int(row["frame"]),
                f"{dx:.6f}", f"{dy:.6f}", f"{dz:.6f}",
                f"{dist_2d:.6f}", f"{dist_3d:.6f}",
            ])

    print(f"[apply_scale] Cikti : {DELTA_OUT}")
    print()

    # Doğrulama: scaled son frame değeri
    with open(DELTA_OUT, newline="") as f:
        scaled_rows = list(csv.DictReader(f))
    last = scaled_rows[-1]
    print(f"[apply_scale] Scaled son frame delta: "
          f"dx={float(last['dx']):.4f} m  "
          f"dy={float(last['dy']):.4f} m  "
          f"dz={float(last['dz']):.4f} m  "
          f"dist_2d={float(last['dist_2d']):.4f} m")

    gt_last = gt_rows[-1]
    print(f"[apply_scale] GT    son frame delta  : "
          f"dx={float(gt_last['dx']):.4f} m  "
          f"dy={float(gt_last['dy']):.4f} m")


if __name__ == "__main__":
    main()
