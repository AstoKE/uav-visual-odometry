#!/usr/bin/env python3
"""
run_estimator_on_dataset.py — SLAM trajectory → SLAMPoseEstimator pipeline

Bir SLAM trajectory CSV'sini (droid veya orb) alır, simüle edilmiş
health flag'leri ile SLAMPoseEstimator'dan geçirir ve sonuçları CSV'ye kaydeder.

Giriş formatı (trajectory CSV):
    frame, x, y, z, [...]   — ham SLAM kümülatif pozisyonlar

Çıktı (est_*.csv):
    frame, health, gt_x, gt_y, est_x, est_y, est_z, slam_dx, slam_dy

Kullanım:
    python3 competition/run_estimator_on_dataset.py --model droid
    python3 competition/run_estimator_on_dataset.py --model orb
    python3 competition/run_estimator_on_dataset.py --model droid --out custom.csv
"""

import argparse
import csv
import os
import sys
import time
import logging
import numpy as np

_REPO = os.path.expanduser("~/code/uav-visual-odometry")
sys.path.insert(0, _REPO)

from competition.slam_pose_estimator import SLAMPoseEstimator
from competition.simulate_health import make_health_flags

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_estimator")

TRAJ_MAP = {
    "droid": os.path.join(_REPO, "slam/outputs/trajectory_droid_figure8.csv"),
    "orb":   os.path.join(_REPO, "slam/outputs/trajectory_orb_figure8.csv"),
}
GT_PATH  = os.path.join(_REPO, "evaluation/ground_truth_figure8_v3.csv")
OUT_DIR  = os.path.join(_REPO, "competition/results")


# ── Yardımcı yükleyiciler ─────────────────────────────────────────────────────

def load_trajectory(path: str) -> np.ndarray:
    """SLAM trajectory CSV'yi (frame, x, y) olarak yükle."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append((int(row["frame"]), float(row["x"]), float(row["y"])))
    return np.array(rows)   # (N, 3)


def load_gt(path: str) -> np.ndarray:
    """GT CSV'yi (frame, dx, dy) olarak yükle — kümülatif displacement."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append((
                int(row["frame"]),
                float(row["dx"]),
                float(row["dy"]),
                float(row.get("dz", 0.0)),
            ))
    return np.array(rows)   # (N, 4)


# ── Ana pipeline ──────────────────────────────────────────────────────────────

def run(model: str, traj_path: str, gt_path: str, out_path: str) -> None:
    log.info(f"Model: {model.upper()}")
    log.info(f"Trajectory: {traj_path}")

    traj = load_trajectory(traj_path)
    gt   = load_gt(gt_path)
    N    = min(len(traj), len(gt))

    log.info(f"Frame sayısı: {N}")

    # Health flag üret ve kaydet
    health_arr, _ = make_health_flags(n_frames=N, seed=42)
    health_path = os.path.join(OUT_DIR, "health_flags.npy")
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(health_path, health_arr)
    log.info(f"Health=1: {health_arr.sum()}  Health=0: {(health_arr==0).sum()}")

    # Per-frame SLAM artımlı deltaları hesapla
    slam_x = traj[:N, 1]
    slam_y = traj[:N, 2]
    slam_dx = np.diff(slam_x, prepend=slam_x[0])
    slam_dy = np.diff(slam_y, prepend=slam_y[0])
    slam_dx[0] = 0.0
    slam_dy[0] = 0.0

    # GT kümülatif pozisyon (referans olarak kullanılacak)
    gt_x_cum = gt[:N, 1]
    gt_y_cum = gt[:N, 2]
    gt_z_cum = gt[:N, 3]

    # Estimator
    estimator = SLAMPoseEstimator(calib_min_frames=30, calib_update_every=50)

    results = []
    frame_times = []

    for i in range(N):
        health = int(health_arr[i])

        # Referans pozisyon: health=1'de GT kümülatif (gerçek konumu simüle eder)
        if health == 1:
            ref_pos = (float(gt_x_cum[i]), float(gt_y_cum[i]), float(gt_z_cum[i]))
        else:
            ref_pos = None

        t0 = time.perf_counter()
        est_x, est_y, est_z = estimator.update(
            slam_dx=float(slam_dx[i]),
            slam_dy=float(slam_dy[i]),
            ref_pos=ref_pos,
            health=health,
        )
        dt = time.perf_counter() - t0
        frame_times.append(dt)

        results.append({
            "frame":   i,
            "health":  health,
            "gt_x":    round(float(gt_x_cum[i]), 5),
            "gt_y":    round(float(gt_y_cum[i]), 5),
            "est_x":   round(est_x, 5),
            "est_y":   round(est_y, 5),
            "est_z":   round(est_z, 5),
            "slam_dx": round(float(slam_dx[i]), 7),
            "slam_dy": round(float(slam_dy[i]), 7),
        })

        if i % 50 == 0:
            state = estimator.get_state()
            calib_str = "[CALIB]" if estimator.calibrated else f"[n={estimator.calib_data_count}]"
            log.info(
                f"Frame {i:3d}  h={health}  "
                f"gt=({gt_x_cum[i]:6.2f},{gt_y_cum[i]:6.2f})  "
                f"est=({est_x:6.2f},{est_y:6.2f})  {calib_str}"
            )

    # Çıktı CSV kaydet
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # Runtime özeti
    frame_times = np.array(frame_times[1:])  # ilk frame atla (setup)
    avg_fps = 1.0 / frame_times.mean() if frame_times.mean() > 0 else 0
    max_latency_ms = float(frame_times.max() * 1000)

    log.info(f"Sonuçlar kaydedildi: {out_path}")
    log.info(f"Avg FPS: {avg_fps:.1f}  Max latency: {max_latency_ms:.2f}ms")
    log.info(f"Kalibrasyon: {estimator.calibrated}  "
             f"Calib frames: {estimator.calib_data_count}")

    # Runtime dosyasına ekle
    runtime_path = os.path.join(OUT_DIR, "runtime.txt")
    _append_runtime(runtime_path, model, avg_fps, max_latency_ms, estimator)


def _append_runtime(path: str, model: str, fps: float, max_ms: float,
                    est: SLAMPoseEstimator) -> None:
    lines = []
    if os.path.exists(path):
        with open(path) as f:
            lines = f.readlines()
    # Mevcut modeli güncelle veya ekle
    marker = f"[{model.upper()}]"
    new_lines = [l for l in lines if not l.startswith(marker)]
    state = est.get_state()
    M = np.array(state["M"])
    new_lines.append(
        f"{marker}\n"
        f"  avg_fps       = {fps:.1f}\n"
        f"  max_latency   = {max_ms:.3f} ms\n"
        f"  calibrated    = {est.calibrated}\n"
        f"  calib_frames  = {est.calib_data_count}\n"
        f"  M             = [[{M[0,0]:.5f},{M[0,1]:.5f}],[{M[1,0]:.5f},{M[1,1]:.5f}]]\n"
        f"  scale_x       = {state['scale_x']:.5f}\n"
        f"  scale_y       = {state['scale_y']:.5f}\n\n"
    )
    with open(path, "w") as f:
        f.writelines(new_lines)
    log.info(f"Runtime kaydedildi: {path}")


def main():
    parser = argparse.ArgumentParser(description="SLAM → SLAMPoseEstimator pipeline")
    parser.add_argument("--model", choices=["droid", "orb"], default="droid")
    parser.add_argument("--traj", default=None,
                        help="SLAM trajectory CSV (varsayılan: model'e göre)")
    parser.add_argument("--gt",   default=GT_PATH,
                        help="Ground truth CSV")
    parser.add_argument("--out",  default=None,
                        help="Çıktı CSV (varsayılan: results/est_<model>.csv)")
    args = parser.parse_args()

    traj_path = args.traj or TRAJ_MAP[args.model]
    out_path  = args.out  or os.path.join(OUT_DIR, f"est_{args.model}.csv")

    if not os.path.exists(traj_path):
        log.error(f"Trajectory bulunamadı: {traj_path}")
        sys.exit(1)
    if not os.path.exists(args.gt):
        log.error(f"GT bulunamadı: {args.gt}")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    run(args.model, traj_path, args.gt, out_path)


if __name__ == "__main__":
    main()
