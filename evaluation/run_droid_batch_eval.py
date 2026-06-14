#!/usr/bin/env python3
"""
run_droid_batch_eval.py — DROID-SLAM batch değerlendirmesi

DROID-SLAM'ın gerçek performansını ölçer:
  1. Tüm video DROID'e beslenir
  2. terminate() → global bundle adjustment → optimize edilmiş yörünge
  3. Sim3 hizalaması (health=1 çiftleriyle)
  4. Resmi MAE_3D skoru (§9.2)

Online moddan farkı: global BA yapılır → DROID'in gerçek tavanı görülür.

Kullanım:
    conda run -n droid_clean python3 evaluation/run_droid_batch_eval.py
    conda run -n droid_clean python3 evaluation/run_droid_batch_eval.py \\
        --stride 3 --health comp --out evaluation/results_sample
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

_REPO        = os.path.expanduser("~/code/uav-visual-odometry")
_DROID_DIR   = os.path.join(_REPO, "DROID-SLAM")
_DROID_WEIGHTS = os.path.join(_DROID_DIR, "checkpoints", "droid.pth")

# DROID-SLAM modüllerini sys.path'e ekle
for p in [os.path.join(_DROID_DIR, "droid_slam"), _DROID_DIR, _REPO]:
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_VIDEO  = os.path.expanduser("~/Downloads/THYZ_2026_Ornek_Veri_1.MP4")
DEFAULT_CSV    = os.path.expanduser("~/Downloads/THYZ_2026_Ornek_Veri_1_translation.csv")
DEFAULT_OUT    = os.path.join(_REPO, "evaluation/results_sample")


# ── Veri yükleme ──────────────────────────────────────────────────────────────

def load_reference_csv(path: str) -> np.ndarray:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append([float(r["translation_x"]),
                         float(r["translation_y"]),
                         float(r["translation_z"])])
    pos = np.array(rows, dtype=np.float64)
    pos -= pos[0]
    return pos


def make_health_flags(n: int, calib: int, mode: str, seed: int) -> np.ndarray:
    from competition.simulate_health import make_health_flags as _make
    flags = np.zeros(n, dtype=np.int8)
    flags[:min(calib, n)] = 1
    if mode == "comp" and n > calib:
        rest, _ = _make(n - calib, seed=seed, scenario="competition")
        flags[calib:] = rest
    elif mode == "all_alive":
        flags[:] = 1
    elif mode == "all_dead":
        flags[:] = 0
    return flags


# ── DROID batch çalıştırma ────────────────────────────────────────────────────

def run_droid_on_video(video_path: str, ref_pos: np.ndarray,
                       stride: int, image_size: list,
                       fx: float, fy: float, cx: float, cy: float,
                       buffer_size: int = 512) -> np.ndarray:
    """
    Tüm videoyu DROID'e besle → terminate() → (N_frames, 3) yörünge.
    """
    import torch
    import argparse as ap
    from droid import Droid

    args = ap.Namespace(
        weights=_DROID_WEIGHTS, image_size=image_size, buffer=buffer_size,
        stereo=False, disable_vis=True, filter_thresh=2.4, beta=0.3,
        warmup=8, keyframe_thresh=4.0, frontend_thresh=16.0,
        frontend_window=25, frontend_radius=2, frontend_nms=1,
        backend_thresh=22.5, backend_radius=2, backend_nms=3, upsample=False,
    )

    print(f"[DROID] Model yükleniyor: {_DROID_WEIGHTS}")
    droid = Droid(args)
    intrinsics = torch.as_tensor([fx, fy, cx, cy], dtype=torch.float).cuda()
    h_t, w_t   = image_size

    cap     = cv2.VideoCapture(video_path)
    n_total = len(ref_pos)
    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[DROID] Video: {orig_w}×{orig_h}, besleniyor (stride={stride})...")

    t0         = time.perf_counter()
    vid_idx    = 0
    eval_idx   = 0
    frame_list = []   # (eval_idx, frame_np) — stream için

    while eval_idx < n_total:
        ret, frame = cap.read()
        if not ret:
            break
        if vid_idx % stride != 0:
            vid_idx += 1
            continue

        img = cv2.resize(frame, (w_t, h_t))
        img_t = (torch.as_tensor(img).permute(2,0,1).float().unsqueeze(0).cuda())
        droid.track(eval_idx, img_t, intrinsics=intrinsics)
        frame_list.append(img_t)

        eval_idx += 1
        vid_idx  += 1

        if eval_idx % 500 == 0:
            fps = eval_idx / (time.perf_counter() - t0)
            kf  = droid.video.counter.value
            print(f"  [{eval_idx:5d}/{n_total}]  fps={fps:.1f}  keyframes={kf}")

    cap.release()
    n_proc = eval_idx
    print(f"[DROID] Besleme tamamlandı: {n_proc} kare, "
          f"{time.perf_counter()-t0:.1f}s")

    # Global bundle adjustment + tüm kare yörüngesi
    print("[DROID] Global BA başlıyor (terminate)...")
    t1 = time.perf_counter()
    # Terminate ile tüm karelerin optimize edilmiş pozlarını al
    import torch.nn.functional as F
    from trajectory_filler import PoseTrajectoryFiller
    traj_filler = PoseTrajectoryFiller(droid.net, droid.video)

    # stream: her kare için (t, image, intrinsics) üretici
    def stream_gen():
        for i, img_t in enumerate(frame_list):
            yield i, img_t, intrinsics

    # Backend global optimize
    from droid_backend import DroidBackend
    backend = DroidBackend(droid.net, droid.video, args)
    backend(7)  # 7 iterations

    # Tüm karelere poz ata
    camera_trajectory = traj_filler(stream_gen())  # (N, 7) SE3

    print(f"[DROID] BA tamamlandı: {time.perf_counter()-t1:.1f}s")

    # SE3 → (N, 7) tensor → (N, 3) translasyon
    # lietorch SE3: .vec() → [..., tx, ty, tz, qx, qy, qz, qw]
    poses_np = camera_trajectory.vec().cpu().numpy()   # (n_proc, 7)
    translations = poses_np[:, :3]               # (n_proc, 3)

    # Orijinal kare sayısına genişlet (nearest-neighbor)
    # translations[j] → orijinal kare j*stride'a karşılık gelir
    full_traj = np.zeros((n_total, 3), dtype=np.float64)
    for orig_i in range(n_total):
        j = min(orig_i // stride, len(translations) - 1)
        full_traj[orig_i] = translations[j]
    return full_traj


# ── Sim3 hizalama + skor hesaplama ───────────────────────────────────────────

def evaluate_trajectory(droid_traj: np.ndarray, ref_pos: np.ndarray,
                        health: np.ndarray, out_dir: str) -> dict:
    """
    DROID yörüngesini Sim3 ile referansa hizala, MAE_3D hesapla.

    health=1 karelerde referansı direkt kullan (sıfır hata).
    health=0 karelerde Sim3 ile hizalanmış DROID tahminini kullan.
    """
    from competition.sim3_aligner import umeyama_alignment
    from competition.score_official import score_summary

    n = len(ref_pos)

    # Sim3: health=1 çiftleriyle tek seferlik Umeyama (batch eval — rolling rejection yok)
    alive_idx = [i for i in range(n) if health[i] == 1]
    src_pts = droid_traj[alive_idx]   # (K, 3)
    dst_pts = ref_pos[alive_idx]       # (K, 3)

    # Debug: trajektori istatistikleri
    print(f"[Debug] droid_traj range: x=[{droid_traj[:,0].min():.3f}, {droid_traj[:,0].max():.3f}]  "
          f"y=[{droid_traj[:,1].min():.3f}, {droid_traj[:,1].max():.3f}]  "
          f"z=[{droid_traj[:,2].min():.3f}, {droid_traj[:,2].max():.3f}]")
    print(f"[Debug] ref_pos range:   x=[{ref_pos[:,0].min():.3f}, {ref_pos[:,0].max():.3f}]  "
          f"y=[{ref_pos[:,1].min():.3f}, {ref_pos[:,1].max():.3f}]  "
          f"z=[{ref_pos[:,2].min():.3f}, {ref_pos[:,2].max():.3f}]")

    s, R, t = umeyama_alignment(src_pts, dst_pts)
    est_calib = (s * (R @ src_pts.T)).T + t
    rmse_cal = float(np.sqrt(np.mean(np.sum((est_calib - dst_pts)**2, axis=1))))
    calibrated = len(alive_idx) >= 3
    print(f"[Eval] Sim3 (batch): n_pairs={len(alive_idx)}  s={s:.4f}  RMSE={rmse_cal:.4f}m  calibrated={calibrated}")

    # Optimal strateji: health=1 → ref, health=0 → Sim3(DROID)
    est_pos = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        if health[i] == 1:
            est_pos[i] = ref_pos[i]
        elif calibrated:
            est_pos[i] = s * (R @ droid_traj[i]) + t
        else:
            est_pos[i] = ref_pos[i]

    # Resmi MAE_3D★
    dead_idx = [i for i in range(n) if health[i] == 0]
    dead_err = sum(
        math.sqrt((est_pos[i,0]-ref_pos[i,0])**2 +
                  (est_pos[i,1]-ref_pos[i,1])**2 +
                  (est_pos[i,2]-ref_pos[i,2])**2)
        for i in dead_idx
    )
    mae_3d_official = dead_err / n if n > 0 else float("nan")
    mae_3d_dead     = dead_err / len(dead_idx) if dead_idx else float("nan")

    errs_2d = np.sqrt(
        (est_pos[:,0]-ref_pos[:,0])**2 + (est_pos[:,1]-ref_pos[:,1])**2
    )
    dead_mask = health == 0
    rmse_2d   = float(np.sqrt(np.mean(errs_2d[dead_mask]**2))) if dead_mask.any() else float("nan")
    max_drift = float(np.max(errs_2d[dead_mask])) if dead_mask.any() else float("nan")

    metrics = {
        "n_frames":        n,
        "n_dead":          len(dead_idx),
        "n_dead_pct":      round(100*len(dead_idx)/n, 1),
        "mae_3d_official": round(mae_3d_official, 4),
        "mae_3d_dead":     round(mae_3d_dead, 4),
        "rmse_2d_dead":    round(rmse_2d, 4),
        "max_drift_2d":    round(max_drift, 4),
        "final_drift_2d":  round(float(errs_2d[-1]), 4),
        "sim3_calibrated": calibrated,
        "sim3_pairs":      len(alive_idx),
        "sim3_scale":      round(float(s), 4),
        "sim3_rmse":       round(float(rmse_cal), 4),
    }

    os.makedirs(out_dir, exist_ok=True)
    _save_results(ref_pos, est_pos, droid_traj, health, errs_2d, metrics, out_dir)
    _print_summary(metrics)
    return metrics


# ── Çıktı ─────────────────────────────────────────────────────────────────────

def _print_summary(m: dict) -> None:
    print()
    print("=" * 56)
    print("  DROID-SLAM BATCH DEĞERLENDİRME SONUÇLARI")
    print("=" * 56)
    print(f"  Kare sayısı      : {m['n_frames']}")
    print(f"  Health=0 kare    : {m['n_dead']} ({m['n_dead_pct']}%)")
    print(f"  {'─'*50}")
    print(f"  MAE_3D★ (resmi)  : {m['mae_3d_official']} m   ← §9.2 yarışma skoru")
    print(f"  MAE_3D (dead)    : {m['mae_3d_dead']} m")
    print(f"  RMSE_2D (dead)   : {m['rmse_2d_dead']} m")
    print(f"  Max drift 2D     : {m['max_drift_2d']} m")
    print(f"  Final drift 2D   : {m['final_drift_2d']} m")
    print(f"  {'─'*50}")
    print(f"  Sim3 kalibre     : {'Evet' if m['sim3_calibrated'] else 'Hayır'}"
          f"  (pairs={m['sim3_pairs']}, scale={m['sim3_scale']}, RMSE={m['sim3_rmse']})")
    print("=" * 56)


def _save_results(ref_pos, est_pos, droid_traj, health, errs_2d, metrics, out_dir):
    n = len(ref_pos)

    # CSV
    csv_path = os.path.join(out_dir, "droid_batch_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "frame","health","ref_x","ref_y","ref_z",
            "droid_x","droid_y","droid_z","est_x","est_y","est_z","err_2d"])
        w.writeheader()
        for i in range(n):
            w.writerow({
                "frame":   i, "health": int(health[i]),
                "ref_x":   round(float(ref_pos[i,0]),4),
                "ref_y":   round(float(ref_pos[i,1]),4),
                "ref_z":   round(float(ref_pos[i,2]),4),
                "droid_x": round(float(droid_traj[i,0]),6),
                "droid_y": round(float(droid_traj[i,1]),6),
                "droid_z": round(float(droid_traj[i,2]),6),
                "est_x":   round(float(est_pos[i,0]),4),
                "est_y":   round(float(est_pos[i,1]),4),
                "est_z":   round(float(est_pos[i,2]),4),
                "err_2d":  round(float(errs_2d[i]),4),
            })
    print(f"[Eval] CSV: {csv_path}")

    # Özet
    txt_path = os.path.join(out_dir, "droid_batch_summary.txt")
    with open(txt_path, "w") as f:
        f.write("DROID-SLAM BATCH DEĞERLENDİRME\n" + "="*40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:<22}: {v}\n")
    print(f"[Eval] Özet: {txt_path}")

    # Grafikler
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Yörünge
        fig, ax = plt.subplots(figsize=(12,10))
        ax.plot(ref_pos[:,0], ref_pos[:,1], color="#2196F3", lw=1.0, alpha=0.7,
                label="Referans yörüngesi")
        h1 = [(est_pos[i,0], est_pos[i,1]) for i in range(n) if health[i]==1]
        h0 = [(est_pos[i,0], est_pos[i,1]) for i in range(n) if health[i]==0]
        if h1:
            ax.scatter(*zip(*h1), s=2, color="#4CAF50", alpha=0.5, label="Tahmin (health=1)")
        if h0:
            ax.scatter(*zip(*h0), s=2, color="#F44336", alpha=0.4, label="Tahmin (health=0)")
        ax.plot(ref_pos[0,0], ref_pos[0,1], "go", ms=10, label="Başlangıç")
        ax.plot(ref_pos[-1,0], ref_pos[-1,1], "rs", ms=10, label="Bitiş")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_title(f"DROID-SLAM Batch — Yörünge ({n} kare)\n"
                     f"MAE_3D★={metrics['mae_3d_official']}m  "
                     f"Sim3 scale={metrics['sim3_scale']}", fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
        plt.tight_layout()
        out = os.path.join(out_dir, "droid_batch_trajectory.png")
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[Eval] Yörünge: {out}")

        # Hata
        fig, axes = plt.subplots(2, 1, figsize=(14,8), sharex=True)
        frames = np.arange(n)
        ax1 = axes[0]
        in_dead = False; ds = 0
        for i in range(n):
            if not in_dead and health[i]==0:
                in_dead=True; ds=i
            elif in_dead and (health[i]==1 or i==n-1):
                ax1.axvspan(ds, i, color="#FFCDD2", alpha=0.4); in_dead=False
        if in_dead: ax1.axvspan(ds, n, color="#FFCDD2", alpha=0.4)
        ax1.axhline(1.0, color="#FF5722", ls=":", lw=1.2, label="1m eşiği", alpha=0.8)
        ax1.axhline(5.0, color="#9C27B0", ls=":", lw=1.0, label="5m eşiği", alpha=0.7)
        ax1.plot(frames, errs_2d, color="#2196F3", lw=0.7, label="2D Hata (m)")
        ax1.set_ylabel("2D Hata (m)", fontsize=11)
        ax1.set_title(f"DROID-SLAM Batch — Konum Hatası ({n} kare)", fontsize=11)
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3); ax1.set_ylim(bottom=0)
        ax2 = axes[1]
        ax2.fill_between(frames, health.astype(float), 0, where=health==1,
                         color="#4CAF50", alpha=0.6, label="health=1")
        ax2.fill_between(frames, 1, health.astype(float), where=health==0,
                         color="#F44336", alpha=0.4, label="health=0")
        ax2.set_xlabel("Kare"); ax2.set_ylabel("Health"); ax2.set_yticks([0,1])
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(out_dir, "droid_batch_error.png")
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[Eval] Hata: {out}")
    except ImportError:
        print("[Eval] matplotlib yok, grafikler atlandı")


# ── Ana giriş ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DROID-SLAM batch değerlendirme")
    parser.add_argument("--video",         default=DEFAULT_VIDEO)
    parser.add_argument("--csv",           default=DEFAULT_CSV)
    parser.add_argument("--out",           default=DEFAULT_OUT)
    parser.add_argument("--stride",        type=int,   default=3)
    parser.add_argument("--calib_frames",  type=int,   default=450)
    parser.add_argument("--health",        default="comp",
                        choices=["comp","calib_only","all_dead","all_alive"])
    parser.add_argument("--seed",          type=int,   default=42)
    # 216×384 = 16:9, divisible by 8 (req. by DROID); 1920/5=384, 1080/5=216
    # fx=1389.7*(384/1920)=277.9  fy=1387.1*(216/1080)=277.4
    # cx=954*(384/1920)=190.8     cy=558.9*(216/1080)=111.8
    parser.add_argument("--fx",            type=float, default=277.9)
    parser.add_argument("--fy",            type=float, default=277.4)
    parser.add_argument("--cx",            type=float, default=190.8)
    parser.add_argument("--cy",            type=float, default=111.8)
    # image_size=[h, w]; 216×384 preserves 16:9 and both divisible by 8
    parser.add_argument("--image_size",    type=int, nargs=2, default=[216, 384])
    parser.add_argument("--buffer",        type=int,   default=512)
    parser.add_argument("--max_frames",    type=int,   default=None)
    args = parser.parse_args()

    print(f"[main] CSV yükleniyor: {args.csv}")
    ref_pos = load_reference_csv(args.csv)
    n_use   = min(len(ref_pos), args.max_frames) if args.max_frames else len(ref_pos)
    ref_pos = ref_pos[:n_use]

    health = make_health_flags(n_use, args.calib_frames, args.health, args.seed)
    print(f"[main] {n_use} kare  health=0: {(health==0).sum()}  "
          f"({100*(health==0).sum()/n_use:.1f}%)")

    # DROID batch çalıştır
    droid_traj = run_droid_on_video(
        video_path   = args.video,
        ref_pos      = ref_pos,
        stride       = args.stride,
        image_size   = args.image_size,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        buffer_size  = args.buffer,
    )

    print(f"[main] DROID yörüngesi: {droid_traj.shape}")

    # Hizala ve değerlendir
    evaluate_trajectory(droid_traj, ref_pos, health, args.out)


if __name__ == "__main__":
    main()
