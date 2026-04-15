#!/usr/bin/env python3
"""
run_competition.py — Yarışma ana çalıştırıcı

Akış:
  1. /session/start → kamera parametrelerini al
  2. Her frame için:
       a. /frame/next → (frame_img, ref_pos, health)
       b. estimator.update(frame, ref_pos, health)
       c. health=1 → ref'i direkt gönder (veya kendi tahmin)
          health=0 → estimator tahminini gönder
       d. /frame/result → tahmin gönder
  3. /session/end → skor al

Kullanım:
    python3 ~/code/uav-visual-odometry/competition/run_competition.py \
        --url http://SUNUCU_IP:PORT \
        --token TOKEN \
        --calib 457 457 320 240

Opsiyonel (test için yerel video):
    python3 run_competition.py --offline --video /path/to/video.mp4
"""

import argparse
import logging
import sys
import time
import os
import json
import csv
import numpy as np
import cv2

# Proje kökünü import yoluna ekle
_REPO = os.path.expanduser("~/code/uav-visual-odometry")
sys.path.insert(0, _REPO)

from competition.estimator import OnlineEstimator
from competition.client    import CompetitionClient, FrameData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_competition")


# ─────────────────────────────────────────────────────────────────────────────
# Strateji: health=1'de ne gönderilir?
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_HEALTHY = "reference"
# "reference" → sunucudan gelen ref değeri gönder (güvenli, puan kaybetme)
# "own"       → kendi tahminini gönder (kalibrasyonu test etmek için)


def run_online(args):
    """Gerçek sunucu ile bağlantı."""

    client = CompetitionClient(base_url=args.url, token=args.token)

    log.info("Oturum başlatılıyor...")
    cam = client.start_session()
    log.info(f"Kamera: fx={cam.fx}, fy={cam.fy}, cx={cam.cx}, cy={cam.cy}")

    # Argümandan gelen kalibrasyon öncelikli, sonra sunucudan gelen
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    if args.calib:
        fx, fy, cx, cy = args.calib

    estimator = OnlineEstimator(fx=fx, fy=fy, cx=cx, cy=cy)

    results_log = []
    frame_count  = 0
    health0_count = 0
    t_start = time.time()

    log.info("Frame işleme başlıyor...")

    while True:
        fd = client.get_next_frame()
        if fd is None:
            break

        frame_count += 1
        ref_pos = (fd.ref_x, fd.ref_y, fd.ref_z)

        # Estimator güncelle
        est_x, est_y, est_z = estimator.update(fd.image, ref_pos, fd.health)

        # Ne göndereceğimize karar ver
        if fd.health == 1 and STRATEGY_HEALTHY == "reference":
            send_x, send_y, send_z = fd.ref_x, fd.ref_y, fd.ref_z
        else:
            send_x, send_y, send_z = est_x, est_y, est_z

        if fd.health == 0:
            health0_count += 1

        # Sunucuya gönder
        ok = client.submit_result(fd.frame_id, send_x, send_y, send_z)

        # Log
        results_log.append({
            "frame":   fd.frame_id,
            "health":  fd.health,
            "ref_x":   fd.ref_x, "ref_y":  fd.ref_y,  "ref_z":  fd.ref_z,
            "est_x":   est_x,    "est_y":  est_y,      "est_z":  est_z,
            "sent_x":  send_x,   "sent_y": send_y,     "sent_z": send_z,
            "ok":      ok,
        })

        if frame_count % 50 == 0:
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            calib_str = f"[CALIB OK]" if estimator.calibrated else \
                        f"[calib {estimator.calib_data_count}/{estimator.calib_data_count}]"
            log.info(
                f"Frame {frame_count:4d}  health={fd.health}  "
                f"pos=({send_x:6.2f},{send_y:6.2f},{send_z:6.2f})  "
                f"fps={fps:.1f}  {calib_str}"
            )

    elapsed = time.time() - t_start
    log.info(f"\n{'='*55}")
    log.info(f"Toplam frame    : {frame_count}")
    log.info(f"Health=0 frame  : {health0_count}")
    log.info(f"Süre            : {elapsed:.1f}s  ({frame_count/elapsed:.1f} fps)")
    log.info(f"Kalibre edildi  : {estimator.calibrated}")
    est_state = estimator.get_state()
    log.info(f"Scale x/y       : {est_state['scale_x']:.4f} / {est_state['scale_y']:.4f}")
    log.info(f"Sign x/y        : {est_state['sign_x']:.0f} / {est_state['sign_y']:.0f}")
    log.info(f"Swap xy         : {est_state['swap_xy']}")

    # Sonuçları kaydet
    _save_results(results_log)

    # Oturumu kapat
    try:
        final = client.end_session()
        log.info(f"Final skor: {final}")
    except Exception as e:
        log.warning(f"Session end hatası: {e}")


def run_offline(args):
    """
    Yerel video/görüntü dizini ile offline test.
    Ground truth CSV varsa RMSE hesaplar.
    """
    if args.video and os.path.isfile(args.video):
        frames = _load_video_frames(args.video)
        log.info(f"Video yüklendi: {len(frames)} frame")
    elif args.imgdir and os.path.isdir(args.imgdir):
        frames = _load_image_dir(args.imgdir)
        log.info(f"Görüntü dizini: {len(frames)} frame")
    else:
        log.error("--video veya --imgdir gerekli (offline modda)")
        sys.exit(1)

    fx, fy, cx, cy = args.calib if args.calib else (457.0, 457.0, 320.0, 240.0)
    estimator = OnlineEstimator(fx=fx, fy=fy, cx=cx, cy=cy)

    # Ground truth varsa yükle
    gt_pos = None
    if args.gt and os.path.isfile(args.gt):
        gt_pos = _load_gt(args.gt)
        log.info(f"Ground truth: {len(gt_pos)} satır")

    results_log = []
    # İlk 450 frame sağlıklı, geri kalanı sağlıksız olarak simüle et
    health_cutoff = args.health_cutoff  # varsayılan 450

    for i, frame in enumerate(frames):
        health = 1 if i < health_cutoff else 0
        ref_pos = gt_pos[i] if (gt_pos is not None and i < len(gt_pos)) else None

        est_x, est_y, est_z = estimator.update(frame, ref_pos, health)

        results_log.append({
            "frame":  i,
            "health": health,
            "est_x":  est_x, "est_y": est_y, "est_z": est_z,
            "ref_x":  ref_pos[0] if ref_pos else None,
            "ref_y":  ref_pos[1] if ref_pos else None,
            "ref_z":  ref_pos[2] if ref_pos else None,
        })

        if i % 100 == 0:
            calib_str = "[CALIB]" if estimator.calibrated else f"[{estimator.calib_data_count}]"
            log.info(f"Frame {i:4d}  h={health}  est=({est_x:6.2f},{est_y:6.2f})  {calib_str}")

    # RMSE hesapla (health=0 bölümünde)
    if gt_pos is not None:
        _compute_rmse(results_log, gt_pos, health_cutoff)

    _save_results(results_log)
    log.info(f"Estimator state: {estimator.get_state()}")


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────

def _load_video_frames(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _load_image_dir(path: str) -> list[np.ndarray]:
    files = sorted(f for f in os.listdir(path) if f.endswith(".png") or f.endswith(".jpg"))
    frames = []
    for f in files:
        img = cv2.imread(os.path.join(path, f))
        if img is not None:
            frames.append(img)
    return frames


def _load_gt(path: str) -> list[tuple[float, float, float]]:
    """ground_truth_*.csv formatı: frame, x, y, z, dx, dy, dz"""
    gt = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt.append((float(row["dx"]), float(row["dy"]), float(row.get("dz", 0))))
    return gt


def _compute_rmse(
    results: list[dict],
    gt: list[tuple],
    health_cutoff: int,
) -> None:
    errs = []
    for i, r in enumerate(results):
        if r["health"] == 0 and i < len(gt):
            ex = r["est_x"] - gt[i][0]
            ey = r["est_y"] - gt[i][1]
            errs.append((ex, ey))
    if not errs:
        log.info("RMSE: health=0 frame yok veya GT eksik")
        return
    errs = np.array(errs)
    rmse_x  = float(np.sqrt(np.mean(errs[:, 0]**2)))
    rmse_y  = float(np.sqrt(np.mean(errs[:, 1]**2)))
    rmse_2d = float(np.sqrt(np.mean(np.sum(errs**2, axis=1))))
    log.info(f"\n{'='*45}")
    log.info(f"RMSE (health=0 bölümü, n={len(errs)})")
    log.info(f"  RMSE_x  : {rmse_x:.4f} m")
    log.info(f"  RMSE_y  : {rmse_y:.4f} m")
    log.info(f"  RMSE_2D : {rmse_2d:.4f} m")
    log.info(f"{'='*45}")


def _save_results(results: list[dict]) -> None:
    out = os.path.join(_REPO, "competition/results_log.csv")
    if not results:
        return
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    log.info(f"Sonuçlar kaydedildi: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="UAV Yarışma Pozisyon Kestirimi"
    )
    # Sunucu bağlantısı
    parser.add_argument("--url",   default="http://localhost:8080",
                        help="Sunucu base URL")
    parser.add_argument("--token", default=None,
                        help="API token/key")

    # Kalibrasyon
    parser.add_argument("--calib", nargs=4, type=float,
                        metavar=("FX","FY","CX","CY"),
                        default=None,
                        help="Kamera parametreleri (öncelikli, sunucuyu geçersiz kılar)")

    # Offline test
    parser.add_argument("--offline", action="store_true",
                        help="Sunucuya bağlanmadan offline test")
    parser.add_argument("--video",  default=None,
                        help="Offline mod: video dosyası (.mp4/.avi)")
    parser.add_argument("--imgdir", default=None,
                        help="Offline mod: PNG/JPG görüntü dizini")
    parser.add_argument("--gt",     default=None,
                        help="Offline mod: ground truth CSV (ground_truth_figure8_v3.csv)")
    parser.add_argument("--health-cutoff", type=int, default=450,
                        help="Offline mod: kaçıncı frameden sonra health=0 (varsayılan: 450)")

    args = parser.parse_args()

    if args.offline:
        run_offline(args)
    else:
        run_online(args)


if __name__ == "__main__":
    main()
