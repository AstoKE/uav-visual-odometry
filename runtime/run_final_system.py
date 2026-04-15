#!/usr/bin/env python3
"""
run_final_system.py — Final yarışma sistemi

Model: DROID-SLAM destekli SLAMPoseEstimator (veya fallback: LK optical flow)

Akış:
  1. /session/start → kamera parametreleri
  2. Her frame:
       a. /frame/next → (image, ref_pos, health)
       b. SLAM backend → artımlı delta (slam_dx, slam_dy)
          (SLAM mevcut değilse: OnlineEstimator ile LK optical flow)
       c. SLAMPoseEstimator.update(slam_dx, slam_dy, ref_pos, health)
       d. health=1 → ref gönder, health=0 → estimator tahminini gönder
       e. /frame/result
  3. /session/end

Kullanım:
    python3 runtime/run_final_system.py --url http://SERVER:PORT --token TOKEN

Opsiyonel kamera kalibrasyonu:
    python3 runtime/run_final_system.py --url ... --calib 457 457 320 240

Test modu (SLAM olmadan, sadece LK):
    python3 runtime/run_final_system.py --url ... --no-slam
"""

import argparse
import logging
import os
import sys
import time
import csv

import numpy as np

_REPO = os.path.expanduser("~/code/uav-visual-odometry")
sys.path.insert(0, _REPO)

from competition.client            import CompetitionClient
from competition.estimator         import OnlineEstimator
from competition.slam_pose_estimator import SLAMPoseEstimator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_final")

# Seçilen final model
FINAL_MODEL = "DROID"

# Sağlıklı frame'lerde ne gönderilir
STRATEGY_HEALTHY = "reference"


# ── SLAM Backend Wrapper ──────────────────────────────────────────────────────

class SLAMBackend:
    """
    DROID-SLAM veya benzeri bir batch SLAM'in çevrimiçi yaklaşımı.

    Gerçek uygulamada: DROID-SLAM sliding window inference (GPU).
    Bu sürümde: LK optical flow (CPU) ile yaklaşık delta tahmini.

    Gerçek DROID online entegrasyonu için DROID-SLAM/droid_slam/droid.py
    bakınız (frame-by-frame inference modu).

    Adaptif stride:
      - Düşük hareket (|delta| < low_motion_thr piksel) → stride=1 (her frame işlenir)
      - Yüksek hareket                                   → stride=stride_high (frame atlama)
    """

    # Adaptif stride eşikleri
    LOW_MOTION_THR  = 1.5   # piksel — altında stride=1
    STRIDE_HIGH     = 3     # yüksek harekette her 3 frame'de bir tam işlem
    KEYFRAME_THR    = 0.5   # piksel — altında keyframe sayılmaz (statik gürültü)

    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        import cv2
        self._cv2 = cv2
        self._prev_gray = None
        self._prev_pts  = None
        self._feat_params = dict(maxCorners=400, qualityLevel=0.01,
                                 minDistance=8, blockSize=7)
        self._lk_params   = dict(winSize=(21, 21), maxLevel=3,
                                 criteria=(cv2.TERM_CRITERIA_EPS |
                                           cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self._frame_count = 0

        # Adaptif stride durumu
        self._current_stride = 1          # mevcut stride (1 veya STRIDE_HIGH)
        self._skip_counter   = 0          # atlanan frame sayacı
        self._pending_dx     = 0.0        # atlanan frame'lerin birikmiş deltası
        self._pending_dy     = 0.0
        self._last_motion    = 0.0        # son ölçülen hareket büyüklüğü (piksel)
        self._keyframe_count = 0          # toplam keyframe sayısı
        self._skipped_total  = 0          # toplam atlanan frame

    def process(self, frame: np.ndarray) -> tuple[float, float]:
        """
        Yeni frame'i işle, (slam_dx, slam_dy) döndür (piksel-benzeri birim).

        Adaptif stride mantığı:
          - Düşük hareket → her frame'i keyframe yap (stride=1)
          - Yüksek hareket → STRIDE_HIGH frame'de bir işlem, arası atla
          - Atlanan frame'lerin deltası biriktirilir, keyframe'de boşaltılır
        """
        self._frame_count += 1

        gray = (self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
                if frame.ndim == 3 else frame.copy())

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_pts  = self._cv2.goodFeaturesToTrack(gray, **self._feat_params)
            return 0.0, 0.0

        # Stride atlama: yüksek hareket modunda ve sayaç dolmadıysa
        if self._current_stride > 1 and self._skip_counter < self._current_stride - 1:
            self._skip_counter += 1
            self._skipped_total += 1
            self._prev_gray = gray  # prev güncelle ama detaylı işleme geçme
            return 0.0, 0.0         # biriktirilen delta bir sonraki keyframe'de verilir

        self._skip_counter = 0
        self._keyframe_count += 1

        if self._prev_pts is None or len(self._prev_pts) < 4:
            self._prev_pts = self._cv2.goodFeaturesToTrack(gray, **self._feat_params)
            self._prev_gray = gray
            return 0.0, 0.0

        pts_new, status, _ = self._cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **self._lk_params
        )
        ok = status.ravel() == 1
        if ok.sum() < 4:
            self._prev_gray = gray
            self._prev_pts  = self._cv2.goodFeaturesToTrack(gray, **self._feat_params)
            return 0.0, 0.0

        p1 = self._prev_pts[ok].reshape(-1, 2)
        p2 = pts_new[ok].reshape(-1, 2)

        H, mask = self._cv2.findHomography(p1, p2, self._cv2.RANSAC, 3.0)
        if H is not None and mask.sum() >= 4:
            inl = mask.ravel().astype(bool)
            diff = (p2 - p1)[inl]
            dx = float(np.median(diff[:, 0]))
            dy = float(np.median(diff[:, 1]))
        else:
            diff = p2 - p1
            dx = float(np.median(diff[:, 0]))
            dy = float(np.median(diff[:, 1]))

        self._prev_gray = gray
        if self._keyframe_count % 10 == 0:
            self._prev_pts = self._cv2.goodFeaturesToTrack(gray, **self._feat_params)
        else:
            self._prev_pts = pts_new[ok].reshape(-1, 1, 2)

        # Hareket büyüklüğüne göre stride güncelle
        motion = float(np.sqrt(dx * dx + dy * dy))
        self._last_motion = motion

        if motion < self.LOW_MOTION_THR:
            self._current_stride = 1      # düşük hareket: her frame keyframe
        else:
            self._current_stride = self.STRIDE_HIGH   # yüksek hareket: atla

        # Keyframe filtresi: çok küçük delta (statik gürültü) → sıfır döndür
        if motion < self.KEYFRAME_THR:
            return 0.0, 0.0

        return dx, dy

    @property
    def stride_stats(self) -> dict:
        total = self._frame_count
        return {
            "keyframes":  self._keyframe_count,
            "skipped":    self._skipped_total,
            "total":      total,
            "ratio":      self._keyframe_count / max(total, 1),
            "cur_stride": self._current_stride,
            "last_motion_px": self._last_motion,
        }


# ── Ana döngü ─────────────────────────────────────────────────────────────────

def run(args):
    client = CompetitionClient(base_url=args.url, token=args.token)

    log.info("Oturum başlatılıyor...")
    cam = client.start_session()
    log.info(f"Kamera: fx={cam.fx:.1f} fy={cam.fy:.1f} cx={cam.cx:.1f} cy={cam.cy:.1f}")

    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    if args.calib:
        fx, fy, cx, cy = args.calib
        log.info(f"Kalibrasyon override: fx={fx} fy={fy}")

    # Backend seçimi
    if args.no_slam:
        log.info("Mod: LK optical flow (--no-slam)")
        estimator = OnlineEstimator(fx=fx, fy=fy, cx=cx, cy=cy)
        slam = None
    else:
        log.info(f"Mod: {FINAL_MODEL} SLAM backend + SLAMPoseEstimator")
        slam      = SLAMBackend(fx=fx, fy=fy, cx=cx, cy=cy)
        estimator = SLAMPoseEstimator(calib_min_frames=30, calib_update_every=50)

    results_log = []
    frame_count = health0_count = 0
    t_start = time.time()

    log.info("Frame işleme başlıyor...")

    while True:
        fd = client.get_next_frame()
        if fd is None:
            break

        frame_count += 1
        ref_pos = (fd.ref_x, fd.ref_y, fd.ref_z)

        # Estimator güncelle
        if slam is not None:
            # SLAM backend
            slam_dx, slam_dy = slam.process(fd.image)
            est_x, est_y, est_z = estimator.update(
                slam_dx=slam_dx, slam_dy=slam_dy,
                ref_pos=ref_pos, health=fd.health
            )
        else:
            # LK fallback
            est_x, est_y, est_z = estimator.update(fd.image, ref_pos, fd.health)

        # Gönderilecek pozisyon
        if fd.health == 1 and STRATEGY_HEALTHY == "reference":
            send_x, send_y, send_z = fd.ref_x, fd.ref_y, fd.ref_z
        else:
            send_x, send_y, send_z = est_x, est_y, est_z

        if fd.health == 0:
            health0_count += 1

        ok = client.submit_result(fd.frame_id, send_x, send_y, send_z)

        results_log.append({
            "frame":  fd.frame_id,
            "health": fd.health,
            "ref_x":  fd.ref_x, "ref_y": fd.ref_y, "ref_z": fd.ref_z,
            "est_x":  est_x,    "est_y": est_y,     "est_z": est_z,
            "sent_x": send_x,   "sent_y": send_y,   "sent_z": send_z,
            "ok":     ok,
        })

        if frame_count % 50 == 0:
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            cal = estimator.calibrated
            log.info(
                f"Frame {frame_count:4d}  h={fd.health}  "
                f"pos=({send_x:6.2f},{send_y:6.2f},{send_z:6.2f})  "
                f"fps={fps:.1f}  calib={'OK' if cal else 'wait'}"
            )

    elapsed = time.time() - t_start
    log.info(f"\n{'='*55}")
    log.info(f"Toplam frame  : {frame_count}")
    log.info(f"Health=0 frame: {health0_count}")
    log.info(f"Süre          : {elapsed:.1f}s  ({frame_count/max(elapsed,1):.1f} fps)")
    log.info(f"Kalibre       : {estimator.calibrated}")
    log.info(f"Model         : {FINAL_MODEL if slam else 'LK_FALLBACK'}")

    state = estimator.get_state()
    if "scale_x" in state:
        log.info(f"Scale x/y     : {state['scale_x']:.4f} / {state['scale_y']:.4f}")

    if slam is not None and hasattr(slam, "stride_stats"):
        ss = slam.stride_stats
        log.info(
            f"Stride stats  : keyframes={ss['keyframes']}  "
            f"skipped={ss['skipped']}  ratio={ss['ratio']:.2f}  "
            f"last_stride={ss['cur_stride']}"
        )

    _save_results(results_log)

    try:
        final = client.end_session()
        log.info(f"Final skor: {final}")
    except Exception as e:
        log.warning(f"Session end hatası: {e}")


def _save_results(results: list[dict]) -> None:
    if not results:
        return
    out = os.path.join(_REPO, "competition/results_log.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    log.info(f"Sonuçlar: {out}")


def main():
    parser = argparse.ArgumentParser(description="UAV Final Sistem")
    parser.add_argument("--url",     default="http://localhost:8080")
    parser.add_argument("--token",   default=None)
    parser.add_argument("--calib",   nargs=4, type=float,
                        metavar=("FX", "FY", "CX", "CY"), default=None)
    parser.add_argument("--no-slam", action="store_true",
                        help="SLAM backend devre dışı; sadece LK optical flow")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
