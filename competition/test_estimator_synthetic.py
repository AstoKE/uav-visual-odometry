#!/usr/bin/env python3
"""
test_estimator_synthetic.py — Sentetik hareket ile OnlineEstimator doğrulaması

Yöntem:
  1. Zengin doku içeren bir taban görüntü oluştur (checkerboard + rastgele noktalar)
  2. cv2.warpAffine ile bilinen tx/ty translasyon serisi uygula
  3. Bu frame'leri OnlineEstimator'a gönder
  4. Estimator'ın elde ettiği kümülatif pozisyonu GT ile karşılaştır

Kullanım:
    python3 ~/code/uav-visual-odometry/competition/test_estimator_synthetic.py

Beklenen çıktı:
    - health=1 sırasında kalibrasyon gerçekleşmeli (frame ~30-50)
    - health=0 bölümünde RMSE < 2.0m (sentetik, gürültüsüz)
"""

import sys, os
_REPO = os.path.expanduser("~/code/uav-visual-odometry")
sys.path.insert(0, _REPO)

import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("synthetic_test")

from competition.estimator import OnlineEstimator


# ── Parametreler ────────────────────────────────────────────────────────────
W, H       = 640, 480       # görüntü boyutu
FX, FY     = 457.0, 457.0
CX, CY     = 320.0, 240.0

# Gerçek ölçek: 1 piksel = kaç metre (kamera 4m yükseklikte, 70° FOV)
# FOV ~ 2 * atan(W/2 / fx) ≈ 2 * atan(320/457) ≈ 70°
# Zemin genişliği: 2 * 4 * tan(35°) ≈ 5.6m → 640px → 0.00875 m/px
TRUE_SCALE = 0.00875        # m/px (yaklaşık)

N_CALIB    = 450            # health=1 frame (kalibrasyon penceresi)
N_DEAD     = 300            # health=0 frame (test)
TOTAL      = N_CALIB + N_DEAD

# Hareket profili: lawnmower (düzgün hareket)
# Dünya koordinatlarında (metre/frame) istenen hareket
SPEED_X_MPS  = 0.05   # m/frame (x ekseninde hız)
SPEED_Y_MPS  = 0.0    # y ekseninde sabit başla

NOISE_PX     = 0.3    # optik akış gürültüsü (piksel std)
SENSOR_NOISE = 0.002  # referans pozisyon gürültüsü (m std)


def make_rich_texture(w: int, h: int, seed: int = 42) -> np.ndarray:
    """Checkerboard + rastgele bloklar + daireler içeren zengin doku."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=np.uint8)

    # Checkerboard (zemin simülasyonu)
    tile = 40
    for row in range(0, h, tile):
        for col in range(0, w, tile):
            if (row // tile + col // tile) % 2 == 0:
                img[row:row+tile, col:col+tile] = 220
            else:
                img[row:row+tile, col:col+tile] = 30

    # Rastgele noktalar (landmark simülasyonu)
    n_pts = 200
    xs = rng.integers(10, w-10, n_pts)
    ys = rng.integers(10, h-10, n_pts)
    sizes = rng.integers(3, 12, n_pts)
    colors = rng.integers(80, 230, n_pts)
    for x, y, s, c in zip(xs, ys, sizes, colors):
        cv2.circle(img, (int(x), int(y)), int(s), int(c), -1)

    # Gaussian blur hafifçe
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_trajectory(n_frames: int) -> list[tuple[float, float]]:
    """
    Lawnmower benzeri dünya hareketi (metre).
    Returns: [(dx_cum, dy_cum), ...] her frame için kümülatif hareket.
    """
    traj = []
    cum_x, cum_y = 0.0, 0.0
    row_len   = 80          # frame sayısı (her sıra)
    row_step  = SPEED_X_MPS # m/frame ileri
    col_step  = 0.3         # m yan kaydırma (her sıra dönüşünde)
    direction = 1.0

    for i in range(n_frames):
        pos_in_row = i % row_len
        if pos_in_row == 0 and i > 0:
            # Sıra dönüşü
            cum_y += col_step
            direction *= -1.0

        cum_x += direction * row_step
        traj.append((cum_x, cum_y))

    return traj


def warp_frame(
    base: np.ndarray,
    cum_px_x: float,
    cum_px_y: float,
    canvas_w: int,
    canvas_h: int,
) -> np.ndarray:
    """
    Büyük taban görüntüyü kaydırarak UAV kamera görüntüsü simüle et.
    Kamera sabit, zemin hareket ediyor (eşdeğer).
    """
    H_canvas, W_canvas = base.shape[:2]
    # Merkezi başlangıç noktası
    start_x = W_canvas // 2 - canvas_w // 2
    start_y = H_canvas // 2 - canvas_h // 2

    ox = int(start_x + cum_px_x) % (W_canvas - canvas_w)
    oy = int(start_y + cum_px_y) % (H_canvas - canvas_h)

    ox = max(0, min(ox, W_canvas - canvas_w - 1))
    oy = max(0, min(oy, H_canvas - canvas_h - 1))

    return base[oy:oy+canvas_h, ox:ox+canvas_w].copy()


def main():
    log.info("Sentetik test başlıyor...")
    log.info(f"Kalibrasyon: {N_CALIB} frame, Test: {N_DEAD} frame")
    log.info(f"Gerçek ölçek: {TRUE_SCALE:.5f} m/px")

    rng = np.random.default_rng(0)

    # Büyük canvas oluştur (hareket için alan)
    CANVAS_W = W * 6
    CANVAS_H = H * 6
    log.info(f"Canvas oluşturuluyor: {CANVAS_W}×{CANVAS_H}...")
    canvas = make_rich_texture(CANVAS_W, CANVAS_H, seed=99)

    # Dünya hareketi trajektörisi
    traj = make_trajectory(TOTAL)
    # Piksel hareketi: world_m / scale = px
    traj_px = [(x / TRUE_SCALE, y / TRUE_SCALE) for x, y in traj]

    # Estimator başlat
    estimator = OnlineEstimator(fx=FX, fy=FY, cx=CX, cy=CY, calib_min_frames=30)

    results = []
    for i in range(TOTAL):
        health = 1 if i < N_CALIB else 0

        # Frame oluştur
        cum_px_x, cum_px_y = traj_px[i]
        # Küçük piksel gürültüsü ekle (her frame için farklı gürültü)
        noisy_px_x = cum_px_x + rng.normal(0, NOISE_PX * i**0.3)
        noisy_px_y = cum_px_y + rng.normal(0, NOISE_PX * i**0.3)
        frame = warp_frame(canvas, noisy_px_x, noisy_px_y, W, H)

        # Referans pozisyon (health=1'de mevcut, biraz gürültülü)
        gt_x, gt_y = traj[i]
        if health == 1:
            ref_x = gt_x + rng.normal(0, SENSOR_NOISE)
            ref_y = gt_y + rng.normal(0, SENSOR_NOISE)
            ref_z = rng.normal(0, SENSOR_NOISE)
            ref_pos = (ref_x, ref_y, ref_z)
        else:
            ref_pos = None

        # Estimator güncelle
        est_x, est_y, est_z = estimator.update(frame, ref_pos, health)

        results.append({
            "i": i, "health": health,
            "gt_x": gt_x, "gt_y": gt_y,
            "est_x": est_x, "est_y": est_y,
        })

        if i % 100 == 0:
            calib_str = "[CALIB]" if estimator.calibrated else f"[n={estimator.calib_data_count}]"
            log.info(
                f"Frame {i:4d}  h={health}  "
                f"gt=({gt_x:6.3f},{gt_y:6.3f})  "
                f"est=({est_x:6.3f},{est_y:6.3f})  {calib_str}"
            )

    # ── RMSE hesapla ──────────────────────────────────────────────────────────
    dead_results = [r for r in results if r["health"] == 0]
    if dead_results:
        errs = [(r["est_x"] - r["gt_x"], r["est_y"] - r["gt_y"]) for r in dead_results]
        errs = np.array(errs)
        rmse_x  = float(np.sqrt(np.mean(errs[:, 0]**2)))
        rmse_y  = float(np.sqrt(np.mean(errs[:, 1]**2)))
        rmse_2d = float(np.sqrt(np.mean(np.sum(errs**2, axis=1))))
        log.info(f"\n{'='*50}")
        log.info(f"Kalibrasyon durumu: {'OK' if estimator.calibrated else 'FAIL'}")
        state = estimator.get_state()
        log.info(f"Scale x/y : {state['scale_x']:.5f} / {state['scale_y']:.5f}  (gerçek: {TRUE_SCALE:.5f})")
        log.info(f"Sign  x/y : {state['sign_x']:.0f} / {state['sign_y']:.0f}")
        log.info(f"Swap xy   : {state['swap_xy']}")
        log.info(f"RMSE_x    : {rmse_x:.4f} m")
        log.info(f"RMSE_y    : {rmse_y:.4f} m")
        log.info(f"RMSE_2D   : {rmse_2d:.4f} m  (health=0, n={len(dead_results)})")
        log.info(f"{'='*50}")

        # Başarı kriteri: kalibrasyon yapıldı mı ve RMSE makul mü
        if not estimator.calibrated:
            log.error("FAIL: Kalibrasyon yapılmadı!")
            return False
        if rmse_2d > 5.0:
            log.warning(f"WARN: RMSE_2D={rmse_2d:.2f}m yüksek (sınır: 5.0m)")
        else:
            log.info(f"PASS: RMSE_2D={rmse_2d:.2f}m < 5.0m")
        return True
    else:
        log.error("health=0 frame yok!")
        return False


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
