"""
droid_estimator.py — DROID-SLAM tabanlı online pozisyon tahmincisi

OnlineEstimator (ORB tabanlı) ile aynı update() arayüzü.
DROID-SLAM veya CUDA mevcut değilse OnlineEstimator'a otomatik fallback.

Mimari:
    Video Frame
      → DROID-SLAM track()       (keyframe bazlı derin öğrenme VO)
      → video.poses[:3] oku      (tx,ty,tz — DROID iç koordinat sistemi)
      → Sim3Aligner              (health=1 çiftleriyle DROID→dünya hizalama)
      → Dünya pozisyonu (metre)

Gereksinimler:
    conda env: droid_clean
    - torch >= 2.0 (CUDA)
    - lietorch
    - DROID-SLAM submodule: DROID-SLAM/droid_slam/
    - Model ağırlıkları: DROID-SLAM/checkpoints/droid.pth

Kullanım:
    from competition.droid_estimator import DROIDEstimator
    est = DROIDEstimator(fx=463, fy=462, cx=318, cy=186)
    x, y, z = est.update(frame, ref_pos, health)
    print(est.backend_name)  # "droid" veya "orb_fallback"
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_REPO        = os.path.expanduser("~/code/uav-visual-odometry")
_DROID_DIR   = os.path.join(_REPO, "DROID-SLAM")
_DROID_WEIGHTS = os.path.join(_DROID_DIR, "checkpoints", "droid.pth")


def _try_import_droid():
    """DROID-SLAM import dene. Başarısızsa (None, None) döndür."""
    try:
        import torch
        if not torch.cuda.is_available():
            log.info("[DROID] CUDA yok — ORB fallback")
            return None, None
        if not os.path.exists(_DROID_WEIGHTS):
            log.info(f"[DROID] Ağırlık bulunamadı: {_DROID_WEIGHTS} — ORB fallback")
            return None, None

        for p in [os.path.join(_DROID_DIR, "droid_slam"), _DROID_DIR]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from droid import Droid
        return torch, Droid
    except ImportError as e:
        log.info(f"[DROID] Import başarısız ({e}) — ORB fallback")
        return None, None


class DROIDEstimator:
    """
    DROID-SLAM tabanlı online pozisyon tahmincisi.

    health=1 → referans pozisyonu direkt döndür (optimal strateji, sıfır hata).
              Aynı zamanda DROID'i besle + Sim3 kalibrasyon çifti ekle.
    health=0 → DROID'den okunan son pozu Sim3 ile dünya koordinatına çevir.

    health 0→1 geçişi: T_world sıfırla + Sim3 yeniden kalibre et (OnlineEstimator ile aynı).
    """

    def __init__(
        self,
        fx: float = 463.2,
        fy: float = 462.4,
        cx: float = 318.0,
        cy: float = 186.3,
        dist_coeffs: Optional[list] = None,
        altitude_m: Optional[float] = None,
        image_size: Optional[list] = None,
        buffer_size: int = 512,
        force_orb: bool = False,
    ):
        self._fx, self._fy = fx, fy
        self._cx, self._cy = cx, cy
        self._frame_count = 0
        self._use_droid   = False
        self._droid       = None
        self._torch       = None

        if not force_orb:
            torch_mod, Droid = _try_import_droid()
            if torch_mod is not None:
                self._torch     = torch_mod
                self._use_droid = True
                self._init_droid(Droid, image_size or [240, 320], buffer_size,
                                 fx, fy, cx, cy)

        if not self._use_droid:
            self._init_orb_fallback(fx, fy, cx, cy, dist_coeffs, altitude_m)

    # ── Başlatıcılar ──────────────────────────────────────────────────────────

    def _init_droid(self, Droid, image_size, buffer_size, fx, fy, cx, cy):
        import argparse
        args = argparse.Namespace(
            weights          = _DROID_WEIGHTS,
            image_size       = image_size,
            buffer           = buffer_size,
            stereo           = False,
            disable_vis      = True,
            filter_thresh    = 2.4,
            beta             = 0.3,
            warmup           = 8,
            keyframe_thresh  = 4.0,
            frontend_thresh  = 16.0,
            frontend_window  = 25,
            frontend_radius  = 2,
            frontend_nms     = 1,
            backend_thresh   = 22.5,
            backend_radius   = 2,
            backend_nms      = 3,
            upsample         = False,
        )
        try:
            self._droid       = Droid(args)
            self._intrinsics  = self._torch.as_tensor(
                [fx, fy, cx, cy], dtype=self._torch.float
            ).cuda()
            self._image_size  = image_size

            # Sim3 hizalama (DROID iç koordinat → dünya/metre)
            from competition.sim3_aligner import Sim3Aligner
            self._sim3        = Sim3Aligner(min_pairs=5, update_every=10, window_size=80)
            self._calib_droid: list = []   # (droid_pos_3d, ref_pos_3d) çiftleri
            self._calib_ref:   list = []

            # Durum
            self._last_droid_pos = np.zeros(3, dtype=np.float64)
            self._last_world_pos = np.zeros(3, dtype=np.float64)
            self._prev_health    = 1

            log.info(f"[DROID] Başlatıldı — image_size={image_size} buffer={buffer_size}")
        except Exception as e:
            log.warning(f"[DROID] Başlatma hatası ({e}) — ORB fallback")
            self._use_droid = False
            self._init_orb_fallback(self._fx, self._fy, self._cx, self._cy, None, None)

    def _init_orb_fallback(self, fx, fy, cx, cy, dist_coeffs, altitude_m):
        from competition.estimator import OnlineEstimator
        dist_arr = (np.array(dist_coeffs, dtype=np.float64)
                    if dist_coeffs is not None else None)
        self._orb = OnlineEstimator(
            fx=fx, fy=fy, cx=cx, cy=cy,
            dist_coeffs=dist_arr,
            altitude_m=altitude_m,
            n_features=1000,
            sim3_min_pairs=5,
            sim3_update_every=20,
            max_jump_m=8.0,
        )
        log.info("[DROID] ORB fallback başlatıldı")

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        return "droid" if self._use_droid else "orb_fallback"

    def update(self, frame: np.ndarray, ref_pos: Optional[tuple], health: int) -> tuple:
        self._frame_count += 1
        if self._use_droid:
            return self._update_droid(frame, ref_pos, health)
        return self._orb.update(frame, ref_pos, health)

    def get_state(self) -> dict:
        if not self._use_droid:
            s = self._orb.get_state()
            s["backend"] = "orb_fallback"
            return s
        return {
            "frame":           self._frame_count,
            "backend":         "droid",
            "calibrated":      self._sim3.calibrated,
            "sim3_n_pairs":    self._sim3.n_pairs,
            "sim3_scale":      round(self._sim3.scale, 3),
            "sim3_rmse":       round(self._sim3.rmse_calib, 4) if not __import__('math').isnan(self._sim3.rmse_calib) else float('nan'),
            "world_x":         float(self._last_world_pos[0]),
            "world_y":         float(self._last_world_pos[1]),
            "ema_x":           float(self._last_world_pos[0]),
            "ema_y":           float(self._last_world_pos[1]),
            "confidence":      1.0,
            "drift_rejected":  0,
            "last_inliers":    0,
            "jump_threshold":  0.0,
            "median_vel_m":    0.0,
            "health0_streak":  0,
            "calib_pairs_raw": len(self._calib_droid),
        }

    # ── DROID iç implementasyonu ──────────────────────────────────────────────

    def _update_droid(self, frame: np.ndarray, ref_pos, health: int) -> tuple:
        # Health 0→1 geçişi: Sim3 sıfırla (OnlineEstimator ile aynı strateji)
        if health == 1 and self._prev_health == 0:
            from competition.sim3_aligner import Sim3Aligner
            self._sim3       = Sim3Aligner(min_pairs=5, update_every=10, window_size=80)
            self._calib_droid.clear()
            self._calib_ref.clear()
            log.debug(f"[DROID] health 0→1 frame={self._frame_count}: Sim3 sıfırlandı")
        self._prev_health = health

        # Frame'i DROID'e besle
        self._feed_to_droid(frame)

        # Mevcut DROID pozunu oku
        droid_pos = self._read_droid_pos()
        if droid_pos is not None:
            self._last_droid_pos = droid_pos

        if health == 1 and ref_pos is not None:
            # Kalibrasyon çifti ekle
            self._sim3.add(self._last_droid_pos,
                           np.array(ref_pos[:3], dtype=np.float64))
            # Optimal strateji: referansı direkt döndür
            self._last_world_pos = np.array(ref_pos[:3], dtype=np.float64)
            return float(ref_pos[0]), float(ref_pos[1]), float(ref_pos[2])

        # health=0: Sim3 ile dünya pozisyonu
        if self._sim3.calibrated:
            wp = self._sim3.apply(self._last_droid_pos)
            self._last_world_pos = wp
        # Sim3 henüz kalibrasyon olmadıysa son bilinen pozda kal
        return (float(self._last_world_pos[0]),
                float(self._last_world_pos[1]),
                float(self._last_world_pos[2]))

    def _frame_to_tensor(self, frame: np.ndarray):
        """BGR numpy → DROID: [1, 3, H, W] float CUDA tensör."""
        h, w = self._image_size
        img  = cv2.resize(frame, (w, h))
        return (self._torch.as_tensor(img)
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                .cuda())

    def _feed_to_droid(self, frame: np.ndarray) -> None:
        try:
            img_t = self._frame_to_tensor(frame)
            self._droid.track(self._frame_count, img_t,
                              intrinsics=self._intrinsics)
        except Exception as e:
            log.debug(f"[DROID] track hatası frame={self._frame_count}: {e}")

    def _read_droid_pos(self) -> Optional[np.ndarray]:
        """
        DROID video buffer'dan son keyframe'in translasyon vektörünü oku.
        Poses formatı: [tx, ty, tz, qx, qy, qz, qw] (identity = [0,0,0,0,0,0,1])
        """
        try:
            counter = self._droid.video.counter.value
            if counter < 1:
                return None
            idx  = min(counter - 1, self._droid.video.poses.shape[0] - 1)
            pose = self._droid.video.poses[idx].cpu().numpy()  # (7,)
            return pose[:3].astype(np.float64)                 # (tx, ty, tz)
        except Exception as e:
            log.debug(f"[DROID] pose okuma hatası: {e}")
            return None
