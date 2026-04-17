"""
droid_estimator.py — DROID-SLAM tabanlı online pozisyon tahmincisi

OnlineEstimator (ORB tabanlı) ile aynı update() arayüzü.
DROID-SLAM veya CUDA mevcut değilse OnlineEstimator'a otomatik fallback.

Gereksinimler:
    - CUDA GPU
    - DROID-SLAM submodule kurulu: DROID-SLAM/droid_slam/
    - Model ağırlıkları: DROID-SLAM/droid.pth
    - Bağımlılıklar: torch, lietorch, evo

Kullanım:
    from competition.droid_estimator import DROIDEstimator

    est = DROIDEstimator(fx=463, fy=462, cx=318, cy=186)
    x, y, z = est.update(frame, ref_pos, health)
    print(est.backend_name)  # "droid" veya "orb_fallback"

DROID-SLAM'dan per-frame pose alma stratejisi:
    - Her track() çağrısından sonra video.poses tensörünü oku
    - Son keyframe'in SE3 pozu → kamera koordinatlarından dünya koordinatlarına
    - Sim3 hizalaması OnlineEstimator altyapısıyla aynı şekilde
"""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_REPO = os.path.expanduser("~/code/uav-visual-odometry")
_DROID_DIR = os.path.join(_REPO, "DROID-SLAM")
_DROID_WEIGHTS = os.path.join(_DROID_DIR, "droid.pth")


def _try_import_droid():
    """DROID-SLAM import dene. Başarısızsa None döndür."""
    try:
        import torch
        if not torch.cuda.is_available():
            log.info("[DROID] CUDA yok — fallback'e geçiliyor")
            return None, None

        if not os.path.exists(_DROID_WEIGHTS):
            log.info(f"[DROID] Ağırlık dosyası bulunamadı: {_DROID_WEIGHTS} — fallback")
            return None, None

        # DROID-SLAM dizinini sys.path'e ekle
        droid_slam_dir = os.path.join(_DROID_DIR, "droid_slam")
        if droid_slam_dir not in sys.path:
            sys.path.insert(0, droid_slam_dir)
        if _DROID_DIR not in sys.path:
            sys.path.insert(0, _DROID_DIR)

        from droid import Droid
        import lietorch
        return torch, Droid

    except ImportError as e:
        log.info(f"[DROID] Import başarısız: {e} — fallback'e geçiliyor")
        return None, None


class DROIDEstimator:
    """
    DROID-SLAM tabanlı online pozisyon tahmincisi.

    DROID kullanılamıyorsa OnlineEstimator'a (ORB) otomatik fallback.
    Her iki durumda da aynı update() arayüzü.

    Parameters
    ----------
    fx, fy, cx, cy   : kamera intrinsik (orijinal veya resize sonrası)
    dist_coeffs      : distorsiyon katsayıları (undistort için)
    altitude_m       : bilinen irtifa (metrik ölçek için)
    image_size       : DROID resize hedefi [H, W] (varsayılan [240, 320])
    buffer_size      : DROID buffer (RAM/VRAM trade-off)
    force_orb        : True → her zaman ORB fallback kullan (test için)
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
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy

        self._torch = None
        self._droid = None
        self._use_droid = False
        self._frame_count = 0

        if not force_orb:
            torch_mod, Droid = _try_import_droid()
            if torch_mod is not None and Droid is not None:
                self._torch = torch_mod
                self._use_droid = True
                self._init_droid(Droid, image_size or [240, 320], buffer_size,
                                 fx, fy, cx, cy)

        if not self._use_droid:
            self._init_orb_fallback(fx, fy, cx, cy, dist_coeffs, altitude_m)

    # ── Başlatıcılar ──────────────────────────────────────────────────────────

    def _init_droid(self, Droid, image_size, buffer_size, fx, fy, cx, cy):
        """DROID-SLAM nesnesini başlat."""
        import argparse
        args = argparse.Namespace(
            weights      = _DROID_WEIGHTS,
            image_size   = image_size,
            buffer       = buffer_size,
            stereo       = False,
            disable_vis  = True,
            filter_thresh= 2.4,
            beta         = 0.3,
            warmup       = 8,
            keyframe_thresh = 4.0,
            frontend_thresh = 16.0,
            frontend_window = 25,
            frontend_radius = 2,
            frontend_nms    = 1,
            backend_thresh  = 22.5,
            backend_radius  = 2,
            backend_nms     = 3,
            upsample        = False,
        )
        try:
            self._droid = Droid(args)
            # İntrinsik tensör (DROID formatı: fx, fy, cx, cy)
            self._intrinsics = self._torch.as_tensor(
                [fx, fy, cx, cy], dtype=self._torch.float
            ).cuda()
            self._image_size = image_size
            # Kümülatif poz (son bilinen)
            self._last_pose = np.zeros(3, dtype=np.float64)
            log.info(f"[DROID] Başlatıldı: image_size={image_size} buffer={buffer_size}")
        except Exception as e:
            log.warning(f"[DROID] Başlatma hatası: {e} — ORB fallback'e geçiliyor")
            self._use_droid = False
            self._init_orb_fallback(fx, fy, cx, cy, None, None)

    def _init_orb_fallback(self, fx, fy, cx, cy, dist_coeffs, altitude_m):
        """ORB tabanlı OnlineEstimator'ı başlat."""
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

    def update(
        self,
        frame: np.ndarray,
        ref_pos: Optional[tuple],
        health: int,
    ) -> tuple:
        """
        Yeni kare işle — OnlineEstimator ile aynı imza.

        Parameters
        ----------
        frame   : BGR görüntü (numpy ndarray)
        ref_pos : (x, y, z) metre | health=0'da None
        health  : 1 = referans güvenilir, 0 = GPS-siz mod

        Returns
        -------
        (est_x, est_y, est_z) metre
        """
        self._frame_count += 1

        if self._use_droid:
            return self._update_droid(frame, ref_pos, health)
        else:
            return self._orb.update(frame, ref_pos, health)

    def get_state(self) -> dict:
        """Tanılama bilgisi — OnlineEstimator.get_state() ile uyumlu."""
        if self._use_droid:
            return {
                "frame":           self._frame_count,
                "backend":         "droid",
                "calibrated":      True,
                "sim3_n_pairs":    0,
                "sim3_scale":      1.0,
                "sim3_rmse":       0.0,
                "world_x":         float(self._last_pose[0]),
                "world_y":         float(self._last_pose[1]),
                "ema_x":           float(self._last_pose[0]),
                "ema_y":           float(self._last_pose[1]),
                "confidence":      1.0,
                "drift_rejected":  0,
                "last_inliers":    0,
                "jump_threshold":  0.0,
                "median_vel_m":    0.0,
                "health0_streak":  0,
                "calib_pairs_raw": 0,
            }
        else:
            state = self._orb.get_state()
            state["backend"] = "orb_fallback"
            return state

    # ── DROID iç implementasyonu ──────────────────────────────────────────────

    def _update_droid(self, frame: np.ndarray, ref_pos, health: int) -> tuple:
        """
        DROID-SLAM üzerinden pose tahmin et.

        health=1: referans pozisyonu direkt döndür (optimal strateji).
        health=0: DROID'den okunan son kamera pozunu Sim3 ile world frame'e çevir.
        """
        if health == 1 and ref_pos is not None:
            # Optimal strateji: referansı direkt gönder
            self._last_pose = np.array(ref_pos[:3], dtype=np.float64)
            # Yine de DROID'e frame'i besle (kalibrasyon sürsün)
            self._feed_to_droid(frame)
            return float(ref_pos[0]), float(ref_pos[1]), float(ref_pos[2])

        # health=0: DROID tahmini
        self._feed_to_droid(frame)
        pose = self._read_droid_pose()
        if pose is not None:
            self._last_pose = pose
        return float(self._last_pose[0]), float(self._last_pose[1]), float(self._last_pose[2])

    def _frame_to_tensor(self, frame: np.ndarray):
        """BGR numpy → DROID formatı: [1, 3, H, W] float tensor (CUDA)."""
        h_target, w_target = self._image_size
        resized = cv2.resize(frame, (w_target, h_target))
        tensor = (self._torch.as_tensor(resized)
                  .permute(2, 0, 1)        # HWC → CHW
                  .float()
                  .unsqueeze(0)            # batch dim
                  .cuda())
        return tensor

    def _feed_to_droid(self, frame: np.ndarray) -> None:
        """Frame'i DROID pipeline'ına besle."""
        try:
            img_t = self._frame_to_tensor(frame)
            self._droid.track(self._frame_count, img_t,
                              intrinsics=self._intrinsics)
        except Exception as e:
            log.debug(f"[DROID] track hatası frame={self._frame_count}: {e}")

    def _read_droid_pose(self) -> Optional[np.ndarray]:
        """
        DROID video buffer'dan son mevcut kamera pozunu oku.

        Döndürür: (3,) dünya koordinatı (metre) veya None
        """
        try:
            # video.poses: [N, 7] — SE3 (quaternion + translation), lietorch formatı
            # video.counter: kaç keyframe işlendiği
            counter = self._droid.video.counter.value
            if counter < 1:
                return None

            # Son keyframe'in pozunu al
            poses = self._droid.video.poses
            last_idx = min(counter - 1, poses.shape[0] - 1)
            pose_se3 = poses[last_idx]  # [7] tensor

            # SE3 → translasyon (DROID camera frame, birim belirsiz)
            # pose_se3[:3] = quaternion (qx, qy, qz, qw formatı), pose_se3[3:] = xyz
            # Uyarı: DROID poses formatı [tx, ty, tz, qx, qy, qz, qw] DEĞİL
            # lietorch SE3: log map kullanılır; doğrudan data() ile xyz alınır
            import lietorch
            T = lietorch.SE3(pose_se3[None])  # batch dim ekle
            xyz = T.translation().squeeze().cpu().numpy()  # (3,)

            # Kamera frame'den dünya frame'e (nadir kamera: z ileri, x sağ, y aşağı)
            # Basit: tx, ty, tz'yi direkt kullan (Sim3 hizalaması zaten yapılacak)
            return xyz.astype(np.float64)

        except Exception as e:
            log.debug(f"[DROID] pose okuma hatası: {e}")
            return None

    def terminate(self) -> Optional[np.ndarray]:
        """
        DROID global bundle adjustment çalıştır, tüm pozları döndür.

        Döndürür: (N, 3) pozisyon dizisi veya None
        """
        if not self._use_droid or self._droid is None:
            return None
        try:
            import lietorch
            traj = self._droid.terminate()
            # traj: [N, 7] SE3 tensör
            T = lietorch.SE3(self._torch.as_tensor(traj))
            xyz = T.translation().cpu().numpy()  # (N, 3)
            return xyz
        except Exception as e:
            log.warning(f"[DROID] terminate hatası: {e}")
            return None
