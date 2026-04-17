"""
slam_pose_estimator.py — SLAM trajectory-backed pozisyon kestirici

OnlineEstimator ile aynı 2×2 kalibrasyonu kullanır, fark:
  - Optical flow yerine önceden hesaplanmış SLAM artımlı deltaları alır
  - Offline dataset testi ve SLAM karşılaştırması için tasarlanmıştır

Z ekseni stratejisi (yarışma §9.2 — MAE_3D):
  - health=1: Z doğrudan ref_pos[2]'den alınır (tam doğru)
  - health=0: son bilinen Z değeri korunur (irtifa sabit varsayımı)
              UAV'lar tipik olarak GPS-denied sırasında irtifa tutarlar.

Kullanım:
    from competition.slam_pose_estimator import SLAMPoseEstimator
    est = SLAMPoseEstimator()
    for i, (slam_dx, slam_dy, ref_pos, health) in enumerate(frames):
        wx, wy, wz = est.update(slam_dx, slam_dy, ref_pos, health)
"""

from __future__ import annotations

import numpy as np
import logging
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


class SLAMPoseEstimator:
    """
    SLAM artımlı pozisyon deltaları üzerinden kalibre edilmiş dünya konumu kestirici.

    Parameters
    ----------
    calib_min_frames : int
        Kalibrasyon için minimum health=1 delta çifti sayısı
    calib_update_every : int
        Periyodik yeniden kalibrasyon aralığı (health=1 frame'lerde)
    """

    def __init__(
        self,
        calib_min_frames: int = 30,
        calib_update_every: int = 50,
    ):
        self.calib_min_frames = calib_min_frames
        self.calib_update_every = calib_update_every

        # Kümülatif SLAM koordinatları (ham, ölçeksiz)
        self._cum_sx = 0.0
        self._cum_sy = 0.0

        # Kalibrasyon durumu
        self._calibrated = False
        self._M = np.eye(2, dtype=np.float64)
        self._calib_data: List[tuple] = []  # (d_sx, d_sy, d_rx, d_ry, d_rz)

        # Önceki değerler
        self._prev_ref: Optional[tuple] = None
        self._prev_cum = (0.0, 0.0)
        self._ref_origin: Optional[tuple] = None

        # Z anchor: health=0'da son bilinen Z korunur (irtifa sabit varsayımı)
        self._last_known_z: float = 0.0

        # Dünya pozisyonu
        self._world_x = 0.0
        self._world_y = 0.0
        self._world_z = 0.0

        self._frame_count = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        slam_dx: float,
        slam_dy: float,
        ref_pos: Optional[Tuple[float, float, float]],
        health: int,
    ) -> Tuple[float, float, float]:
        """
        Yeni bir frame için SLAM delta ile güncelle.

        Parameters
        ----------
        slam_dx, slam_dy : float
            Bu frame ile önceki frame arasındaki ham SLAM artımlı pozisyon (ölçeksiz birim)
        ref_pos : (rx, ry, rz) metre | None
            health=1'de gelen referans; health=0'da None
        health : int
            1 = referans güvenilir, 0 = kendi tahmini

        Returns
        -------
        (wx, wy, wz) metre, ilk frame'e göre yer değiştirme
        """
        self._frame_count += 1

        # İlk frame
        if self._frame_count == 1:
            if health == 1 and ref_pos is not None:
                self._ref_origin = ref_pos
                self._prev_ref = ref_pos
            return (0.0, 0.0, 0.0)

        # Kümülatif SLAM güncelle
        self._cum_sx += slam_dx
        self._cum_sy += slam_dy

        # health=1 → kalibrasyon verisi topla
        if health == 1 and ref_pos is not None:
            if self._ref_origin is None:
                self._ref_origin = ref_pos
                self._prev_ref = ref_pos
                self._prev_cum = (self._cum_sx, self._cum_sy)
            elif self._prev_ref is not None:
                d_sx = self._cum_sx - self._prev_cum[0]
                d_sy = self._cum_sy - self._prev_cum[1]
                d_rx = ref_pos[0] - self._prev_ref[0]
                d_ry = ref_pos[1] - self._prev_ref[1]
                d_rz = ref_pos[2] - self._prev_ref[2]
                self._calib_data.append((d_sx, d_sy, d_rx, d_ry, d_rz))
                self._prev_ref = ref_pos
                self._prev_cum = (self._cum_sx, self._cum_sy)

            n = len(self._calib_data)
            if n >= self.calib_min_frames:
                if not self._calibrated or (n % self.calib_update_every == 0):
                    self._run_calibration()

        # Z anchor güncelle: health=1'de gerçek Z'yi kaydet
        if health == 1 and ref_pos is not None and self._ref_origin is not None:
            self._last_known_z = ref_pos[2] - self._ref_origin[2]

        # Dünya pozisyonu (X, Y)
        if self._calibrated:
            v = self._M @ np.array([self._cum_sx, self._cum_sy])
            self._world_x, self._world_y = float(v[0]), float(v[1])
            # Z: kalibrasyon sonrası da son bilinen Z'yi kullan (irtifa sabit)
            self._world_z = self._last_known_z
        elif health == 1 and ref_pos is not None and self._ref_origin is not None:
            self._world_x = ref_pos[0] - self._ref_origin[0]
            self._world_y = ref_pos[1] - self._ref_origin[1]
            self._world_z = ref_pos[2] - self._ref_origin[2]
        # health=0 ve henüz kalibrasyon yoksa: son bilinen Z'yi koru
        else:
            self._world_z = self._last_known_z

        return (self._world_x, self._world_y, self._world_z)

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def calib_data_count(self) -> int:
        return len(self._calib_data)

    def get_state(self) -> dict:
        M = self._M
        return {
            "frame":      self._frame_count,
            "calibrated": self._calibrated,
            "calib_n":    len(self._calib_data),
            "cum_sx":     self._cum_sx,
            "cum_sy":     self._cum_sy,
            "world_x":    self._world_x,
            "world_y":    self._world_y,
            "M":          M.tolist(),
            "scale_x":    float(abs(M[0, 0])),
            "scale_y":    float(abs(M[1, 1])),
        }

    # ── Kalibrasyon ───────────────────────────────────────────────────────────

    def _run_calibration(self) -> None:
        data = np.array(self._calib_data, dtype=np.float64)
        cix = data[:, 0]; ciy = data[:, 1]
        rx  = data[:, 2]; ry  = data[:, 3]; rz = data[:, 4]

        motion_mag = np.sqrt(cix**2 + ciy**2)
        ref_motion = np.sqrt(rx**2 + ry**2)

        img_thresh = max(float(np.percentile(motion_mag, 20)), 1e-6)
        mask = (motion_mag > img_thresh) & (ref_motion > 0.005)
        if mask.sum() < 8:
            mask = motion_mag > 1e-7
        if mask.sum() < 5:
            log.warning(f"Kalibrasyon: yeterli delta yok ({mask.sum()})")
            return

        A = np.column_stack([cix[mask], ciy[mask]])
        try:
            cx, _, _, _ = np.linalg.lstsq(A, rx[mask], rcond=None)
            cy, _, _, _ = np.linalg.lstsq(A, ry[mask], rcond=None)
        except np.linalg.LinAlgError:
            log.warning("Kalibrasyon: lstsq başarısız")
            return

        M = np.array([[cx[0], cx[1]], [cy[0], cy[1]]])

        # İzotropik düzeltme
        swap = abs(M[0, 1]) > abs(M[0, 0])
        dom_x = abs(M[0, 1]) if swap else abs(M[0, 0])
        dom_y = abs(M[1, 0]) if swap else abs(M[1, 1])
        if dom_x > 1e-9 and dom_y < 0.3 * dom_x:
            sign = 1.0 if (M[1, 0] if swap else M[1, 1]) >= 0 else -1.0
            if swap: M[1, 0] = sign * dom_x
            else:    M[1, 1] = sign * dom_x
        elif dom_y > 1e-9 and dom_x < 0.3 * dom_y:
            sign = 1.0 if (M[0, 1] if swap else M[0, 0]) >= 0 else -1.0
            if swap: M[0, 1] = sign * dom_y
            else:    M[0, 0] = sign * dom_y

        # Stabilite: büyük Y gerilemesi engelle
        if self._calibrated:
            new_dy = abs(M[1, 0]) if swap else abs(M[1, 1])
            prev_dy = abs(self._M[1, 0]) if (abs(self._M[0, 1]) > abs(self._M[0, 0])) else abs(self._M[1, 1])
            if new_dy < 0.5 * prev_dy:
                return

        est = (M @ np.column_stack([cix, ciy]).T).T
        rmse = float(np.sqrt(np.mean((est[:, 0] - rx)**2 + (est[:, 1] - ry)**2)))

        self._M = M
        self._calibrated = True
        log.info(
            f"[{self.__class__.__name__}] n={mask.sum()}/{len(data)}  RMSE={rmse:.4f}m  "
            f"M=[[{M[0,0]:.4f},{M[0,1]:.4f}],[{M[1,0]:.4f},{M[1,1]:.4f}]]"
        )
