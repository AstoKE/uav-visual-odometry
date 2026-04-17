"""
estimator.py — Teknofest İkinci Görev: Pozisyon Tespiti Sistemi

PDF Bölüm 3.1 mimarisine uygun implementasyon:

  Video Frame
    ↓ Kamera Kalibrasyonu & Undistortion        (PDF 4.1)
    ↓ Feature Extraction (ORB)                  (PDF 4.2)
    ↓ Feature Matching (BFMatcher + Lowe)       (PDF 4.2)
    ↓ RANSAC + Homography (planar sahne)        (PDF 4.2)
    ↓ Relative Pose Estimation (R, t)           (PDF 4.2)
    ↓ Trajectory Integration                    (PDF 4.2)
    ↓ (İlk 450 kare) Sim(3) Alignment           (PDF 4.3)
    ↓ GPS-denied Mode                           (PDF 4.4)
    → ΔX, ΔY, ΔZ çıktısı (metre, referans çerçeve)

Güçlendirmeler (robustness):
  • Adaptive drift threshold: confidence × max_jump_m (düşük güven → sıkı eşik)
  • Confidence score: RANSAC inlier oranı + temporal smoothing
  • Temporal smoothing: health=0'da EMA; health=1'de hafif EMA (gecikmesiz)
  • Robust Sim(3) weighting: Huber ağırlıklı kalibrasyon çiftleri
    (outlier çiftler downweight edilir → kalibrasyon kararlılığı artar)
  • Velocity estimation: son 5 keyframe'den median hız tahmini
    (adaptive threshold için kullanılır)

Kullanım:
    from competition.estimator import OnlineEstimator
    est = OnlineEstimator(fx=457, fy=457, cx=320, cy=240)
    for frame, ref_pos, health in frames:
        x, y, z = est.update(frame, ref_pos, health)
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Optional

import cv2
import numpy as np

from competition.sim3_aligner import Sim3Aligner

log = logging.getLogger(__name__)


class OnlineEstimator:
    """
    Teknofest İkinci Görev — Online Pozisyon Tahmincisi

    PDF mimarisi: ORB → Homography → R,t → Sim(3) → ΔX, ΔY, ΔZ

    Parameters
    ----------
    fx, fy, cx, cy         : float — kamera intrinsik parametreleri
    dist_coeffs            : array — distortion katsayıları (simülasyon: None)
    n_features             : int   — ORB keypoint üst sınırı
    lowe_ratio             : float — Lowe ratio test eşiği (önerilen: 0.70–0.80)
    ransac_thresh          : float — Homography RANSAC piksel eşiği
    ema_alpha              : float — EMA ağırlığı (1.0=yok, 0.0=max smooth)
    max_jump_m             : float — drift rejection taban eşiği (metre)
    adaptive_drift         : bool  — True → eşik = confidence × max_jump_m
    sim3_min_pairs         : int   — Sim(3) kalibrasyon başlatma eşiği
    sim3_update_every      : int   — Sim(3) periyodik güncelleme aralığı
    min_keyframe_flow_px   : float — keyframe tetiklemek için minimum LK akışı (px)
    altitude_m             : float — bilinen kamera irtifası (metrik ölçek için)
    sim3_robust_sigma      : float — Huber ağırlıklandırma sigma (m); None=devre dışı
    """

    def __init__(
        self,
        fx: float = 457.0,
        fy: float = 457.0,
        cx: float = 320.0,
        cy: float = 240.0,
        dist_coeffs=None,
        n_features: int = 1500,
        lowe_ratio: float = 0.75,
        ransac_thresh: float = 2.0,
        ema_alpha: float = 0.7,
        max_jump_m: float = 3.0,
        adaptive_drift: bool = True,
        sim3_min_pairs: int = 10,
        sim3_update_every: int = 25,
        min_keyframe_flow_px: float = 5.0,
        altitude_m: Optional[float] = None,
        sim3_robust_sigma: Optional[float] = 1.5,
    ):
        # 4.1 Kamera matrisi
        self.K = np.array([[fx,  0.0, cx],
                           [0.0,  fy, cy],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        self._dist = (np.asarray(dist_coeffs, dtype=np.float64)
                      if dist_coeffs is not None else None)

        self.ema_alpha      = ema_alpha
        self.max_jump_m     = max_jump_m
        self._adaptive_drift = adaptive_drift

        # ── 4.2 ORB + BFMatcher ──────────────────────────────────────────────
        self._orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=15,
        )
        self._matcher    = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._lowe_ratio = lowe_ratio
        self._ransac_thr = ransac_thresh

        # ── 4.4 Keyframe tabanlı yapı ────────────────────────────────────────
        self._min_keyframe_flow_px: float = min_keyframe_flow_px
        self._keyframe_gray: Optional[np.ndarray] = None

        # Bilinen irtifa → metrik ölçek (opsiyonel)
        self._altitude_m: Optional[float] = altitude_m

        # Kümülatif yörünge (4×4 homojen, VO çerçevesi)
        self._T_world = np.eye(4, dtype=np.float64)

        # ── 4.3 Sim(3) hizalama (Huber-robust opsiyonel, sliding window) ────────
        self._sim3 = Sim3Aligner(
            min_pairs=sim3_min_pairs,
            update_every=sim3_update_every,
            window_size=80,
        )
        self._sim3_robust_sigma: float | None = sim3_robust_sigma

        # Ham kalibrasyon çiftleri (robust weighting için saklanır)
        self._calib_vo:  list[np.ndarray] = []
        self._calib_ref: list[np.ndarray] = []

        # Dünya pozisyonu (metrik)
        self._world_x = 0.0
        self._world_y = 0.0
        self._world_z = 0.0

        # EMA durumu
        self._ema_x    = 0.0
        self._ema_y    = 0.0
        self._ema_z    = 0.0
        self._ema_init = False

        # Confidence (RANSAC inlier oranı + temporal smoothing)
        self._confidence     = 0.5        # başlangıç: orta güven
        self._last_inliers   = 0
        self._drift_rejected = 0
        self._pos_history: deque = deque(maxlen=20)

        # Hız tahmini — vektör (vx, vy) olarak tutulur
        self._vel_history: deque = deque(maxlen=5)      # scalar m/frame (adaptive threshold)
        self._vel_vec_history: deque = deque(maxlen=8)  # (vx, vy) vektörler (extrapolation)

        # Motion model: son başarılı T
        self._last_valid_T: Optional[np.ndarray] = None

        self._frame_count   = 0
        self._kf_count      = 0
        self._health0_count = 0   # art arda health=0 frame sayısı
        self._prev_health   = 1   # reset-on-health1 için önceki health durumu

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        ref_pos: Optional[tuple],
        health: int,
    ) -> tuple:
        """
        Yeni kare işle.

        Parameters
        ----------
        frame   : BGR veya gri görüntü ndarray
        ref_pos : (rx, ry, rz) metre | health=0'da None
        health  : 1 = referans güvenilir, 0 = GPS-siz mod

        Returns
        -------
        (est_x, est_y, est_z) — referans çerçevesinde metre, başlangıçtan itibaren
        """
        self._frame_count += 1
        gray = self._preprocess(frame)

        # ── Health 0→1 geçişi: VO origin sıfırla ─────────────────────────────
        # Birikim yapmış VO drift'i bir sonraki health=0 segmentine taşıma.
        # T_world identity'ye döner, Sim3 yeni health=1 segmentiyle yeniden fit edilir.
        if health == 1 and self._prev_health == 0 and self._keyframe_gray is not None:
            # Tam reset: T_world + Sim3 + kalibrasyon verisi.
            # Blackout sonrası birikmiş VO drift'i bir sonraki health=0'a taşıma.
            # min_pairs=5 ile hızlı yeniden kalibrasyon (5 health=1 kare yeterli).
            self._T_world      = np.eye(4, dtype=np.float64)
            self._last_valid_T = None
            self._sim3         = Sim3Aligner(
                min_pairs    = 5,
                update_every = self._sim3.update_every,
                window_size  = self._sim3.window_size,
            )
            self._calib_vo.clear()
            self._calib_ref.clear()
            self._pos_history.clear()
            self._vel_vec_history.clear()
            log.debug(f"[Reset] health 0→1 frame={self._frame_count}: tam sıfırlama")
        self._prev_health = health

        # ── İlk kare: keyframe başlat ─────────────────────────────────────────
        if self._keyframe_gray is None:
            self._keyframe_gray = gray
            if health == 1 and ref_pos is not None:
                wx, wy, wz = float(ref_pos[0]), float(ref_pos[1]), float(ref_pos[2])
                self._world_x, self._world_y, self._world_z = wx, wy, wz
                self._ema_x, self._ema_y, self._ema_z = wx, wy, wz
                self._ema_init = True
            return self._ema_x, self._ema_y, self._ema_z

        # ── Keyframe hareketi kontrol et (LK flow, hızlı) ─────────────────────
        motion_px = self._estimate_motion_px(self._keyframe_gray, gray)

        if health == 0:
            self._health0_count += 1
        else:
            self._health0_count = 0

        if motion_px < self._min_keyframe_flow_px:
            # Yeterli hareket yok — pozisyon aynen kalsın
            if health == 1 and ref_pos is not None:
                new_wx = float(ref_pos[0])
                new_wy = float(ref_pos[1])
                new_wz = float(ref_pos[2])
                self._world_x, self._world_y, self._world_z = new_wx, new_wy, new_wz
                self._apply_ema(new_wx, new_wy, new_wz, health)
            return self._ema_x, self._ema_y, self._ema_z

        # ── Keyframe: yeterli hareket birikti, VO hesapla ─────────────────────
        self._kf_count += 1
        rel_R, rel_t, n_inl, T_rel = self._compute_relative_pose_pair(
            self._keyframe_gray, gray
        )
        self._keyframe_gray = gray
        self._last_inliers  = n_inl

        # ── Yörünge entegrasyonu ──────────────────────────────────────────────
        if rel_R is not None:
            self._T_world      = self._T_world @ T_rel
            self._last_valid_T = T_rel.copy()
        elif self._last_valid_T is not None and self._frame_count > 2:
            T_motion = np.eye(4, dtype=np.float64)
            T_motion[:3, :3] = self._last_valid_T[:3, :3]
            T_motion[:3, 3]  = self._last_valid_T[:3, 3] * 0.5
            self._T_world = self._T_world @ T_motion

        vo_pos = self._T_world[:3, 3].copy()

        # ── 4.3 Sim(3) kalibrasyon çifti ekle (health=1) ─────────────────────
        if health == 1 and ref_pos is not None:
            ref = np.array(ref_pos[:3], dtype=np.float64)
            self._add_calib_pair(vo_pos, ref)

        # ── Dünya pozisyonu hesapla ───────────────────────────────────────────
        if health == 1 and ref_pos is not None:
            new_wx = float(ref_pos[0])
            new_wy = float(ref_pos[1])
            new_wz = float(ref_pos[2])
        elif self._sim3.calibrated:
            wp     = self._sim3.apply(vo_pos)
            new_wx = float(wp[0])
            new_wy = float(wp[1])
            new_wz = float(wp[2])
        else:
            new_wx, new_wy, new_wz = self._world_x, self._world_y, self._world_z

        # ── 4.4 Adaptive drift detection + rollback ───────────────────────────
        if health == 0 and self._sim3.calibrated and self._pos_history and rel_R is not None:
            threshold = self._adaptive_jump_threshold()
            jump = math.sqrt((new_wx - self._world_x) ** 2 +
                             (new_wy - self._world_y) ** 2)
            if jump > threshold:
                self._drift_rejected += 1
                self._T_world = self._T_world @ np.linalg.inv(T_rel)
                # Donmak yerine hız vektörüyle extrapolation
                new_wx, new_wy, new_wz = self._extrapolate_position()
                log.debug(
                    f"[Drift] frame={self._frame_count}  jump={jump:.2f}m > "
                    f"thresh={threshold:.2f}m — extrapolate "
                    f"({new_wx:.2f},{new_wy:.2f})"
                )

        # Hız güncelle — hem scalar (adaptive threshold) hem vektör (extrapolation)
        if self._pos_history:
            prev = self._pos_history[-1]
            vx   = new_wx - prev[0]
            vy   = new_wy - prev[1]
            spd  = math.sqrt(vx ** 2 + vy ** 2)
            self._vel_history.append(spd)
            # Sadece gerçek hareketten vektör ekle (rejected değil)
            if spd > 0.01:
                self._vel_vec_history.append((vx, vy))

        self._world_x = new_wx
        self._world_y = new_wy
        self._world_z = new_wz
        self._pos_history.append((new_wx, new_wy, new_wz))

        # ── EMA smoothing ─────────────────────────────────────────────────────
        self._update_confidence(n_inl)
        self._apply_ema(new_wx, new_wy, new_wz, health)
        return self._ema_x, self._ema_y, self._ema_z

    # ── Yardımcı metodlar (update içinde kullanılan) ──────────────────────────

    def _extrapolate_position(self) -> tuple:
        """
        Son bilinen hız vektörlerinin medyanıyla pozisyon tahmini.
        VO gürültülü/drift yaptığında donmak yerine son hareketi sürdür.
        Geçmiş yoksa mevcut pozisyonu döndür (freeze).
        """
        if len(self._vel_vec_history) >= 2:
            vxs = [v[0] for v in self._vel_vec_history]
            vys = [v[1] for v in self._vel_vec_history]
            med_vx = float(np.median(vxs))
            med_vy = float(np.median(vys))
            return (self._world_x + med_vx,
                    self._world_y + med_vy,
                    self._world_z)
        return self._world_x, self._world_y, self._world_z

    def _apply_ema(self, wx: float, wy: float, wz: float, health: int) -> None:
        """
        EMA uygula. health=1'de hızlı yakınsama (gecikmesiz);
        health=0'da standart smoothing.
        """
        a = self.ema_alpha if health == 0 else min(self.ema_alpha + 0.2, 1.0)
        if not self._ema_init:
            self._ema_x, self._ema_y, self._ema_z = wx, wy, wz
            self._ema_init = True
        else:
            self._ema_x = a * wx + (1 - a) * self._ema_x
            self._ema_y = a * wy + (1 - a) * self._ema_y
            self._ema_z = a * wz + (1 - a) * self._ema_z

    def _adaptive_jump_threshold(self) -> float:
        """
        Adaptive drift eşiği:
          • Temel: max_jump_m
          • Confidence düşükse → eşiği sık → daha agresif rejection
          • Hız yüksekse → eşiği genişlet (hızlı UAV meşru büyük adımlar atabilir)
        """
        if not self._adaptive_drift:
            return self.max_jump_m

        # Confidence faktörü: [0.3, 1.0] aralığında ölçekle
        conf_factor = max(0.3, min(1.0, self._confidence))

        # Hız faktörü: son 5 keyframe medyan hızı
        if self._vel_history:
            median_vel = float(np.median(list(self._vel_history)))
            vel_factor  = max(1.0, min(2.5, 1.0 + median_vel / self.max_jump_m))
        else:
            vel_factor = 1.0

        threshold = self.max_jump_m * conf_factor * vel_factor
        return float(threshold)

    def _add_calib_pair(self, vo_pos: np.ndarray, ref_pos: np.ndarray) -> None:
        """
        Sim(3) kalibrasyon çifti ekle. Robust weighting aktifse Huber-ağırlıklı
        yeniden kalibrasyon çalıştırır (outlier çiftler downweight edilir).
        """
        self._calib_vo.append(vo_pos.copy())
        self._calib_ref.append(ref_pos.copy())

        if self._sim3_robust_sigma is not None and len(self._calib_vo) >= 20:
            # Her 25 çiftte bir robust re-fit
            if len(self._calib_vo) % 25 == 0:
                self._robust_refit()
            else:
                self._sim3.add(vo_pos, ref_pos)
        else:
            self._sim3.add(vo_pos, ref_pos)

    def _robust_refit(self) -> None:
        """
        Huber ağırlıklı Sim(3) yeniden kalibrasyonu.

        Adımlar:
          1. Mevcut Sim(3) parametreleriyle artıkları hesapla
          2. Huber ağırlıkları ata (büyük artıklı çiftleri downweight)
          3. Ağırlıklı alt-küme ile yeniden fit
        """
        if not self._sim3.calibrated or len(self._calib_vo) < 10:
            return

        src = np.array(self._calib_vo,  dtype=np.float64)   # (N, 3)
        dst = np.array(self._calib_ref, dtype=np.float64)   # (N, 3)

        # Mevcut Sim(3) ile tahmin
        s = self._sim3._s
        R = np.array(self._sim3._R)
        t = np.array(self._sim3._t)
        est = (s * (R @ src.T)).T + t                        # (N, 3)

        # Artıklar ve Huber ağırlıkları
        residuals = np.linalg.norm(est - dst, axis=1)        # (N,)
        sigma     = float(self._sim3_robust_sigma)
        weights   = np.where(
            residuals <= sigma,
            1.0,
            sigma / (residuals + 1e-9),
        )                                                     # (N,)

        # Ağırlıklı alt-küme seç (ağırlık > 0.3)
        keep = weights > 0.30
        if keep.sum() < 10:
            log.debug("[RobustSim3] Yeterli inlier yok, skip")
            return

        # Sim3Aligner'ı yeniden oluştur ve güçlü çiftlerle besle
        from competition.sim3_aligner import Sim3Aligner
        new_sim3 = Sim3Aligner(
            min_pairs=self._sim3.min_pairs,
            update_every=self._sim3.update_every,
        )  # min_pairs / update_every → Sim3Aligner.__init__ instance attributes
        for i in np.where(keep)[0]:
            new_sim3.add(src[i], dst[i])

        if new_sim3.calibrated and new_sim3.rmse_calib < self._sim3.rmse_calib * 1.1:
            self._sim3 = new_sim3
            log.info(
                f"[RobustSim3] n={keep.sum()}/{len(src)}  "
                f"RMSE: {new_sim3.rmse_calib:.4f}m  scale={new_sim3.scale:.3f}"
            )

    # ── Özellikler ────────────────────────────────────────────────────────────

    @property
    def calibrated(self) -> bool:
        return self._sim3.calibrated

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def drift_rejected(self) -> int:
        return self._drift_rejected

    def get_state(self) -> dict:
        median_vel = float(np.median(list(self._vel_history))) if self._vel_history else 0.0
        return {
            "frame":           self._frame_count,
            "kf_count":        self._kf_count,
            "calibrated":      self._sim3.calibrated,
            "sim3_n_pairs":    self._sim3.n_pairs,
            "sim3_scale":      self._sim3.scale,
            "sim3_rmse":       self._sim3.rmse_calib,
            "world_x":         self._world_x,
            "world_y":         self._world_y,
            "ema_x":           self._ema_x,
            "ema_y":           self._ema_y,
            "confidence":      self._confidence,
            "drift_rejected":  self._drift_rejected,
            "last_inliers":    self._last_inliers,
            "jump_threshold":  self._adaptive_jump_threshold(),
            "median_vel_m":    median_vel,
            "health0_streak":  self._health0_count,
            "calib_pairs_raw": len(self._calib_vo),
        }

    # ── İç metodlar ───────────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """4.1 Undistortion + griye çevirme."""
        gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame.ndim == 3 else frame.copy())
        if self._dist is not None:
            gray = cv2.undistort(gray, self.K, self._dist)
        return gray

    def _estimate_motion_px(
        self,
        gray_prev: np.ndarray,
        gray_curr: np.ndarray,
        max_corners: int = 100,
    ) -> float:
        """
        LK optical flow ile iki kare arasındaki medyan piksel hareketini tahmin et.

        Bu, ORB + BFMatcher'dan çok daha hızlı (yaklaşık 5-10×) çalışır ve
        sadece keyframe tetikleme kararı için kullanılır.

        Döndürür: medyan piksel hareketi (scalar float)
        """
        pts = cv2.goodFeaturesToTrack(
            gray_prev,
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=8,
        )
        if pts is None or len(pts) < 4:
            return 0.0

        pts2, status, _ = cv2.calcOpticalFlowPyrLK(
            gray_prev, gray_curr, pts, None,
            winSize=(15, 15), maxLevel=2,
        )
        if pts2 is None or status is None:
            return 0.0

        good = status.ravel().astype(bool)
        if good.sum() < 3:
            return 0.0

        flow_mag = np.linalg.norm(
            pts2[good].reshape(-1, 2) - pts[good].reshape(-1, 2), axis=1
        )
        return float(np.median(flow_mag))

    def _compute_relative_pose_pair(
        self,
        gray_prev: np.ndarray,
        gray_curr: np.ndarray,
    ):
        """
        4.2 Görüntüden hareket çıkarımı — planar sahne uyumlu.

        Adımlar:
          1. ORB feature çıkarımı (önceki ve mevcut kare)
          2. BFMatcher kNN + Lowe ratio test
          3. RANSAC + Homography (düzlemsel zemin için E yerine H kullanılır)
             NOT: Essential Matrix zemine dik bakan monoküler kamerada dejenere
             olur. ORB-SLAM3 de bu senaryoda Homography tabanlı pose kurtarma kullanır.
          4. decomposeHomographyMat → R, t seçimi (en çok pozitif derinliğe sahip)
          5. Yedek: medyan inlier flow / fx → translasyon (H decomp başarısız ise)

        Döndürür: (R, t, n_inliers, T_4x4) veya (None, None, 0, I_4x4)
        """
        T_eye = np.eye(4, dtype=np.float64)

        # 1. ORB
        kp1, d1 = self._orb.detectAndCompute(gray_prev, None)
        kp2, d2 = self._orb.detectAndCompute(gray_curr,  None)

        if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None, None, 0, T_eye

        # 2. BFMatcher + Lowe ratio test
        raw = self._matcher.knnMatch(d1, d2, k=2)
        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n_m = pair
                if m.distance < self._lowe_ratio * n_m.distance:
                    good.append(m)

        if len(good) < 8:
            return None, None, 0, T_eye

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # Yeterli piksel hareketi kontrol et (ortalama)
        flow = pts2 - pts1
        mean_flow = float(np.mean(np.abs(flow)))
        if mean_flow < 0.5:   # çok küçük hareket → sıfır delta
            return None, None, 0, T_eye

        # 3. RANSAC + Homography
        H, mask_h = cv2.findHomography(pts1, pts2, cv2.RANSAC, self._ransac_thr)

        if mask_h is not None:
            inl_mask = mask_h.ravel().astype(bool)
            n_inl    = int(inl_mask.sum())
            inl_flow = flow[inl_mask]
        else:
            inl_mask = np.ones(len(flow), dtype=bool)
            n_inl    = len(flow)
            inl_flow = flow

        if n_inl < 4:
            return None, None, 0, T_eye

        # 4a. Homography decompose → R, t (en iyi çözümü seç)
        R, t = None, None
        if H is not None:
            try:
                _, Rs, ts, normals = cv2.decomposeHomographyMat(H, self.K)
                R, t = self._select_best_decomp(Rs, ts, normals, pts1, pts2, mask_h)
            except cv2.error:
                pass

        if R is None:
            # 4b. Yedek: medyan inlier flow → translasyon (rotasyon ihmal)
            #     Zemine dik, translasyon-baskın hareket için geçerli
            fx_val = self.K[0, 0]
            fy_val = self.K[1, 1]
            if self._altitude_m is not None:
                # Metrik ölçek: Δu_px × h / f → Δx metre
                tx = float(np.median(inl_flow[:, 0])) * self._altitude_m / fx_val
                ty = float(np.median(inl_flow[:, 1])) * self._altitude_m / fy_val
            else:
                tx = float(np.median(inl_flow[:, 0])) / fx_val
                ty = float(np.median(inl_flow[:, 1])) / fy_val
            t  = np.array([[tx], [ty], [0.0]])
            R  = np.eye(3, dtype=np.float64)
            log.debug(f"[VO] H-decomp başarısız, fallback flow: tx={tx:.4f} ty={ty:.4f}")

        # Altitude-weighted scale: t (unit-norm from Homography) →
        # metrik birimlere çevir. Homography t yönü doğruysa ölçek ≈ altitude/f.
        if self._altitude_m is not None and R is not None and not (R == np.eye(3)).all():
            # Tahmini t büyüklüğü: ortalama inlier flow magnitude × altitude/f
            flow_mag_px = float(np.mean(np.linalg.norm(inl_flow, axis=1)))
            scale_t = flow_mag_px * self._altitude_m / self.K[0, 0]
            t_arr = np.asarray(t).ravel()
            t_norm = np.linalg.norm(t_arr)
            if t_norm > 1e-9:
                t = (t_arr / t_norm * scale_t).reshape(3, 1)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3]  = np.asarray(t).ravel()

        return R, t, n_inl, T

    def _select_best_decomp(self, Rs, ts, normals, pts1, pts2, mask):
        """
        decomposeHomographyMat çözümleri arasından en uygun (R, t) seç.

        Kriter: En çok noktanın kameranın önünde (pozitif derinlik) olduğu çözüm.
        Normal filtreleme uygulanmaz — aşağı bakan kamerada normal yönü tutarsız
        olabilir.
        """
        inl = mask.ravel().astype(bool)
        p1  = pts1[inl]
        p2  = pts2[inl]
        if len(p1) == 0:
            return None, None

        best_R, best_t, best_score = None, None, -1

        for i in range(len(Rs)):
            R_i = Rs[i]
            t_i = np.asarray(ts[i]).ravel()

            # Noktaları üçgenleme ile derinlik kontrolü
            P1   = self.K @ np.eye(3, 4)
            P2_m = np.zeros((3, 4), dtype=np.float64)
            P2_m[:3, :3] = R_i
            P2_m[:3, 3]  = t_i
            P2   = self.K @ P2_m

            try:
                pts4d = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
                pts3d = pts4d[:3] / (pts4d[3:] + 1e-9)
                score = int((pts3d[2] > 0).sum())
            except cv2.error:
                score = 0

            if score > best_score:
                best_score = score
                best_R = R_i
                best_t = t_i.reshape(3, 1)

        # Çok az pozitif derinlik → güvenilmez çözüm
        if best_score < max(4, int(0.3 * len(p1))):
            return None, None

        return best_R, best_t

    def _update_confidence(self, n_inliers: int) -> None:
        max_feat = 150.0
        score = min(n_inliers / max_feat, 1.0)
        self._confidence = float(0.8 * score + 0.2 * self._confidence)
