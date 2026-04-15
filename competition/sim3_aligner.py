"""
sim3_aligner.py — Sim(3) Hizalama Modülü

PDF Bölüm 4.3: İlk 450 kare boyunca görsel odometri çıktısı ve referans
pozisyon bilgisi kullanılarak Sim(3) hizalama yapılmaktadır.

Sim(3) dönüşümü:
    world_pos = s * R @ vo_pos + t

Umeyama (1991) kapalı-form çözümü ile:
    - s : ölçek faktörü  (birim-skala VO → metrik dünya)
    - R : 3×3 rotasyon matrisi (koordinat ekseni hizalama)
    - t : 3-vektör translasyon ofseti

Kullanım:
    aligner = Sim3Aligner(min_pairs=10, update_every=20)
    aligner.add(vo_pos, ref_pos)      # health=1 frame'lerde çağır
    wx, wy, wz = aligner.apply(vo_pos)  # herhangi bir frame'de kullan
"""

import logging
import numpy as np

log = logging.getLogger(__name__)


# ── Kapalı-form Umeyama hizalama ──────────────────────────────────────────────

def umeyama_alignment(src: np.ndarray, dst: np.ndarray
                      ) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Sim(3) hizalama: dst ≈ s * R @ src + t

    Parameters
    ----------
    src : (N, 3) — kaynak nokta bulutu (VO koordinat çerçevesi)
    dst : (N, 3) — hedef nokta bulutu (referans/dünya koordinat çerçevesi)

    Returns
    -------
    s   : float    — ölçek faktörü
    R   : (3, 3)   — rotasyon matrisi
    t   : (3,)     — translasyon vektörü
    """
    n = len(src)
    if n < 3:
        return 1.0, np.eye(3), np.zeros(3)

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)

    src_c = src - mu_s
    dst_c = dst - mu_d

    var_s = float(np.mean(np.sum(src_c ** 2, axis=1)))
    if var_s < 1e-12:
        # Neredeyse sabit kaynak — sadece translasyon
        return 1.0, np.eye(3), mu_d - mu_s

    H = src_c.T @ dst_c / n  # (3, 3)
    U, D, Vt = np.linalg.svd(H)

    det_sign = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    S_diag = np.array([1.0, 1.0, det_sign if det_sign != 0 else 1.0])

    R = Vt.T @ np.diag(S_diag) @ U.T
    s = float(np.dot(D, S_diag) / var_s)
    t = mu_d - s * R @ mu_s

    # Sağlıklılık kontrolü
    if not np.isfinite(s) or s <= 0 or s > 1e5:
        log.warning(f"Umeyama: geçersiz ölçek s={s:.4g} — yedek s=1 kullanılıyor")
        s = 1.0
        R = np.eye(3)
        t = mu_d - mu_s

    return s, R, t


# ── Sim(3) Aligner sınıfı ─────────────────────────────────────────────────────

class Sim3Aligner:
    """
    Artımlı Sim(3) hizalayıcı.

    İlk 450 kare (health=1) boyunca (vo_pos, ref_pos) çiftleri toplanır.
    Yeterli çift birikince (min_pairs) Umeyama hizalaması hesaplanır.
    Periyodik olarak (update_every) güncellenir.

    Parameters
    ----------
    min_pairs    : int — hizalama başlatmak için minimum çift sayısı
    update_every : int — kaç yeni çift eklenince yeniden hizalanır
    """

    def __init__(self, min_pairs: int = 10, update_every: int = 20):
        self.min_pairs   = min_pairs
        self.update_every = update_every

        self._pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self._calibrated = False

        # Sim(3) parametreleri
        self._s = 1.0
        self._R = np.eye(3, dtype=np.float64)
        self._t = np.zeros(3, dtype=np.float64)

        self._rmse_calib = float("nan")

    # ── Veri ekleme ve kalibrasyon ─────────────────────────────────────────────

    def add(self, vo_pos: np.ndarray, ref_pos: np.ndarray) -> None:
        """
        Yeni (VO, referans) pozisyon çifti ekle.

        vo_pos  : (3,) VO koordinat çerçevesi (birim ölçek)
        ref_pos : (3,) Referans koordinat çerçevesi (metre)
        """
        self._pairs.append((np.asarray(vo_pos, dtype=np.float64).copy(),
                            np.asarray(ref_pos, dtype=np.float64).copy()))

        n = len(self._pairs)
        if n >= self.min_pairs and (
            not self._calibrated or n % self.update_every == 0
        ):
            self._run_alignment()

    def _run_alignment(self) -> None:
        src = np.array([p[0] for p in self._pairs])  # (N, 3) VO
        dst = np.array([p[1] for p in self._pairs])  # (N, 3) referans

        s, R, t = umeyama_alignment(src, dst)

        # RMSE hesapla
        est = (s * (R @ src.T)).T + t  # (N, 3)
        self._rmse_calib = float(np.sqrt(np.mean(
            np.sum((est - dst) ** 2, axis=1)
        )))

        # Stabilite: yeniden kalibrasyon RMSE'yi %50'den fazla artırıyorsa reddet
        if self._calibrated:
            if self._rmse_calib > 2.0 and len(self._pairs) > 50:
                log.warning(
                    f"Sim3: kalibrasyon RMSE yüksek ({self._rmse_calib:.3f}m), güncelleme reddedildi"
                )
                return

        self._s = s
        self._R = R
        self._t = t
        self._calibrated = True

        log.info(
            f"[Sim3] n={len(self._pairs)}  s={s:.4f}  "
            f"RMSE={self._rmse_calib:.4f}m  "
            f"R_det={np.linalg.det(R):.3f}"
        )

    # ── Uygulama ───────────────────────────────────────────────────────────────

    def apply(self, vo_pos: np.ndarray) -> np.ndarray:
        """
        Sim(3) dönüşümü uygula: world_pos = s * R @ vo_pos + t

        vo_pos : (3,) VO koordinat çerçevesi (birim ölçek)
        Döndürür: (3,) dünya koordinat çerçevesi (metre)
        """
        v = np.asarray(vo_pos, dtype=np.float64)
        return self._s * (self._R @ v) + self._t

    # ── Özellikler ─────────────────────────────────────────────────────────────

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def n_pairs(self) -> int:
        return len(self._pairs)

    @property
    def rmse_calib(self) -> float:
        return self._rmse_calib

    @property
    def scale(self) -> float:
        return self._s

    # min_pairs / update_every: doğrudan instance attribute olarak erişilebilir,
    # _robust_refit içinde Sim3Aligner() inşaatı için bu değerlere ihtiyaç var.

    def get_state(self) -> dict:
        return {
            "calibrated":  self._calibrated,
            "n_pairs":     len(self._pairs),
            "s":           self._s,
            "R":           self._R.tolist(),
            "t":           self._t.tolist(),
            "rmse_calib":  self._rmse_calib,
        }
