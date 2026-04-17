"""
score_official.py — Resmi yarışma puanlama formülü (Denklem 2)

Kaynak: TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışma Şartnamesi §9.2

    E = (1/N) * Σ √((x̂ᵢ - xᵢ)² + (ŷᵢ - yᵢ)² + (ẑᵢ - zᵢ)²)

    Yani: health=0 olan TÜM frame'lerin 3D Öklid hata ortalaması (MAE_3D)
    (RMSE değil — mean of distances, not root of mean of squared distances)

Kullanım:
    from competition.score_official import official_score, score_summary

    mae = official_score(est_list, ref_list)
    # est_list / ref_list: [(x, y, z), ...] — aynı uzunlukta

NOT: health=1 frame'lerinde referans pozisyonu göndermek sıfır hata sağlar.
     Hata yalnızca health=0 frame'lerinden gelir — yarışma bu kısımları puanlar.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def euclidean_3d(
    est: Tuple[float, float, float],
    ref: Tuple[float, float, float],
) -> float:
    """Tek frame için 3D Öklid mesafesi (metre)."""
    return math.sqrt(
        (est[0] - ref[0]) ** 2 +
        (est[1] - ref[1]) ** 2 +
        (est[2] - ref[2]) ** 2
    )


def official_score(
    est_positions: Sequence[Tuple[float, float, float]],
    ref_positions: Sequence[Tuple[float, float, float]],
) -> float:
    """
    Resmi puanlama formülü — tüm frame'ler için MAE_3D.

    Parameters
    ----------
    est_positions : [(x̂, ŷ, ẑ), ...]  yarışmacının gönderdiği tahminler
    ref_positions : [(x, y, z),   ...]  gerçek referans pozisyonlar

    Returns
    -------
    float — ortalama yarışmacı hatası (metre)
             küçük = daha iyi puan
    """
    n = len(est_positions)
    if n == 0:
        return float("nan")
    if n != len(ref_positions):
        raise ValueError(
            f"est ve ref uzunlukları eşleşmiyor: {n} != {len(ref_positions)}"
        )
    total = sum(euclidean_3d(e, r) for e, r in zip(est_positions, ref_positions))
    return total / n


def score_summary(
    est_positions: Sequence[Tuple[float, float, float]],
    ref_positions: Sequence[Tuple[float, float, float]],
    health_flags: Sequence[int],
) -> dict:
    """
    Ayrıntılı skor raporu: tüm frame'ler + health=0 bölümü.

    health=1 frame'lerinde referans gönderildiği varsayılır (sıfır hata).
    Bu fonksiyon doğrulama ve simülasyon içindir.

    Parameters
    ----------
    est_positions  : [(x̂, ŷ, ẑ), ...]
    ref_positions  : [(x, y, z), ...]
    health_flags   : [1 veya 0, ...]

    Returns
    -------
    dict ile:
      mae_3d_all      — tüm frame'ler üzerinden MAE_3D
      mae_3d_dead     — yalnızca health=0 frame'ler (asıl yarışma skoru)
      mae_3d_alive    — yalnızca health=1 frame'ler (referans gönderilirse 0)
      n_total, n_dead, n_alive
      max_err_3d      — en kötü tek frame hatası
      errors          — tüm frame hataları listesi (float)
    """
    n = len(est_positions)
    errors = [euclidean_3d(e, r)
              for e, r in zip(est_positions, ref_positions)]

    dead_errs  = [errors[i] for i in range(n) if health_flags[i] == 0]
    alive_errs = [errors[i] for i in range(n) if health_flags[i] == 1]

    def _mean(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    return {
        "mae_3d_all":   _mean(errors),
        "mae_3d_dead":  _mean(dead_errs),   # ← yarışma puanı bu
        "mae_3d_alive": _mean(alive_errs),
        "n_total":      n,
        "n_dead":       len(dead_errs),
        "n_alive":      len(alive_errs),
        "max_err_3d":   max(errors) if errors else float("nan"),
        "errors":       errors,
    }


def health1_optimal_score(
    ref_positions: Sequence[Tuple[float, float, float]],
    health_flags: Sequence[int],
    est_dead_positions: Sequence[Tuple[float, float, float]],
) -> float:
    """
    Optimal strateji: health=1'de referansı direkt gönder (sıfır hata),
    health=0'da kendi tahminini gönder.

    Bu fonksiyon yarışma skorunu doğru hesaplar:
    - health=1 frame'leri için tahmin = referans → hata = 0
    - health=0 frame'leri için tahmin = est_dead_positions

    Parameters
    ----------
    ref_positions      : tüm frame'ler için referans
    health_flags       : tüm frame'ler için health
    est_dead_positions : yalnızca health=0 frame'leri için tahminler

    Returns
    -------
    float — MAE_3D (resmi yarışma skoru)
    """
    n = len(ref_positions)
    dead_idx = [i for i in range(n) if health_flags[i] == 0]

    if len(dead_idx) != len(est_dead_positions):
        raise ValueError(
            f"health=0 frame sayısı ({len(dead_idx)}) ile "
            f"est_dead_positions ({len(est_dead_positions)}) eşleşmiyor"
        )

    # health=1 → hata=0 (referans gönderildi)
    total = 0.0
    dead_iter = iter(est_dead_positions)
    for i in range(n):
        if health_flags[i] == 0:
            est = next(dead_iter)
            ref = ref_positions[i]
            total += euclidean_3d(est, ref)
        # health=1: hata=0, toplama dahil değil (sıfır katkı)

    # Yarışma formülü: N = toplam frame sayısı
    return total / n
