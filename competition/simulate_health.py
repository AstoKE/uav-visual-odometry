#!/usr/bin/env python3
"""
simulate_health.py — Yarışma health flag + frame drop senaryosu üretici

Desteklenen senaryolar:
  standard          — klasik kesintili pattern (kısa/uzun bloklar)
  burst             — sık kısa kesmeler (2–8 frame)
  blackout          — tek büyük kesinti (100–400 frame)
  competition       — 5 dakika / 2250 frame gerçekçi yarışma senaryosu
  competition_long  — 10 dakika / 4500 frame çoklu büyük blackout senaryosu

Tüm senaryolarda:
  • İlk 450 frame (calib_end): health=1 garantili
  • Rastgele single-frame dropout (sağlıklı framelerde anlık kayıp)
  • %5–10 frame drop simülasyonu (frame sırası boşlukları)

Kullanım:
    python3 competition/simulate_health.py [--frames 2250] [--scenario competition]
    python3 competition/simulate_health.py --print --scenario burst --frames 600
    python3 competition/simulate_health.py --frames 2250 --scenario competition --drop-rate 0.07
"""

import argparse
import numpy as np
import os

REPO        = os.path.expanduser("~/code/uav-visual-odometry")
DEFAULT_OUT = os.path.join(REPO, "competition/results/health_flags.npy")

# Yarışma PDF §3: ilk 450 frame kalibrasyon penceresi
CALIB_FRAMES = 450


# ── Frame drop simülasyonu ────────────────────────────────────────────────────

def simulate_frame_drops(
    n_frames:  int,
    rng,
    drop_rate: float = 0.07,
) -> np.ndarray:
    """
    Frame drop maskesi üret: 1 = frame var, 0 = frame düşürüldü.

    Gerçek sistemlerde USB/network gecikmesi nedeniyle %5–10 frame
    düşebilir. Dropped frame'lerde tahmin önceki değeri tutar.

    drop_rate : 0.05–0.10 arası tipik değer
    """
    mask = np.ones(n_frames, dtype=np.int8)
    # Cluster-based drop: ardışık 1-3 frame grupları
    i = CALIB_FRAMES  # kalibrasyon penceresinde drop yok
    while i < n_frames:
        if rng.random() < drop_rate:
            drop_len = int(rng.integers(1, 4))
            mask[i:min(i + drop_len, n_frames)] = 0
            i += drop_len + int(rng.integers(5, 20))  # küme sonrası boşluk
        else:
            i += 1
    return mask


# ── Health flag senaryoları ───────────────────────────────────────────────────

def _apply_dropout(flags: np.ndarray, rng,
                   n_min: int = 20, n_max: int = 40) -> None:
    """Sağlıklı framelere rastgele single-frame dropout (in-place)."""
    n_drop  = rng.integers(n_min, n_max + 1)
    healthy = np.where(flags == 1)[0]
    # Kalibrasyon penceresini koru
    healthy = healthy[healthy >= CALIB_FRAMES]
    if len(healthy) >= n_drop:
        chosen = rng.choice(healthy, size=int(n_drop), replace=False)
        flags[chosen] = 0


def scenario_standard(n: int, rng) -> np.ndarray:
    """
    Klasik pattern:
      0–calib_end   : health=1
      calib–calib+90: health=0
      sonra          : health=1 blokları + health=0 uzun kesintiler
    """
    flags     = np.ones(n, dtype=np.int8)
    c         = min(CALIB_FRAMES, n)
    seg1_end  = min(c + 90,  n)
    seg2_end  = min(c + 150, n)
    seg3_end  = n

    flags[c:seg1_end]     = 0
    flags[seg2_end:n]     = 0
    _apply_dropout(flags, rng)
    return flags


def scenario_burst(n: int, rng) -> np.ndarray:
    """
    Sık kısa kesmeler (2–8 frame) kalibrasyondan sonra.
    Son %30: tamamen kesik.
    """
    flags    = np.ones(n, dtype=np.int8)
    c        = min(CALIB_FRAMES, n)

    i = c + 5
    while i < int(n * 0.70):
        healthy_len = int(rng.integers(8, 30))
        i          += healthy_len
        if i >= int(n * 0.70):
            break
        fail_len = int(rng.integers(2, 9))
        flags[i:min(i + fail_len, n)] = 0
        i += fail_len

    flags[int(n * 0.70):] = 0
    _apply_dropout(flags, rng, n_min=5, n_max=15)
    return flags


def scenario_blackout(n: int, rng) -> np.ndarray:
    """
    İlk calib_end frame sağlıklı; ortada 150–400 framelık tek büyük blackout;
    sonunda kısa toparlanma penceresi.
    """
    flags    = np.ones(n, dtype=np.int8)
    c        = min(CALIB_FRAMES, n)

    # Büyük blackout
    bo_start = int(rng.integers(c + 20, max(c + 80, int(n * 0.35))))
    bo_len   = int(rng.integers(min(150, n // 4), min(400, int(n * 0.65))))
    bo_end   = min(bo_start + bo_len, int(n * 0.88))
    flags[bo_start:bo_end] = 0

    _apply_dropout(flags, rng)
    return flags


def scenario_competition(n: int, rng) -> np.ndarray:
    """
    Gerçekçi yarışma senaryosu — PDF §3 uyumlu:

      Seg 0 (0–calib_end)    : health=1 garantili  (kalibrasyon penceresi)
      Seg 1 (calib–30%)      : burst failure başlar
      Seg 2 (30%–55%)        : büyük kesinti (150–300 frame)
      Seg 3 (55%–75%)        : kısa toparlanma + sık kesintiler
      Seg 4 (75%–100%)       : uzun kesinti (GPS-denied final)

    n < 2250 için oransal ölçekleme uygulanır.
    """
    flags  = np.ones(n, dtype=np.int8)
    scale  = n / 2250.0
    c      = min(CALIB_FRAMES, int(450 * scale))

    seg1_end = int(n * 0.30)
    seg2_end = int(n * 0.55)
    seg3_end = int(n * 0.75)

    # Seg 1: burst failure
    i = c
    while i < seg1_end:
        healthy_len = int(rng.integers(15, 45))
        i          += healthy_len
        if i >= seg1_end:
            break
        fail_len = int(rng.integers(3, 15))
        flags[i:min(i + fail_len, seg1_end)] = 0
        i += fail_len

    # Seg 2: büyük blackout
    bo_len = int(rng.integers(int(150 * scale), int(300 * scale) + 1))
    flags[seg1_end:min(seg1_end + bo_len, seg2_end)] = 0

    # Seg 3: kısa toparlanma + burst
    i = seg2_end
    while i < seg3_end:
        healthy_len = int(rng.integers(8, 28))
        i          += healthy_len
        if i >= seg3_end:
            break
        fail_len = int(rng.integers(2, 10))
        flags[i:min(i + fail_len, seg3_end)] = 0
        i += fail_len

    # Seg 4: uzun final kesintisi
    flags[seg3_end:] = 0

    _apply_dropout(flags, rng, n_min=10, n_max=30)
    return flags


def scenario_competition_long(n: int, rng) -> np.ndarray:
    """
    10 dakika / 4500 frame uzun yarışma senaryosu.

    Kalibrasyon (0–450) + çoklu blackout + iki büyük GPS-denied bölgesi.
    Gerçek yarışmalarda sahaya çıkış → dönüş → ikinci tur benzeri senaryo.

      Seg 0 (0–calib_end)    : health=1  (kalibrasyon)
      Seg 1 (calib–20%)      : yoğun burst failure
      Seg 2 (20%–38%)        : büyük blackout (200–450 frame)
      Seg 3 (38%–52%)        : kısa toparlanma + ara burst
      Seg 4 (52%–65%)        : ikinci büyük blackout (150–350 frame)
      Seg 5 (65%–82%)        : orta yoğunlukta burst
      Seg 6 (82%–100%)       : uzun final GPS-denied
    """
    flags = np.ones(n, dtype=np.int8)
    scale = n / 4500.0
    c     = min(CALIB_FRAMES, int(450 * scale))

    seg1_end = int(n * 0.20)
    seg2_end = int(n * 0.38)
    seg3_end = int(n * 0.52)
    seg4_end = int(n * 0.65)
    seg5_end = int(n * 0.82)

    # Seg 1: yoğun burst
    i = c
    while i < seg1_end:
        healthy_len = int(rng.integers(10, 30))
        i += healthy_len
        if i >= seg1_end:
            break
        fail_len = int(rng.integers(4, 18))
        flags[i:min(i + fail_len, seg1_end)] = 0
        i += fail_len

    # Seg 2: büyük blackout
    bo1_len = int(rng.integers(int(200 * scale), int(450 * scale) + 1))
    flags[seg1_end:min(seg1_end + bo1_len, seg2_end)] = 0

    # Seg 3: toparlanma + burst
    i = seg2_end
    while i < seg3_end:
        healthy_len = int(rng.integers(12, 35))
        i += healthy_len
        if i >= seg3_end:
            break
        fail_len = int(rng.integers(2, 12))
        flags[i:min(i + fail_len, seg3_end)] = 0
        i += fail_len

    # Seg 4: ikinci büyük blackout
    bo2_len = int(rng.integers(int(150 * scale), int(350 * scale) + 1))
    flags[seg3_end:min(seg3_end + bo2_len, seg4_end)] = 0

    # Seg 5: orta burst
    i = seg4_end
    while i < seg5_end:
        healthy_len = int(rng.integers(15, 40))
        i += healthy_len
        if i >= seg5_end:
            break
        fail_len = int(rng.integers(3, 10))
        flags[i:min(i + fail_len, seg5_end)] = 0
        i += fail_len

    # Seg 6: final GPS-denied
    flags[seg5_end:] = 0

    _apply_dropout(flags, rng, n_min=15, n_max=40)
    return flags


SCENARIOS = {
    "standard":          scenario_standard,
    "burst":             scenario_burst,
    "blackout":          scenario_blackout,
    "competition":       scenario_competition,
    "competition_long":  scenario_competition_long,
}


# ── Ana API ───────────────────────────────────────────────────────────────────

def make_health_flags(
    n_frames:  int   = 2250,
    seed:      int   = 42,
    scenario:  str   = "competition",
    drop_rate: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Health flag dizisi + frame drop maskesi üret.

    Parameters
    ----------
    n_frames  : int   — toplam frame sayısı
    seed      : int   — rastlantı tohumu
    scenario  : str   — "standard" | "burst" | "blackout" | "competition" | "competition_long"
    drop_rate : float — frame drop oranı (0 → devre dışı, önerilen: 0.05–0.10)

    Returns
    -------
    health_flags : np.ndarray[int8] shape (n_frames,)  — {0, 1}
    drop_mask    : np.ndarray[int8] shape (n_frames,)  — {0=dropped, 1=present}
    """
    rng  = np.random.default_rng(seed)
    fn   = SCENARIOS.get(scenario, scenario_competition)
    flags = fn(n_frames, rng)

    if drop_rate > 0.0:
        drop_mask = simulate_frame_drops(n_frames, rng, drop_rate)
    else:
        drop_mask = np.ones(n_frames, dtype=np.int8)

    return flags, drop_mask


def health_summary(
    flags:     np.ndarray,
    drop_mask: np.ndarray,
    scenario:  str = "",
) -> str:
    n   = len(flags)
    n1  = int(flags.sum())
    n0  = n - n1
    n_drop = int((drop_mask == 0).sum())

    # Segment analizi
    segments = []
    i = 0
    while i < n:
        val = int(flags[i])
        j   = i
        while j < n and flags[j] == val:
            j += 1
        segments.append((i, j - 1, val, j - i))
        i = j

    max_h0 = max((s[3] for s in segments if s[2] == 0), default=0)
    n_trans = sum(1 for k in range(1, n) if flags[k-1] != flags[k])

    lines = [
        f"Senaryo           : {scenario or 'competition'}",
        f"Toplam frame      : {n}",
        f"Health=1          : {n1} ({100*n1/n:.1f}%)",
        f"Health=0          : {n0} ({100*n0/n:.1f}%)",
        f"Geçiş sayısı      : {n_trans}",
        f"Maks h=0 uzunluk  : {max_h0} frame",
        f"Frame drop        : {n_drop} ({100*n_drop/n:.1f}%)",
        f"Kalibrasyon güvencesi: frame 0–{min(CALIB_FRAMES-1,n-1)} h=1",
        "",
        "Büyük segmentler (≥10 frame):",
    ]
    for s in segments:
        if s[3] >= 10:
            lines.append(
                f"  [{s[0]:5d}–{s[1]:5d}]  h={s[2]}  ({s[3]:4d} frame)"
            )
    return "\n".join(lines)


PRESETS_DIR = os.path.join(os.path.dirname(DEFAULT_OUT), "..", "health_patterns")

# Hazır preset tanımları: (senaryo, frame_sayısı, seed, drop_rate)
_PRESET_CONFIGS: list[tuple[str, int, int, float]] = [
    ("standard",         2250, 42,  0.07),
    ("burst",            2250, 42,  0.07),
    ("blackout",         2250, 42,  0.07),
    ("competition",      2250, 42,  0.07),
    ("competition",      2250, 99,  0.05),   # alternatif seed
    ("competition",      2250, 7,   0.10),   # yüksek drop
    ("competition_long", 4500, 42,  0.07),
    ("competition_long", 4500, 123, 0.05),
]


def gen_presets(out_dir: str, verbose: bool = True) -> None:
    """Tüm preset senaryoları üret ve `out_dir` altına kaydet."""
    os.makedirs(out_dir, exist_ok=True)
    for scenario, n_frames, seed, drop_rate in _PRESET_CONFIGS:
        name = f"{scenario}_n{n_frames}_s{seed}_dr{int(drop_rate*100):02d}"
        flags, drop_mask = make_health_flags(n_frames, seed, scenario, drop_rate)

        np.save(os.path.join(out_dir, f"{name}_health.npy"),  flags)
        np.save(os.path.join(out_dir, f"{name}_dropmask.npy"), drop_mask)

        if verbose:
            n1 = int(flags.sum())
            print(
                f"  {name:<52}  h1={n1}/{n_frames} "
                f"({100*n1/n_frames:.0f}%)  "
                f"drop={(drop_mask==0).sum()}"
            )
    print(f"\n{len(_PRESET_CONFIGS)} preset kaydedildi → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Health flag + frame drop simülatörü")
    parser.add_argument("--frames",    type=int,   default=2250)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--scenario",  default="competition",
                        choices=list(SCENARIOS.keys()))
    parser.add_argument("--drop-rate", type=float, default=0.07,
                        help="Frame drop oranı (0=devre dışı, önerilen: 0.05–0.10)")
    parser.add_argument("--out",       default=DEFAULT_OUT)
    parser.add_argument("--print",     action="store_true", dest="print_only")
    parser.add_argument("--gen-presets", action="store_true",
                        help="Tüm preset senaryoları üret → competition/health_patterns/")
    parser.add_argument("--presets-dir", default=PRESETS_DIR,
                        help="Preset çıktı dizini")
    args = parser.parse_args()

    if args.gen_presets:
        gen_presets(os.path.realpath(args.presets_dir))
        return

    flags, drop_mask = make_health_flags(
        n_frames=args.frames,
        seed=args.seed,
        scenario=args.scenario,
        drop_rate=args.drop_rate,
    )
    print(health_summary(flags, drop_mask, args.scenario))

    if not args.print_only:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.save(args.out, flags)
        drop_out = args.out.replace("health_flags", "drop_mask")
        np.save(drop_out, drop_mask)
        print(f"\nHealth kaydedildi : {args.out}")
        print(f"Drop mask kaydedildi: {drop_out}")


if __name__ == "__main__":
    main()
