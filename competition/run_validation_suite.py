#!/usr/bin/env python3
"""
run_validation_suite.py — Final validation suite

Predefined senaryoları sırayla çalıştırır, metrikleri toplar ve
competition/validation_results.csv'e yazar.

Senaryolar (4 health pattern × farklı motion seed'leri):
  standard × seeds [42, 7, 13]
  burst     × seeds [42, 7, 13]
  blackout  × seeds [42, 7, 13]
  competition × seeds [42, 7, 13, 99, 31]  ← ana yarışma senaryosu, 5 tekrar

Metrikler (her senaryo için):
  rmse_2d        — health=0 bölümünde 2D konum RMSE (m)
  final_drift    — son frame'de GT'den sapma (m)
  max_drift      — health=0 boyunca maksimum anlık sapma (m)
  recovery_5     — kesinti başladıktan sonra 5 frame içinde <1m hataya ulaşan
                   bölüm yüzdesi (%)
  recovery_10    — aynı, 10 frame eşiği (%)
  calib_ok       — kalibrasyon başarılı mı (0/1)
  calib_n        — kalibrasyon çifti sayısı
  fps            — simülasyon FPS'i (frame/saniye, CPU bound)
  n_dead         — health=0 frame sayısı
  n_dead_pct     — health=0 yüzdesi

Kullanım:
    python3 competition/run_validation_suite.py
    python3 competition/run_validation_suite.py --frames 500 --out competition/validation_results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

_REPO = os.path.expanduser("~/code/uav-visual-odometry")
sys.path.insert(0, _REPO)

from competition.slam_pose_estimator import SLAMPoseEstimator
from competition.simulate_health     import make_health_flags
from competition.score_official      import official_score

DEFAULT_OUT = os.path.join(_REPO, "competition/validation_results.csv")

# ── Predefined senaryo tablosu ────────────────────────────────────────────────

SCENARIOS = [
    # (label,          health_scenario, n_frames, motion_seed, health_seed)
    ("std_s42",        "standard",    300,  42,  42),
    ("std_s07",        "standard",    300,   7,   7),
    ("std_s13",        "standard",    300,  13,  13),
    ("burst_s42",      "burst",       500,  42,  42),
    ("burst_s07",      "burst",       500,   7,   7),
    ("burst_s13",      "burst",       500,  13,  13),
    ("blackout_s42",   "blackout",    600,  42,  42),
    ("blackout_s07",   "blackout",    600,   7,   7),
    ("blackout_s13",   "blackout",    600,  13,  13),
    ("comp_s42",       "competition", 2000, 42,  42),
    ("comp_s07",       "competition", 2000,  7,   7),
    ("comp_s13",       "competition", 2000, 13,  13),
    ("comp_s99",       "competition", 2000, 99,  99),
    ("comp_s31",       "competition", 2000, 31,  31),
]

# ── Yörünge / delta üreticiler ────────────────────────────────────────────────

def gen_gt_figure8(n: int, ax: float = 4.5, ay: float = 3.5,
                   period: float = 25.0, step_dt: float = 0.08) -> np.ndarray:
    pos = np.zeros((n, 3))
    for i in range(n):
        t = i * step_dt
        theta = 2.0 * math.pi * t / period
        pos[i, 0] = ax * math.sin(theta)
        pos[i, 1] = ay * math.sin(2.0 * theta)
    return pos


def gen_slam_deltas(gt_pos: np.ndarray, rng,
                    noise_std: float = 0.00010,
                    drift_rate: float = 0.005,
                    coord_rot_deg: float = 8.0,
                    scale: float = 1 / 35.0) -> np.ndarray:
    n = len(gt_pos)
    angle = math.radians(coord_rot_deg)
    R = np.array([[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle),  math.cos(angle)]])
    slam_pos = np.zeros((n, 2))
    drift_x = drift_y = 0.0
    for i in range(n):
        raw = scale * R @ gt_pos[i, :2]
        drift_x += rng.normal(0, drift_rate * 0.01)
        drift_y += rng.normal(0, drift_rate * 0.01)
        noise = rng.normal(0, noise_std, 2)
        slam_pos[i] = raw + noise + np.array([drift_x, drift_y])
    deltas = np.zeros((n, 2))
    deltas[1:] = np.diff(slam_pos, axis=0)
    return deltas


# ── Tek senaryo çalıştırma ────────────────────────────────────────────────────

def run_scenario(label: str, health_scenario: str, n_frames: int,
                 motion_seed: int, health_seed: int) -> dict:
    rng = np.random.default_rng(motion_seed)
    gt_pos  = gen_gt_figure8(n_frames)
    deltas  = gen_slam_deltas(gt_pos, rng)
    health, _ = make_health_flags(n_frames, seed=health_seed, scenario=health_scenario)

    est = SLAMPoseEstimator(calib_min_frames=30, calib_update_every=50)

    est_x_arr = np.zeros(n_frames)
    est_y_arr = np.zeros(n_frames)
    est_z_arr = np.zeros(n_frames)

    t0 = time.perf_counter()
    for i in range(n_frames):
        h   = int(health[i])
        # GT Z=0 (yatay uçuş simülasyonu) — gerçek veride ref_z değişir
        ref = (float(gt_pos[i, 0]), float(gt_pos[i, 1]), 0.0) if h == 1 else None
        wx, wy, wz = est.update(float(deltas[i, 0]), float(deltas[i, 1]),
                                ref_pos=ref, health=h)
        est_x_arr[i] = wx
        est_y_arr[i] = wy
        est_z_arr[i] = wz
    elapsed = time.perf_counter() - t0

    fps = n_frames / elapsed if elapsed > 0 else float("nan")

    # ── Metrik hesapla ────────────────────────────────────────────────────────
    dead_mask = health == 0
    n_dead    = int(dead_mask.sum())

    # 2D hata (iç değerlendirme)
    errs_2d = np.sqrt((est_x_arr - gt_pos[:, 0])**2 + (est_y_arr - gt_pos[:, 1])**2)

    # Resmi yarışma skoru: MAE_3D — §9.2 Denklem 2
    # health=1'de referans gönderilir (sıfır hata), health=0'da tahmin
    est_tuples = [(est_x_arr[i], est_y_arr[i], est_z_arr[i]) for i in range(n_frames)]
    ref_tuples = [(float(gt_pos[i, 0]), float(gt_pos[i, 1]), 0.0) for i in range(n_frames)]
    # Optimal strateji: health=1'de referans → sıfır katkı
    # health=0'da tahmin hatası / N_toplam
    dead_err_sum = sum(
        math.sqrt((est_x_arr[i] - gt_pos[i, 0])**2 +
                  (est_y_arr[i] - gt_pos[i, 1])**2 +
                  est_z_arr[i]**2)
        for i in range(n_frames) if dead_mask[i]
    )
    mae_3d_official = dead_err_sum / n_frames  # payda = toplam frame (şartname gereği)
    mae_3d_dead     = (dead_err_sum / n_dead) if n_dead > 0 else float("nan")

    rmse_2d     = float(np.sqrt(np.mean(errs_2d[dead_mask]**2))) if n_dead > 0 else float("nan")
    final_drift = float(errs_2d[-1])
    max_drift   = float(np.max(errs_2d[dead_mask])) if n_dead > 0 else float("nan")

    # recovery-5 / recovery-10: kesinti başından itibaren N frame içinde <1m
    rec5_hits = rec10_hits = 0
    n_episodes = 0
    in_dead    = False
    dead_start = 0

    for i in range(n_frames):
        h = int(health[i])
        if not in_dead and h == 0:
            in_dead    = True
            dead_start = i
            n_episodes += 1
        elif in_dead and h == 1:
            in_dead = False
        elif in_dead and h == 0:
            frames_in = i - dead_start
            if errs_2d[i] < 1.0:
                if frames_in <= 5:
                    rec5_hits  += 1
                    rec10_hits += 1
                    in_dead = False
                elif frames_in <= 10:
                    rec10_hits += 1
                    in_dead = False

    recovery_5  = round(100.0 * rec5_hits  / max(n_episodes, 1), 1)
    recovery_10 = round(100.0 * rec10_hits / max(n_episodes, 1), 1)

    def _r(v):
        return round(v, 4) if not math.isnan(v) else "N/A"

    return {
        "label":            label,
        "health_scenario":  health_scenario,
        "n_frames":         n_frames,
        "motion_seed":      motion_seed,
        "health_seed":      health_seed,
        "n_dead":           n_dead,
        "n_dead_pct":       round(100.0 * n_dead / n_frames, 1),
        # ── Resmi yarışma skoru (§9.2 Denklem 2) ──────────────────────────────
        "mae_3d_official":  _r(mae_3d_official),  # health=1→ref, /N_toplam
        "mae_3d_dead":      _r(mae_3d_dead),       # yalnızca dead frame'ler
        # ── İç değerlendirme metrikleri ────────────────────────────────────────
        "rmse_2d":          _r(rmse_2d),
        "final_drift":      _r(final_drift),
        "max_drift":        _r(max_drift),
        "recovery_5_pct":   recovery_5,
        "recovery_10_pct":  recovery_10,
        "n_episodes":       n_episodes,
        "calib_ok":         1 if est.calibrated else 0,
        "calib_n":          est.calib_data_count,
        "fps":              round(fps, 1),
    }


# ── Tablo yazdırma ────────────────────────────────────────────────────────────

_HDR = (
    f"{'Label':<16} {'Scenario':<12} {'N':>5} "
    f"{'Dead%':>6} {'MAE_3D★':>8} {'RMSE_2D':>8} {'MaxDrift':>9} "
    f"{'Rec5%':>6} {'Rec10%':>7} {'CalOK':>6} {'FPS':>6}"
)
_SEP = "-" * len(_HDR)

# ★ MAE_3D = §9.2 Denklem 2 resmi yarışma skoru (health=1→ref gönder, /N_toplam)


def _fmt_row(r: dict) -> str:
    return (
        f"{r['label']:<16} {r['health_scenario']:<12} {r['n_frames']:>5} "
        f"{r['n_dead_pct']:>5.1f}% {str(r['mae_3d_official']):>8} {str(r['rmse_2d']):>8} "
        f"{str(r['max_drift']):>9} {r['recovery_5_pct']:>5.1f}% "
        f"{r['recovery_10_pct']:>6.1f}% {r['calib_ok']:>6} {r['fps']:>6.1f}"
    )


def print_summary(results: list) -> None:
    print()
    print("=" * len(_HDR))
    print("  VALIDATION SUITE SONUÇLARI")
    print("=" * len(_HDR))
    print(_HDR)
    print(_SEP)

    prev_scenario = None
    for r in results:
        if prev_scenario and r["health_scenario"] != prev_scenario:
            print(_SEP)
        prev_scenario = r["health_scenario"]
        print(_fmt_row(r))

    print(_SEP)

    # Aggregate: competition senaryosu özet
    comp = [r for r in results if r["health_scenario"] == "competition"]
    if comp:
        mae_vals  = [float(r["mae_3d_official"]) for r in comp if r["mae_3d_official"] != "N/A"]
        rmse_vals = [float(r["rmse_2d"])         for r in comp if r["rmse_2d"]         != "N/A"]
        fd_vals   = [float(r["final_drift"])     for r in comp]
        rec5_vals = [r["recovery_5_pct"]         for r in comp]
        print()
        print("  Competition senaryosu özet (n={})".format(len(comp)))
        print(f"  MAE_3D★   mean={np.mean(mae_vals):.4f}  std={np.std(mae_vals):.4f}  "
              f"min={np.min(mae_vals):.4f}  max={np.max(mae_vals):.4f}")
        print(f"  RMSE_2D   mean={np.mean(rmse_vals):.4f}  std={np.std(rmse_vals):.4f}")
        print(f"  FinalDrift mean={np.mean(fd_vals):.4f}  std={np.std(fd_vals):.4f}")
        print(f"  Recovery5% mean={np.mean(rec5_vals):.1f}%")
        print(f"  ★ MAE_3D = §9.2 Denklem 2 (resmi yarışma skoru)")

    print("=" * len(_HDR))


# ── Ana giriş noktası ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Final validation suite")
    parser.add_argument("--frames",  type=int,  default=None,
                        help="Tüm senaryoları bu frame sayısına sabitle (test için)")
    parser.add_argument("--out",     default=DEFAULT_OUT,
                        help="Çıktı CSV yolu")
    parser.add_argument("--no-csv",  action="store_true",
                        help="CSV çıktısını atla")
    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.frames:
        scenarios = [(l, hs, args.frames, ms, hse)
                     for l, hs, _, ms, hse in scenarios]

    results = []
    total   = len(scenarios)
    print(f"Validation suite başlıyor — {total} senaryo")
    print()

    for idx, (label, hs, nf, ms, hse) in enumerate(scenarios, 1):
        print(f"  [{idx:2d}/{total}] {label:<18} frames={nf}  "
              f"health={hs}  seed={ms}", end="", flush=True)
        r = run_scenario(label, hs, nf, ms, hse)
        results.append(r)
        print(f"  →  RMSE_2D={r['rmse_2d']}  fps={r['fps']}")

    print_summary(results)

    if not args.no_csv:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"\nKaydedildi: {args.out}")


if __name__ == "__main__":
    main()
