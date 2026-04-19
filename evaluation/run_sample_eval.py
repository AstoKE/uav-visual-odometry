#!/usr/bin/env python3
"""
run_sample_eval.py — Gerçek TEKNOFEST örnek verisi üzerinde OnlineEstimator değerlendirmesi

Veri seti:
    THYZ_2026_Ornek_Veri_1.MP4           — 1920×1080, 30fps, 9022 kare (~5 dk)
    THYZ_2026_Ornek_Veri_1_translation.csv — Kümülatif referans pozisyonları (x, y, z metre)

Değerlendirme akışı:
    1. Video kare kare yükle (opsiyonel --stride ile hızlandır)
    2. Referans pozisyonlarını CSV'den yükle (ilk kare = orijin)
    3. Health bayraklarını simüle et (ilk --calib_frames kare health=1)
    4. Her kare için OnlineEstimator.update() çalıştır
       • health=1 → referansı direkt kullan (sıfır hata — optimal strateji)
       • health=0 → tahmin çıktısını kullan
    5. Resmi MAE_3D skorunu hesapla (§9.2 Denklem 2)
    6. Sonuçları CSV + grafik olarak kaydet

Kullanım:
    python3 evaluation/run_sample_eval.py
    python3 evaluation/run_sample_eval.py \\
        --video ~/Downloads/THYZ_2026_Ornek_Veri_1.MP4 \\
        --csv   ~/Downloads/THYZ_2026_Ornek_Veri_1_translation.csv \\
        --stride 3 --calib_frames 450 --out evaluation/results_sample
    python3 evaluation/run_sample_eval.py --health all_dead   # health=0 = tam kör mod
    python3 evaluation/run_sample_eval.py --health comp       # competition simülasyonu

Kamera parametreleri:
    Video 1920×1080 içindir; --resize 640 360 ile yeniden boyutlandırılır.
    fx/fy değerleri yeniden boyutlandırma oranıyla otomatik ölçeklenir.
    Varsayılan: fx=fy=1100 (1920×1080 ~70° HFOV tahmini).

Çıktılar:
    results_sample/
        sample_eval_results.csv    — kare başına tahmin + hata
        sample_eval_trajectory.png — GT vs tahmin yörüngesi
        sample_eval_error.png      — zaman içinde hata grafiği
        sample_eval_summary.txt    — metrik özeti
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

_REPO = os.path.expanduser("~/code/uav-visual-odometry")
sys.path.insert(0, _REPO)

from competition.simulate_health import make_health_flags
from competition.score_official  import score_summary, health1_optimal_score

# ── Varsayılan yollar ─────────────────────────────────────────────────────────

DEFAULT_VIDEO  = os.path.expanduser("~/Downloads/THYZ_2026_Ornek_Veri_1.MP4")
DEFAULT_CSV    = os.path.expanduser("~/Downloads/THYZ_2026_Ornek_Veri_1_translation.csv")
DEFAULT_OUT    = os.path.join(_REPO, "evaluation/results_sample")

# ── CSV yükleme ───────────────────────────────────────────────────────────────

def load_reference_csv(csv_path: str) -> np.ndarray:
    """
    CSV'den kümülatif referans pozisyonlarını yükle.

    Sütunlar: translation_x, translation_y, translation_z, frame_numbers
    Döndürür: (N, 3) float64 array — ilk kare orijine normalize edilmiş
    """
    positions = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            positions.append([
                float(row["translation_x"]),
                float(row["translation_y"]),
                float(row["translation_z"]),
            ])
    pos = np.array(positions, dtype=np.float64)
    # İlk kare orijine normalize et
    pos -= pos[0]
    return pos


# ── Health flag simülasyonu ───────────────────────────────────────────────────

def make_health_array(n: int, mode: str, calib_frames: int, seed: int) -> np.ndarray:
    """
    Health bayrak dizisi oluştur.

    mode:
        "calib_only"  — sadece ilk calib_frames kare health=1, kalan health=0
        "comp"        — competition senaryosu (make_health_flags)
        "all_dead"    — tüm kareler health=0 (en kötü durum)
        "all_alive"   — tüm kareler health=1 (referans mevcut — skor=0)
    """
    flags = np.zeros(n, dtype=np.int8)

    if mode == "all_alive":
        flags[:] = 1
        return flags

    if mode == "all_dead":
        return flags  # hepsi 0

    # İlk calib_frames kare kesinlikle health=1 (§2.2.2)
    flags[:min(calib_frames, n)] = 1

    if mode == "calib_only":
        return flags

    if mode == "comp":
        remaining = n - calib_frames
        if remaining > 0:
            rest, _ = make_health_flags(remaining, seed=seed, scenario="competition")
            flags[calib_frames:] = rest

    return flags


# ── Ana değerlendirme döngüsü ─────────────────────────────────────────────────

def run_evaluation(
    video_path: str,
    ref_pos: np.ndarray,
    health: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    dist_coeffs: Optional[List[float]],
    resize_w: int,
    resize_h: int,
    altitude_m: Optional[float],
    stride: int,
    out_dir: str,
    backend: str = "orb",
    log_every: int = 100,
    kf_flow: float = 5.0,
) -> dict:
    """
    Video üzerinde OnlineEstimator çalıştır, metrikleri hesapla.

    stride > 1 ise hem video kare hem referans kare her stride'ıncısı alınır.

    Döndürür: metrik dict
    """
    os.makedirs(out_dir, exist_ok=True)

    # stride > 1 → alt-örnekleme
    n_total = len(ref_pos)
    if stride > 1:
        idx      = np.arange(0, n_total, stride)
        ref_pos  = ref_pos[idx]
        health   = health[idx]
        n_total  = len(ref_pos)
        print(f"[eval] stride={stride} → {n_total} kare değerlendirilecek")
    else:
        print(f"[eval] stride=1 → {n_total} kare değerlendirilecek")

    # Kamera ölçekleme (yeniden boyutlandırmaya göre)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[eval] Video: {orig_w}×{orig_h}, {total_vid_frames} kare")

    scale_x = resize_w / orig_w
    scale_y = resize_h / orig_h
    fx_sc   = fx * scale_x
    fy_sc   = fy * scale_y
    cx_sc   = cx * scale_x
    cy_sc   = cy * scale_y
    print(f"[eval] Yeniden boyut: {resize_w}×{resize_h}, "
          f"fx={fx_sc:.1f} fy={fy_sc:.1f} cx={cx_sc:.1f} cy={cy_sc:.1f}")

    # Distorsiyon katsayılarını ölçekle (undistort orijinal çözünürlükte,
    # sonra resize — cv2.undistort K ile çalışır, scale sonrası K da ölçeklenir)
    dist_arr = np.array(dist_coeffs, dtype=np.float64) if dist_coeffs else None

    # Estimator seç: droid veya orb (varsayılan)
    if backend == "droid":
        from competition.droid_estimator import DROIDEstimator
        est = DROIDEstimator(
            fx=fx_sc, fy=fy_sc, cx=cx_sc, cy=cy_sc,
            dist_coeffs=list(dist_arr) if dist_arr is not None else None,
            altitude_m=altitude_m,
        )
        print(f"[eval] Backend: {est.backend_name}")
    else:
        from competition.estimator import OnlineEstimator
        est = OnlineEstimator(
            fx=fx_sc, fy=fy_sc, cx=cx_sc, cy=cy_sc,
            dist_coeffs=dist_arr,
            altitude_m=altitude_m,
            n_features=1000,
            sim3_min_pairs=5,
            sim3_update_every=1,
            max_jump_m=8.0,
            min_keyframe_flow_px=kf_flow,
        )

    rows      = []
    est_pos   = np.zeros((n_total, 3), dtype=np.float64)
    frame_idx = 0         # video frame sayacı (okunan)
    eval_idx  = 0         # değerlendirme sayacı (stride uygulanmış)
    t0        = time.perf_counter()

    while eval_idx < n_total:
        ret, frame = cap.read()
        if not ret:
            print(f"[eval] UYARI: Video bitti — {eval_idx}/{n_total} kare işlendi")
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        # Yeniden boyutlandır
        if (frame.shape[1], frame.shape[0]) != (resize_w, resize_h):
            frame = cv2.resize(frame, (resize_w, resize_h))

        h_flag = int(health[eval_idx])
        rp     = (float(ref_pos[eval_idx, 0]),
                  float(ref_pos[eval_idx, 1]),
                  float(ref_pos[eval_idx, 2])) if h_flag == 1 else None

        # health=1 → optimal strateji: referans gönder (sıfır hata katkısı)
        if h_flag == 1 and rp is not None:
            est_x, est_y, est_z = rp
            est.update(frame, rp, h_flag)  # estimator'ı kalibre etmeye devam et
        else:
            est_x, est_y, est_z = est.update(frame, None, h_flag)

        est_pos[eval_idx] = [est_x, est_y, est_z]

        rows.append({
            "frame":  eval_idx * stride,
            "health": h_flag,
            "ref_x":  round(float(ref_pos[eval_idx, 0]), 4),
            "ref_y":  round(float(ref_pos[eval_idx, 1]), 4),
            "ref_z":  round(float(ref_pos[eval_idx, 2]), 4),
            "est_x":  round(est_x, 4),
            "est_y":  round(est_y, 4),
            "est_z":  round(est_z, 4),
        })

        eval_idx  += 1
        frame_idx += 1

        if eval_idx % log_every == 0:
            elapsed = time.perf_counter() - t0
            fps     = eval_idx / elapsed if elapsed > 0 else 0.0
            state   = est.get_state()
            print(f"  [{eval_idx:5d}/{n_total}]  "
                  f"fps={fps:.1f}  calib={state['calibrated']}  "
                  f"scale={state['sim3_scale']:.3f}  "
                  f"conf={state['confidence']:.2f}  "
                  f"rejected={state['drift_rejected']}")

    cap.release()
    n_proc = eval_idx

    elapsed = time.perf_counter() - t0
    fps     = n_proc / elapsed if elapsed > 0 else 0.0
    print(f"[eval] Tamamlandı: {n_proc} kare, {elapsed:.1f}s, {fps:.1f} fps")

    # ── Metrik hesapla ────────────────────────────────────────────────────────
    ref_proc  = ref_pos[:n_proc]
    est_proc  = est_pos[:n_proc]
    hlth_proc = health[:n_proc]

    est_tuples = [(float(est_proc[i, 0]), float(est_proc[i, 1]), float(est_proc[i, 2]))
                  for i in range(n_proc)]
    ref_tuples = [(float(ref_proc[i, 0]), float(ref_proc[i, 1]), float(ref_proc[i, 2]))
                  for i in range(n_proc)]

    sc = score_summary(est_tuples, ref_tuples, hlth_proc.tolist())

    # Resmi MAE_3D★: health=1'de ref gönderdik → sıfır hata
    # health=0'da tahmin → hata / N_toplam
    dead_idx  = [i for i in range(n_proc) if hlth_proc[i] == 0]
    dead_err  = sum(
        math.sqrt((est_proc[i,0]-ref_proc[i,0])**2 +
                  (est_proc[i,1]-ref_proc[i,1])**2 +
                  (est_proc[i,2]-ref_proc[i,2])**2)
        for i in dead_idx
    )
    mae_3d_official = dead_err / n_proc if n_proc > 0 else float("nan")
    mae_3d_dead     = dead_err / len(dead_idx) if dead_idx else float("nan")

    errs_2d = np.sqrt(
        (est_proc[:, 0] - ref_proc[:, 0])**2 +
        (est_proc[:, 1] - ref_proc[:, 1])**2
    )
    dead_mask = hlth_proc == 0
    rmse_2d   = float(np.sqrt(np.mean(errs_2d[dead_mask]**2))) if dead_mask.any() else float("nan")
    max_drift = float(np.max(errs_2d[dead_mask])) if dead_mask.any() else float("nan")

    final_state = est.get_state()
    metrics = {
        "n_frames":          n_proc,
        "n_dead":            len(dead_idx),
        "n_dead_pct":        round(100.0 * len(dead_idx) / n_proc, 1) if n_proc else 0.0,
        "stride":            stride,
        "fps":               round(fps, 1),
        # ── Resmi skor ──────────────────────────────────────────────────────
        "mae_3d_official":   round(mae_3d_official, 4),
        "mae_3d_dead":       round(mae_3d_dead, 4) if not math.isnan(mae_3d_dead) else "N/A",
        # ── İç metrikler ────────────────────────────────────────────────────
        "rmse_2d_dead":      round(rmse_2d, 4)     if not math.isnan(rmse_2d) else "N/A",
        "max_drift_2d":      round(max_drift, 4)   if not math.isnan(max_drift) else "N/A",
        "final_drift_2d":    round(float(errs_2d[-1]), 4) if n_proc else "N/A",
        # ── Estimator durumu ─────────────────────────────────────────────────
        "calibrated":        final_state["calibrated"],
        "sim3_pairs":        final_state["sim3_n_pairs"],
        "sim3_scale":        round(final_state["sim3_scale"], 4),
        "sim3_rmse":         round(final_state["sim3_rmse"], 4),
        "drift_rejected":    final_state["drift_rejected"],
        "confidence":        round(final_state["confidence"], 3),
    }

    # ── CSV kaydet ────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "sample_eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[eval] CSV: {csv_path}")

    # ── Özet metin ────────────────────────────────────────────────────────────
    _print_summary(metrics)
    txt_path = os.path.join(out_dir, "sample_eval_summary.txt")
    _save_summary(metrics, txt_path)

    # ── Grafikler ─────────────────────────────────────────────────────────────
    _plot_trajectory(ref_proc, est_proc, hlth_proc, n_proc, out_dir)
    _plot_error(errs_2d, hlth_proc, n_proc, out_dir, stride)

    return metrics


# ── Çıktı yardımcıları ────────────────────────────────────────────────────────

def _print_summary(m: dict) -> None:
    print()
    print("=" * 56)
    print("  ÖRNEK VERİ DEĞERLENDİRME SONUÇLARI")
    print("=" * 56)
    print(f"  Kare sayısı      : {m['n_frames']}  (stride={m['stride']})")
    print(f"  Health=0 kare    : {m['n_dead']} ({m['n_dead_pct']}%)")
    print(f"  İşleme hızı      : {m['fps']} fps")
    print(f"  {'─'*50}")
    print(f"  MAE_3D★ (resmi)  : {m['mae_3d_official']} m   ← §9.2 yarışma skoru")
    print(f"  MAE_3D (dead)    : {m['mae_3d_dead']} m")
    print(f"  RMSE_2D (dead)   : {m['rmse_2d_dead']} m")
    print(f"  Max drift 2D     : {m['max_drift_2d']} m")
    print(f"  Final drift 2D   : {m['final_drift_2d']} m")
    print(f"  {'─'*50}")
    print(f"  Kalibre          : {'Evet' if m['calibrated'] else 'Hayır'}"
          f"  (pairs={m['sim3_pairs']}, RMSE={m['sim3_rmse']})")
    print(f"  Sim(3) ölçek     : {m['sim3_scale']}")
    print(f"  Drift reddedilen : {m['drift_rejected']}")
    print(f"  Güven            : {m['confidence']}")
    print("=" * 56)


def _save_summary(m: dict, path: str) -> None:
    lines = [
        "ÖRNEK VERİ DEĞERLENDİRME SONUÇLARI",
        "=" * 40,
    ]
    for k, v in m.items():
        lines.append(f"{k:<22}: {v}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[eval] Özet: {path}")


def _plot_trajectory(ref_pos, est_pos, health, n_frames, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("UYARI: matplotlib yok, grafikler atlandı.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.plot(ref_pos[:, 0], ref_pos[:, 1],
            color="#2196F3", linewidth=1.0, alpha=0.7, label="Referans yörüngesi", zorder=2)

    h1_x = [est_pos[i, 0] for i in range(n_frames) if health[i] == 1]
    h1_y = [est_pos[i, 1] for i in range(n_frames) if health[i] == 1]
    h0_x = [est_pos[i, 0] for i in range(n_frames) if health[i] == 0]
    h0_y = [est_pos[i, 1] for i in range(n_frames) if health[i] == 0]

    if h1_x:
        ax.scatter(h1_x, h1_y, s=2, color="#4CAF50", alpha=0.5, label="Tahmin (health=1)", zorder=3)
    if h0_x:
        ax.scatter(h0_x, h0_y, s=2, color="#F44336", alpha=0.4, label="Tahmin (health=0)", zorder=3)

    ax.plot(ref_pos[0, 0], ref_pos[0, 1], "go", ms=10, label="Başlangıç", zorder=5)
    ax.plot(ref_pos[-1, 0], ref_pos[-1, 1], "rs", ms=10, label="Bitiş", zorder=5)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(f"Yörünge — Gerçek TEKNOFEST Örnek Verisi ({n_frames} kare)\n"
                 f"Mavi=GT, Yeşil=health=1 tahmin, Kırmızı=health=0 tahmin",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    out = os.path.join(out_dir, "sample_eval_trajectory.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] Yörünge: {out}")


def _plot_error(errs_2d, health, n_frames, out_dir, stride):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    frames = np.arange(n_frames) * stride

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1 = axes[0]
    # health=0 bölgelerini gölgele
    in_dead, dead_start = False, 0
    for i in range(n_frames):
        if not in_dead and health[i] == 0:
            in_dead    = True
            dead_start = frames[i]
        elif in_dead and (health[i] == 1 or i == n_frames - 1):
            ax1.axvspan(dead_start, frames[i], color="#FFCDD2", alpha=0.4)
            in_dead = False
    if in_dead:
        ax1.axvspan(dead_start, frames[-1], color="#FFCDD2", alpha=0.4)

    ax1.axhline(1.0, color="#FF5722", linestyle=":", linewidth=1.2, label="1.0 m eşiği", alpha=0.8)
    ax1.axhline(5.0, color="#9C27B0", linestyle=":", linewidth=1.0, label="5.0 m eşiği", alpha=0.7)
    ax1.plot(frames, errs_2d, color="#2196F3", linewidth=0.7, alpha=0.9, label="2D Hata (m)")

    ax1.set_ylabel("2D Hata (m)", fontsize=11)
    ax1.set_title(
        f"Konum Hatası — Gerçek TEKNOFEST Verisi ({n_frames} kare, stride={stride})\n"
        f"(Kırmızı gölge = health=0 bölümleri)",
        fontsize=11)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2 = axes[1]
    ax2.fill_between(frames, health.astype(float), 0,
                     where=health == 1, color="#4CAF50", alpha=0.6, label="health=1")
    ax2.fill_between(frames, 1, health.astype(float),
                     where=health == 0, color="#F44336", alpha=0.4, label="health=0")
    ax2.set_xlabel("Orijinal kare no", fontsize=11)
    ax2.set_ylabel("Health", fontsize=11)
    ax2.set_yticks([0, 1])
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "sample_eval_error.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] Hata grafiği: {out}")


# ── Giriş noktası ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gerçek TEKNOFEST örnek verisi üzerinde OnlineEstimator değerlendirmesi"
    )
    parser.add_argument("--video",   default=DEFAULT_VIDEO,
                        help="Video dosyası yolu (.MP4)")
    parser.add_argument("--csv",     default=DEFAULT_CSV,
                        help="Referans pozisyon CSV yolu")
    parser.add_argument("--out",     default=DEFAULT_OUT,
                        help="Çıktı dizini")
    parser.add_argument("--stride",  type=int, default=3,
                        help="Kare atlama adımı (1=tümü, 3=her 3. kare)")
    parser.add_argument("--calib_frames", type=int, default=450,
                        help="Garantili health=1 kalibrasyon karesi sayısı")
    parser.add_argument("--health",  default="comp",
                        choices=["calib_only", "comp", "all_dead", "all_alive"],
                        help="Health bayrağı modu")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Health simülasyonu rastgele tohumu")
    parser.add_argument("--fx",      type=float, default=1389.7,
                        help="Kamera x odak uzunluğu (1920×1080 için)")
    parser.add_argument("--fy",      type=float, default=1387.1,
                        help="Kamera y odak uzunluğu (1920×1080 için)")
    parser.add_argument("--cx",      type=float, default=954.0,
                        help="Kamera x merkez noktası")
    parser.add_argument("--cy",      type=float, default=558.9,
                        help="Kamera y merkez noktası")
    parser.add_argument("--dist",    type=float, nargs="+",
                        default=[0.1378, -0.2564, 0.0, 0.0],
                        help="Distorsiyon katsayıları k1 k2 p1 p2")
    parser.add_argument("--resize",  type=int, nargs=2, default=[640, 360],
                        metavar=("W", "H"),
                        help="İşleme için yeniden boyutlandırma (genişlik yükseklik)")
    parser.add_argument("--altitude", type=float, default=None,
                        help="Ortalama uçuş irtifası metre (bilinmiyorsa atla)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maksimum işlenecek orijinal kare sayısı (test için)")
    parser.add_argument("--backend",   default="orb",
                        choices=["orb", "droid"],
                        help="VO backend: orb (varsayılan) veya droid (CUDA gerektirir)")
    parser.add_argument("--log_every", type=int, default=100,
                        help="İlerleme log aralığı")
    parser.add_argument("--kf_flow", type=float, default=5.0,
                        help="min_keyframe_flow_px (varsayılan: 5.0)")
    args = parser.parse_args()

    # Veri yükle
    print(f"[eval] CSV yükleniyor: {args.csv}")
    ref_pos = load_reference_csv(args.csv)
    n_csv   = len(ref_pos)
    print(f"[eval] {n_csv} referans pozisyon yüklendi  "
          f"(X: {ref_pos[:,0].min():.1f}..{ref_pos[:,0].max():.1f} m, "
          f"Y: {ref_pos[:,1].min():.1f}..{ref_pos[:,1].max():.1f} m, "
          f"Z: {ref_pos[:,2].min():.1f}..{ref_pos[:,2].max():.1f} m)")

    # max_frames kısıtı
    n_use = min(n_csv, args.max_frames) if args.max_frames else n_csv
    ref_pos = ref_pos[:n_use]

    # Health bayrakları
    health = make_health_array(n_use, args.health, args.calib_frames, args.seed)
    n_dead = int((health == 0).sum())
    print(f"[eval] Health modu: {args.health}  "
          f"health=0: {n_dead}/{n_use} ({100*n_dead/n_use:.1f}%)")

    print(f"[eval] Kamera: fx={args.fx} fy={args.fy} cx={args.cx} cy={args.cy}")
    print(f"[eval] Distorsiyon: {args.dist}")
    if args.altitude:
        print(f"[eval] Irtifa: {args.altitude} m")

    # Değerlendirme
    run_evaluation(
        video_path=args.video,
        ref_pos=ref_pos,
        health=health,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        dist_coeffs=args.dist,
        resize_w=args.resize[0],
        resize_h=args.resize[1],
        altitude_m=args.altitude,
        stride=args.stride,
        out_dir=args.out,
        backend=args.backend,
        log_every=args.log_every,
        kf_flow=args.kf_flow,
    )


if __name__ == "__main__":
    main()
