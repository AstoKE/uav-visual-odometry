#!/usr/bin/env python3
"""
run_teknofest_test.py — Teknofest İkinci Görev test pipeline'ı

Gerçek Gazebo görüntüleri üzerinde tam pipeline testi:
  dataset/small_motion_v3/   (900 kare, 640×480)
  evaluation/ground_truth_figure8_v3_frames900.csv

Health yapısı (PDF'e uygun oran):
  Frame 0–299   → health=1  (kalibrasyon penceresi, ~%33)
  Frame 300–899 → health=0  (GPS-siz mod, ~%67)

  Yarışma oranı: 450:1800 = 1:4
  Test oranı   : 300:600  = 1:2  (aynı algoritma, farklı süre)

Pipeline (PDF Bölüm 3.1):
  Görüntü → ORB → Essential Matrix → R,t → Sim(3) → ΔX,ΔY,ΔZ

Metrikler (PDF Bölüm 7):
  RMSE_x, RMSE_y, RMSE_z
  RMSE_2D, RMSE_3D
  ATE (Absolute Trajectory Error)
  Final drift
  X-Y-Z eksen grafikleri

Kullanım:
    python3 competition/run_teknofest_test.py
    python3 competition/run_teknofest_test.py --calib_frames 300 --images dataset/small_motion_v3
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

_REPO = os.path.expanduser("~/code/uav-visual-odometry")
sys.path.insert(0, _REPO)

from competition.estimator import OnlineEstimator

# ── Varsayılan yollar ─────────────────────────────────────────────────────────

IMG_DIR   = os.path.join(_REPO, "dataset/small_motion_v3")
GT_CSV    = os.path.join(_REPO, "evaluation/ground_truth_figure8_v3_frames900.csv")
CALIB_TXT = os.path.join(_REPO, "dataset/meta/calib.txt")
OUT_DIR   = os.path.join(_REPO, "competition/results")


# ── Veri yükleyiciler ─────────────────────────────────────────────────────────

def load_gt(path: str) -> dict[int, tuple]:
    """GT CSV → frame → (x, y, z) sözlüğü. dx/dy/dz kullanılır (başlangıçtan ofset)."""
    gt = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            frame = int(r["frame"])
            # dx, dy, dz = başlangıç pozisyonundan yer değiştirme (= x0=0 olduğu için x,y,z ile aynı)
            gt[frame] = (float(r["dx"]), float(r["dy"]), float(r.get("dz", 0.0)))
    return gt


def load_calib(path: str) -> tuple[float, float, float, float]:
    vals = list(map(float, open(path).read().split()))
    return vals[0], vals[1], vals[2], vals[3]  # fx, fy, cx, cy


def list_images(img_dir: str) -> list[str]:
    files = sorted(
        f for f in os.listdir(img_dir)
        if f.endswith((".png", ".jpg"))
    )
    return [os.path.join(img_dir, f) for f in files]


# ── Metrik hesaplama (PDF Bölüm 7) ────────────────────────────────────────────

def compute_metrics(est_arr: np.ndarray, gt_arr: np.ndarray,
                    health: np.ndarray) -> dict:
    """
    PDF Bölüm 7 metrikleri:
      RMSE_x, RMSE_y, RMSE_z
      RMSE_2D, RMSE_3D
      ATE (Absolute Trajectory Error) — health=0 bölümünde
      Final drift
    """
    dead = health == 0

    ex = est_arr[:, 0] - gt_arr[:, 0]
    ey = est_arr[:, 1] - gt_arr[:, 1]
    ez = est_arr[:, 2] - gt_arr[:, 2]
    e2d = np.sqrt(ex ** 2 + ey ** 2)
    e3d = np.sqrt(ex ** 2 + ey ** 2 + ez ** 2)

    def rmse(arr): return float(np.sqrt(np.mean(arr ** 2)))

    # RMSE — tüm kareler
    rmse_x_all  = rmse(ex)
    rmse_y_all  = rmse(ey)
    rmse_2d_all = rmse(e2d)

    # RMSE — sadece health=0 (GPS-siz mod)
    if dead.sum() > 0:
        rmse_x  = rmse(ex[dead])
        rmse_y  = rmse(ey[dead])
        rmse_z  = rmse(ez[dead])
        rmse_2d = rmse(e2d[dead])
        rmse_3d = rmse(e3d[dead])
        ate     = float(np.mean(e2d[dead]))   # ATE = ortalama mutlak yörünge hatası
        max_drift = float(np.max(e2d[dead]))
    else:
        rmse_x = rmse_y = rmse_z = rmse_2d = rmse_3d = ate = max_drift = float("nan")

    final_drift = float(e2d[-1])

    return {
        # Tüm kareler
        "rmse_x_all":  rmse_x_all,
        "rmse_y_all":  rmse_y_all,
        "rmse_2d_all": rmse_2d_all,
        # GPS-siz mod (health=0)
        "rmse_x":    rmse_x,
        "rmse_y":    rmse_y,
        "rmse_z":    rmse_z,
        "rmse_2d":   rmse_2d,
        "rmse_3d":   rmse_3d,
        "ate":       ate,
        "max_drift": max_drift,
        "final_drift": final_drift,
    }


# ── Ana test döngüsü ──────────────────────────────────────────────────────────

def _detect_motion_start(img_paths: list[str], threshold: float = 2.0) -> int:
    """
    İlk hareket eden kareyi tespit et (GT ile hizalamak için).

    Ardışık görüntüler arasındaki ortalama piksel farkı > threshold ise
    hareket başlamış sayılır.
    """
    if len(img_paths) < 2:
        return 0
    img0 = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE).astype(float)
    for i in range(1, len(img_paths)):
        img = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE).astype(float)
        if np.mean(np.abs(img - img0)) > threshold:
            print(f"[Dataset] Hareket frame {i}'de başlıyor (piksel farkı > {threshold})")
            return i
    return 0


def run_test(
    img_dir: str      = IMG_DIR,
    gt_csv: str       = GT_CSV,
    calib_txt: str    = CALIB_TXT,
    calib_frames: int = 150,
    gt_offset: int    = 390,   # GT'de sıfırlanan frame numarası
    out_dir: str      = OUT_DIR,
) -> dict:
    """
    Dataset notu:
      small_motion_v3: İlk ~360 kare özdeş (kamera hareketsiz).
      GT (frame 390): figure-8 yeniden sıfırlanır.
      Bu fonksiyon hareketsiz başlangıcı atlayarak GT ile hizalama yapar.

      Kullanılan kareler: img[motion_start .. motion_start+N]
      GT eşlemesi      : image[i] → GT[i - motion_start + gt_offset]
    """
    # Kamera parametreleri
    fx, fy, cx, cy = load_calib(calib_txt)
    print(f"Kamera : fx={fx} fy={fy} cx={cx} cy={cy}")

    # Görüntüler
    all_imgs = list_images(img_dir)
    if not all_imgs:
        raise FileNotFoundError(f"Görüntü bulunamadı: {img_dir}")

    # Hareketsiz başlangıcı atla
    motion_start = _detect_motion_start(all_imgs)
    img_paths = all_imgs[motion_start:]

    # GT
    gt_full = load_gt(gt_csv)
    if not gt_full:
        raise FileNotFoundError(f"GT bulunamadı: {gt_csv}")

    # GT hizalama: image[i] → GT[i + gt_offset]
    # GT sıfırlandığı frame'den itibaren bağıl pozisyon al
    gt_origin = gt_full.get(gt_offset, (0.0, 0.0, 0.0))

    def get_gt(img_idx: int) -> tuple[float, float, float] | None:
        gt_frame = img_idx + gt_offset
        if gt_frame in gt_full:
            gx, gy, gz = gt_full[gt_frame]
            # Bağıl pozisyon (başlangıç noktasına göre)
            return gx - gt_origin[0], gy - gt_origin[1], gz - gt_origin[2]
        return None

    # Kaç kare var?
    max_frames = min(len(img_paths),
                     max(k for k in gt_full if k >= gt_offset) - gt_offset + 1)
    img_paths = img_paths[:max_frames]
    n = len(img_paths)

    print(f"Görüntü: {len(all_imgs)} toplam  →  {n} kullanılıyor "
          f"(frame {motion_start}–{motion_start+n-1})")
    print(f"GT     : {len(gt_full)} satır  GT offset={gt_offset}")

    # Health yapısı (PDF oranı: ~1:4, test: 1:2.4)
    health = np.zeros(n, dtype=np.int8)
    health[:calib_frames] = 1
    n_alive = int((health == 1).sum())
    n_dead  = int((health == 0).sum())
    print(f"\nHealth : {n_alive} kare health=1 (kalibrasyon)  |  {n_dead} kare health=0 (GPS-siz)")
    print(f"Oran   : {n_alive}/{n} = {100*n_alive/n:.0f}% kalibrasyon  "
          f"(Teknofest: 450/2250 = 20%)")

    # Estimator (PDF Bölüm 3.1 mimarisi)
    # stride=3: bu dataset 15fps'de kaydedilmiş, her 3 frame'de gerçek hareket var
    # Gerçek yarışmada (7.5fps, hızlı UAV) stride=1 yeterli olacak
    # min_keyframe_flow_px=5.0: yeni keyframe için minimum LK hareket eşiği
    # Bu dataset 15fps, step-wise hareket — 5px eşiği ~3-5 frame gecikmeli keyframe üretir
    est = OnlineEstimator(
        fx=fx, fy=fy, cx=cx, cy=cy,
        n_features=1500,
        lowe_ratio=0.75,
        ransac_thresh=2.0,
        ema_alpha=0.7,
        max_jump_m=3.0,
        sim3_min_pairs=10,
        sim3_update_every=25,
        min_keyframe_flow_px=5.0,
    )

    est_arr = np.zeros((n, 3))
    gt_arr  = np.zeros((n, 3))
    rows    = []
    t_start = time.perf_counter()

    print(f"\nPipeline çalışıyor...\n{'─'*60}")

    for i, img_path in enumerate(img_paths):
        frame   = cv2.imread(img_path)
        h       = int(health[i])
        gt_pos  = get_gt(i)
        ref_pos = (gt_pos[0], gt_pos[1], gt_pos[2]) if (h == 1 and gt_pos is not None) else None

        ex_i, ey_i, ez_i = est.update(frame, ref_pos, h)

        est_arr[i] = [ex_i, ey_i, ez_i]
        gt_arr[i]  = list(gt_pos) if gt_pos is not None else [ex_i, ey_i, ez_i]

        rows.append({
            "frame":   i,
            "health":  h,
            "gt_x":    round(gt_arr[i, 0], 4),
            "gt_y":    round(gt_arr[i, 1], 4),
            "gt_z":    round(gt_arr[i, 2], 4),
            "est_x":   round(ex_i, 4),
            "est_y":   round(ey_i, 4),
            "est_z":   round(ez_i, 4),
            "inliers": est._last_inliers,
            "sim3_ok": 1 if est.calibrated else 0,
        })

        if i % 50 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t_start
            fps     = (i + 1) / elapsed if elapsed > 0 else 0
            e2d_now = float(np.sqrt((ex_i - gt_arr[i, 0]) ** 2 +
                                    (ey_i - gt_arr[i, 1]) ** 2))
            print(
                f"  [{i+1:4d}/{n}]  h={h}  "
                f"inl={est._last_inliers:3d}  "
                f"calib={'OK' if est.calibrated else '--'}  "
                f"err2d={e2d_now:6.3f}m  fps={fps:6.1f}"
            )

    elapsed = time.perf_counter() - t_start
    fps = n / elapsed

    # Metrikler
    metrics = compute_metrics(est_arr, gt_arr, health)
    metrics["fps"]         = round(fps, 1)
    metrics["n_frames"]    = n
    metrics["n_calib"]     = n_alive
    metrics["n_gps_denied"]= n_dead
    metrics["calib_ok"]    = 1 if est.calibrated else 0
    metrics["sim3_scale"]  = round(est._sim3.scale, 5)
    metrics["sim3_rmse"]   = round(est._sim3.rmse_calib, 5) if est.calibrated else float("nan")
    metrics["drift_rejected"] = est.drift_rejected

    # Çıktılar
    os.makedirs(out_dir, exist_ok=True)
    _save_csv(rows, os.path.join(out_dir, "teknofest_test_results.csv"))
    _print_report(metrics)
    _plot_results(est_arr, gt_arr, health, n, out_dir)

    return metrics


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def _save_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV kaydedildi: {path}")


def _print_report(m: dict) -> None:
    nan_str = lambda v: f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"
    print()
    print("=" * 58)
    print("  Teknofest İkinci Görev — Performans Raporu (PDF §7)")
    print("=" * 58)
    print(f"  Toplam kare          : {m['n_frames']}")
    print(f"  Kalibrasyon (health=1): {m['n_calib']}")
    print(f"  GPS-siz (health=0)   : {m['n_gps_denied']}")
    print(f"  Kalibrasyon başarılı : {'Evet' if m['calib_ok'] else 'Hayır'}")
    print(f"  Sim(3) ölçek         : {nan_str(m['sim3_scale'])}")
    print(f"  Sim(3) RMSE          : {nan_str(m['sim3_rmse'])} m")
    print()
    print("  ── Tüm kareler ──────────────────────────────────────")
    print(f"  RMSE_x (tüm)         : {nan_str(m['rmse_x_all'])} m")
    print(f"  RMSE_y (tüm)         : {nan_str(m['rmse_y_all'])} m")
    print(f"  RMSE_2D (tüm)        : {nan_str(m['rmse_2d_all'])} m")
    print()
    print("  ── GPS-siz mod (health=0) ───────────────────────────")
    print(f"  RMSE_x               : {nan_str(m['rmse_x'])} m")
    print(f"  RMSE_y               : {nan_str(m['rmse_y'])} m")
    print(f"  RMSE_z               : {nan_str(m['rmse_z'])} m")
    print(f"  RMSE_2D              : {nan_str(m['rmse_2d'])} m")
    print(f"  RMSE_3D              : {nan_str(m['rmse_3d'])} m")
    print(f"  ATE                  : {nan_str(m['ate'])} m")
    print(f"  Max drift            : {nan_str(m['max_drift'])} m")
    print(f"  Final drift          : {nan_str(m['final_drift'])} m")
    print()
    print(f"  FPS                  : {m['fps']}")
    print(f"  Drift rejected       : {m['drift_rejected']}")
    print("=" * 58)


def _plot_results(est_arr, gt_arr, health, n, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib yok — grafikler atlandı.")
        return

    frames    = np.arange(n)
    dead_mask = health == 0
    e2d = np.sqrt((est_arr[:, 0] - gt_arr[:, 0]) ** 2 +
                  (est_arr[:, 1] - gt_arr[:, 1]) ** 2)
    calib_end = int((health == 1).sum())

    # ── Plot 1: Yörünge (XY) ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(gt_arr[:, 0],  gt_arr[:, 1],
            color="#2196F3", lw=1.8, alpha=0.8, label="GT Yörüngesi")
    h1 = health == 1
    h0 = health == 0
    ax.scatter(est_arr[h1, 0], est_arr[h1, 1],
               s=4, color="#4CAF50", alpha=0.7, label="Tahmin (health=1)")
    ax.scatter(est_arr[h0, 0], est_arr[h0, 1],
               s=4, color="#F44336", alpha=0.5, label="Tahmin (health=0)")
    ax.plot(gt_arr[0, 0], gt_arr[0, 1], "go", ms=10, label="Başlangıç")
    ax.plot(gt_arr[-1, 0], gt_arr[-1, 1], "rs", ms=10, label="Bitiş")
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("Teknofest İkinci Görev — XY Yörüngesi\n"
                 "ORB + Essential Matrix + Sim(3) Alignment", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "teknofest_trajectory.png"), dpi=150)
    plt.close()

    # ── Plot 2: Hata grafikleri (PDF §7 metrikleri) ───────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 2a. X ekseni hatası
    ax = axes[0]
    _shade_dead(ax, health, n)
    ax.axvline(calib_end, color="#9C27B0", ls="--", lw=1.2, label=f"Calib sonu ({calib_end})")
    ax.plot(frames, est_arr[:, 0] - gt_arr[:, 0],
            color="#2196F3", lw=0.9, label="X hatası (m)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("ΔX hatası (m)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)

    # 2b. Y ekseni hatası
    ax = axes[1]
    _shade_dead(ax, health, n)
    ax.axvline(calib_end, color="#9C27B0", ls="--", lw=1.2)
    ax.plot(frames, est_arr[:, 1] - gt_arr[:, 1],
            color="#FF9800", lw=0.9, label="Y hatası (m)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("ΔY hatası (m)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)

    # 2c. 2D toplam hata
    ax = axes[2]
    _shade_dead(ax, health, n)
    ax.axvline(calib_end, color="#9C27B0", ls="--", lw=1.2)
    ax.axhline(1.0, color="#F44336", ls=":", lw=1.2, alpha=0.8, label="1.0 m eşiği")
    ax.plot(frames, e2d, color="#E91E63", lw=0.9, label="2D Hata (m)")
    ax.set_ylabel("2D Hata (m)", fontsize=10)
    ax.set_xlabel("Kare", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)

    fig.suptitle(
        "Teknofest İkinci Görev — X/Y/2D Hata Grafikleri\n"
        "(Kırmızı gölge = health=0  |  Mor kesik = kalibrasyon sonu)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "teknofest_errors.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Grafikler: {out_dir}/teknofest_trajectory.png")
    print(f"           {out_dir}/teknofest_errors.png")


def _shade_dead(ax, health, n):
    """health=0 bölgelerini kırmızı şeffaf gölge ile göster."""
    in_dead = False
    ds = 0
    for i in range(n):
        if not in_dead and health[i] == 0:
            in_dead = True; ds = i
        elif in_dead and (health[i] == 1 or i == n - 1):
            ax.axvspan(ds, i, color="#FFCDD2", alpha=0.5)
            in_dead = False
    if in_dead:
        ax.axvspan(ds, n, color="#FFCDD2", alpha=0.5)


# ── Giriş noktası ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Teknofest İkinci Görev — Gerçek görüntülerle test"
    )
    parser.add_argument("--images",       default=IMG_DIR)
    parser.add_argument("--gt",           default=GT_CSV)
    parser.add_argument("--calib",        default=CALIB_TXT)
    parser.add_argument("--calib_frames", type=int, default=150,
                        help="health=1 kare sayısı (kalibrasyon penceresi)")
    parser.add_argument("--gt_offset",    type=int, default=390,
                        help="GT'de sıfırlama frame numarası (dataset'e göre)")
    parser.add_argument("--out",          default=OUT_DIR)
    args = parser.parse_args()

    run_test(
        img_dir=args.images,
        gt_csv=args.gt,
        calib_txt=args.calib,
        calib_frames=args.calib_frames,
        gt_offset=args.gt_offset,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
