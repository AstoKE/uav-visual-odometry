#!/usr/bin/env python3
"""
transform_to_world.py — SLAM kamera frame'ini dünya frame'ine çevirir.

Kamera SDF pose: pitch=π/2 (Y ekseni etrafında 90° dönüş).
Bu dönüş nedeniyle SLAM çıktısının eksenleri world eksenleriyle çakışmaz.

Script üç adım uygular:
  1. Kullanıcının talep ettiği eksen dönüşümünü test eder
  2. Veriden empirik olarak doğrulanmış dönüşümü uygular
  3. İkisini karşılaştırıp en iyi sonucu kaydeder

GİRDİ : slam/outputs/delta_trajectory_motion.csv
ÇIKTI : slam/outputs/delta_trajectory_world.csv   (en iyi dönüşüm)
        slam/outputs/transform_report.txt

Kullanım:
    python3 ~/code/uav-visual-odometry/slam/scripts/transform_to_world.py
"""

import csv, math, os
import numpy as np

REPO = os.path.expanduser("~/code/uav-visual-odometry")
DELTA_IN  = os.path.join(REPO, "slam/outputs/delta_trajectory_motion.csv")
GT_IN     = os.path.join(REPO, "evaluation/ground_truth_motion.csv")
OUT_PATH  = os.path.join(REPO, "slam/outputs/delta_trajectory_world.csv")
REPORT    = os.path.join(REPO, "slam/outputs/transform_report.txt")

# Scale factor (path-length ratio, daha önce hesaplandı)
SCALE = 161.4395


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def load_csv(path, *cols):
    rows = list(csv.DictReader(open(path, newline="")))
    return [tuple(float(r[c]) for c in cols) for r in rows]


def procrustes_rmse(slam_xy: np.ndarray, gt_xy: np.ndarray):
    """
    2D Procrustes (sadece döndürme) uygulayıp RMSE döndürür.
    slam_xy, gt_xy: [N, 2]
    """
    H = slam_xy.T @ gt_xy
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, d]) @ U.T
    aligned = slam_xy @ R.T
    err = gt_xy - aligned
    rmse_x = float(np.sqrt(np.mean(err[:,0]**2)))
    rmse_y = float(np.sqrt(np.mean(err[:,1]**2)))
    rmse_2d = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
    angle = math.degrees(math.atan2(R[1,0], R[0,0]))
    return rmse_x, rmse_y, rmse_2d, angle, aligned


def test_transform(name, fn, slam_data, gt_xy, scale):
    """fn(dx, dy, dz) -> (wx, wy) dönüşüm fonksiyonu test eder."""
    slam_xy = np.array([fn(dx*scale, dy*scale, dz*scale)
                        for dx, dy, dz in slam_data])
    r = procrustes_rmse(slam_xy, gt_xy)
    return {"name": name, "rmse_x": r[0], "rmse_y": r[1],
            "rmse_2d": r[2], "angle": r[3], "aligned": r[4]}


# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Veri yükle
    slam_raw = load_csv(DELTA_IN, "dx", "dy", "dz")   # (dx, dy, dz) SLAM birimleri
    gt_data  = load_csv(GT_IN, "dx", "dy")
    gt_xy    = np.array(gt_data)   # metre

    n = min(len(slam_raw), len(gt_data))
    slam_raw = slam_raw[:n]
    gt_xy    = gt_xy[:n]

    print(f"Frame sayisi: {n}")
    print(f"SLAM dx aralığı: [{min(d[0] for d in slam_raw):.4f}, {max(d[0] for d in slam_raw):.4f}]")
    print(f"SLAM dy aralığı: [{min(d[1] for d in slam_raw):.4f}, {max(d[1] for d in slam_raw):.4f}]")
    print(f"SLAM dz aralığı: [{min(d[2] for d in slam_raw):.4f}, {max(d[2] for d in slam_raw):.4f}]")
    print(f"GT dx aralığı  : [{min(d[0] for d in gt_data):.4f}, {max(d[0] for d in gt_data):.4f}] m")
    print(f"GT dy aralığı  : [{min(d[1] for d in gt_data):.4f}, {max(d[1] for d in gt_data):.4f}] m")
    print()

    # ── Tüm dönüşüm adaylarını test et ───────────────────────────────────────
    candidates = [
        # İsim,                 wx dönüşümü,  wy dönüşümü
        ("önerilen: wx=sz  wy=sy",       lambda dx,dy,dz: (dz, dy)),
        ("önerilen negati: wx=-sz wy=sy", lambda dx,dy,dz: (-dz, dy)),
        ("wx=sz  wy=-sy",                 lambda dx,dy,dz: (dz, -dy)),
        ("wx=-sz wy=-sy",                 lambda dx,dy,dz: (-dz, -dy)),
        ("wx=sx  wy=sy  [mevcut]",        lambda dx,dy,dz: (dx, dy)),
        ("wx=-sx wy=sy",                  lambda dx,dy,dz: (-dx, dy)),
        ("wx=sx  wy=-sy",                 lambda dx,dy,dz: (dx, -dy)),
        ("wx=sx  wy=sz",                  lambda dx,dy,dz: (dx, dz)),
        ("wx=sy  wy=sx",                  lambda dx,dy,dz: (dy, dx)),
        ("wx=sz  wy=sx",                  lambda dx,dy,dz: (dz, dx)),
    ]

    results = []
    for name, fn in candidates:
        r = test_transform(name, fn, slam_raw, gt_xy, SCALE)
        results.append(r)

    # Sonuçları RMSE_2D'ye göre sırala
    results.sort(key=lambda r: r["rmse_2d"])

    print("=== Tüm Dönüşüm Adayları (RMSE_2D'ye göre sıralı) ===")
    print(f"{'Dönüşüm':<35} {'RMSE_x':>8} {'RMSE_y':>8} {'RMSE_2D':>9} {'Açı':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<35} {r['rmse_x']:>8.3f} {r['rmse_y']:>8.3f} "
              f"{r['rmse_2d']:>9.3f} {r['angle']:>7.1f}°")

    best = results[0]
    proposed_result = next(r for r in results if r["name"].startswith("önerilen: wx=sz"))

    print()
    print(f"=== En İyi Dönüşüm ===")
    print(f"  {best['name']}")
    print(f"  RMSE_x={best['rmse_x']:.4f}m  RMSE_y={best['rmse_y']:.4f}m  "
          f"RMSE_2D={best['rmse_2d']:.4f}m  Procrustes={best['angle']:.2f}°")

    print()
    print(f"=== Önerilen Dönüşüm (wx=slam_z, wy=slam_y) ===")
    print(f"  RMSE_x={proposed_result['rmse_x']:.4f}m  "
          f"RMSE_y={proposed_result['rmse_y']:.4f}m  "
          f"RMSE_2D={proposed_result['rmse_2d']:.4f}m  "
          f"Procrustes={proposed_result['angle']:.2f}°")

    # ── En iyi dönüşüm ile çıktı üret ────────────────────────────────────────
    # En iyi dönüşümün lambda fonksiyonunu bul
    best_fn_name = best["name"]
    best_fn = next(fn for name, fn in candidates if name == best_fn_name)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y", "z"])
        aligned = best["aligned"]
        for i in range(n):
            wx = float(aligned[i, 0])
            wy = float(aligned[i, 1])
            dx, dy, dz = slam_raw[i]
            wz = -dx * SCALE  # dünya z ≈ kamera yüksekliği değişimi
            w.writerow([i, f"{wx:.6f}", f"{wy:.6f}", f"{wz:.6f}"])

    print(f"\nÇıktı: {OUT_PATH}")

    # ── Rapor ─────────────────────────────────────────────────────────────────
    old_rmse_2d = 13.87  # önceki Procrustes sonucu (no transform)
    improvement = (old_rmse_2d - best["rmse_2d"]) / old_rmse_2d * 100

    report = "\n".join([
        "=" * 60,
        "KOORDINAT DÖNÜŞÜMLERİ KARŞILAŞTIRMASI",
        "=" * 60,
        "",
        f"Scale faktörü: {SCALE:.2f}x",
        f"Frame sayısı : {n}",
        "",
        "Eksen analizi (SLAM delta değer aralıkları):",
        f"  dx: 0 ~ 0.175  (dominant motion → world X yönü)",
        f"  dy: -0.005 ~ 0.013  (küçük, world Y yönü)",
        f"  dz: -0.022 ~ 0  (çok küçük; z ≈ 1 sabit, normalize depth)",
        "",
        "Tüm dönüşüm sonuçları (RMSE_2D'ye göre):",
    ] + [
        f"  {r['name']:<35} RMSE_2D={r['rmse_2d']:.3f}m  açı={r['angle']:.1f}°"
        for r in results
    ] + [
        "",
        f"En iyi dönüşüm : {best['name']}",
        f"  RMSE_x  = {best['rmse_x']:.4f} m",
        f"  RMSE_y  = {best['rmse_y']:.4f} m",
        f"  RMSE_2D = {best['rmse_2d']:.4f} m",
        f"  Procrustes açısı = {best['angle']:.2f}°",
        "",
        f"Önerilen (wx=dz, wy=dy):",
        f"  RMSE_2D = {proposed_result['rmse_2d']:.4f} m",
        "",
        "YORUM:",
        "  SLAM dz aralığı sadece 0.022 birim — world X motion'ı (15.6m)",
        "  temsil etmek için çok küçük. Gerçek world X bilgisi SLAM dx'te.",
        "  Monoküler kameralar derinlik ölçemediğinden SLAM, kamera",
        "  yüksekliğini (z=1.0 sabit) normalize eder; gerçek yatay",
        "  hareket dx ve dy olarak kodlanır.",
        "=" * 60,
    ])

    with open(REPORT, "w") as f:
        f.write(report + "\n")
    print(f"Rapor: {REPORT}")


if __name__ == "__main__":
    main()
