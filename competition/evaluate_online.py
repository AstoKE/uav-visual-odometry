#!/usr/bin/env python3
"""
evaluate_online.py — DROID vs ORB karşılaştırmalı online evaluasyon

Hesaplanan metrikler:
  RMSE_x, RMSE_y, RMSE_2D, max_drift (health=0 bölümünde)
  Final drift (son frame hatası)
  Recovery error (health=0 başladıktan sonra ilk 5 ve 10 frame)
  Health=0 segmentleri ayrı ayrı değerlendirilir

Grafikler (--plots ile):
  trajectory_overlay.png  — GT vs DROID vs ORB yörünge karşılaştırması
  error_vs_time.png       — Kareye göre hata + health=0 bölge vurgusu

Kullanım:
    python3 competition/evaluate_online.py
    python3 competition/evaluate_online.py --droid results/est_droid.csv --orb results/est_orb.csv
    python3 competition/evaluate_online.py --plots
"""

import argparse
import csv
import os
import sys
import numpy as np

_REPO = os.path.expanduser("~/code/uav-visual-odometry")
sys.path.insert(0, _REPO)

RESULTS_DIR = os.path.join(_REPO, "competition/results")
OUT_TXT     = os.path.join(RESULTS_DIR, "results_online.txt")


# ── Yükleyiciler ──────────────────────────────────────────────────────────────

def load_est(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "frame":  int(row["frame"]),
                "health": int(row["health"]),
                "gt_x":   float(row["gt_x"]),
                "gt_y":   float(row["gt_y"]),
                "est_x":  float(row["est_x"]),
                "est_y":  float(row["est_y"]),
            })
    return rows


# ── Metrik hesaplamaları ──────────────────────────────────────────────────────

def compute_metrics(rows: list[dict]) -> dict:
    """Tüm metrikleri hesapla."""
    all_arr = np.array([(r["gt_x"], r["gt_y"], r["est_x"], r["est_y"]) for r in rows])
    dead = [r for r in rows if r["health"] == 0]

    # Genel RMSE (health=0)
    if dead:
        dead_arr = np.array([(r["gt_x"], r["gt_y"], r["est_x"], r["est_y"]) for r in dead])
        err_x = dead_arr[:, 2] - dead_arr[:, 0]
        err_y = dead_arr[:, 3] - dead_arr[:, 1]
        rmse_x  = float(np.sqrt(np.mean(err_x**2)))
        rmse_y  = float(np.sqrt(np.mean(err_y**2)))
        rmse_2d = float(np.sqrt(np.mean(err_x**2 + err_y**2)))
        rmse_3d = rmse_2d  # z sabit (z_est≈0, z_gt≈0)
    else:
        rmse_x = rmse_y = rmse_2d = rmse_3d = float("nan")

    # Final drift (son frame)
    last = rows[-1]
    final_drift = float(np.sqrt(
        (last["est_x"] - last["gt_x"])**2 + (last["est_y"] - last["gt_y"])**2
    ))

    # Recovery: health geçişlerini bul (health=1→0 veya 0→1)
    recovery_5  = []
    recovery_10 = []
    for i in range(1, len(rows)):
        prev_h = rows[i-1]["health"]
        curr_h = rows[i]["health"]
        if prev_h == 1 and curr_h == 0:
            # Yeni health=0 bölümü başlıyor
            window_5  = rows[i:i+5]
            window_10 = rows[i:i+10]
            if len(window_5) == 5:
                e5 = np.array([
                    np.sqrt((r["est_x"]-r["gt_x"])**2 + (r["est_y"]-r["gt_y"])**2)
                    for r in window_5
                ])
                recovery_5.append(float(np.mean(e5)))
            if len(window_10) == 10:
                e10 = np.array([
                    np.sqrt((r["est_x"]-r["gt_x"])**2 + (r["est_y"]-r["gt_y"])**2)
                    for r in window_10
                ])
                recovery_10.append(float(np.mean(e10)))

    rec5  = float(np.mean(recovery_5))  if recovery_5  else float("nan")
    rec10 = float(np.mean(recovery_10)) if recovery_10 else float("nan")

    # Max drift (health=0 bölümünde en kötü frame)
    if dead:
        dead_errors = np.sqrt(
            (dead_arr[:, 2] - dead_arr[:, 0])**2 +
            (dead_arr[:, 3] - dead_arr[:, 1])**2
        )
        max_drift = float(dead_errors.max())
    else:
        max_drift = float("nan")

    # Health=0 segmentleri ayrı ayrı
    h0_segments: list[dict] = []
    i = 0
    while i < len(rows):
        if rows[i]["health"] == 0:
            seg_start = i
            seg_rows = []
            while i < len(rows) and rows[i]["health"] == 0:
                seg_rows.append(rows[i])
                i += 1
            seg_errors = np.array([
                np.sqrt((r["est_x"]-r["gt_x"])**2 + (r["est_y"]-r["gt_y"])**2)
                for r in seg_rows
            ])
            h0_segments.append({
                "start":   seg_rows[0]["frame"],
                "end":     seg_rows[-1]["frame"],
                "length":  len(seg_rows),
                "rmse":    float(np.sqrt(np.mean(seg_errors**2))),
                "max_err": float(seg_errors.max()),
                "mean_err":float(seg_errors.mean()),
            })
        else:
            i += 1

    # Geçiş noktalarında anlık hata
    transition_errors = []
    for i in range(1, len(rows)):
        if rows[i-1]["health"] == 1 and rows[i]["health"] == 0:
            e = np.sqrt((rows[i]["est_x"] - rows[i]["gt_x"])**2 +
                        (rows[i]["est_y"] - rows[i]["gt_y"])**2)
            transition_errors.append(float(e))

    return {
        "n_frames":       len(rows),
        "n_dead":         len(dead),
        "rmse_x":         rmse_x,
        "rmse_y":         rmse_y,
        "rmse_2d":        rmse_2d,
        "rmse_3d":        rmse_3d,
        "max_drift":      max_drift,
        "final_drift":    final_drift,
        "recovery_5":     rec5,
        "recovery_10":    rec10,
        "n_transitions":  len(transition_errors),
        "h0_segments":    h0_segments,
    }


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if not np.isnan(v) else "N/A"
    return str(v)


# ── Karar mantığı ─────────────────────────────────────────────────────────────

def select_model(dm: dict, om: dict) -> tuple[str, str]:
    """
    Karar mantığı:
      IF RMSE farkı > %15 → daha iyi olan
      ELSE IF recovery farkı > %15 → daha iyi recovery
      ELSE → daha stabil (düşük final_drift)
    """
    d_rmse = dm["rmse_2d"]
    o_rmse = om["rmse_2d"]

    if np.isnan(d_rmse) or np.isnan(o_rmse):
        return "DROID", "RMSE hesaplanamadı — varsayılan DROID"

    rmse_diff = abs(d_rmse - o_rmse) / max(d_rmse, o_rmse)
    if rmse_diff > 0.15:
        winner = "DROID" if d_rmse < o_rmse else "ORB"
        return winner, f"RMSE farkı %{100*rmse_diff:.1f} > %15 — {winner} daha iyi RMSE"

    d_rec = dm["recovery_5"]
    o_rec = om["recovery_5"]
    if not np.isnan(d_rec) and not np.isnan(o_rec):
        rec_diff = abs(d_rec - o_rec) / max(d_rec, o_rec)
        if rec_diff > 0.15:
            winner = "DROID" if d_rec < o_rec else "ORB"
            return winner, f"Recovery farkı %{100*rec_diff:.1f} > %15 — {winner} daha hızlı toparlanma"

    # Stabilite: düşük final_drift
    winner = "DROID" if dm["final_drift"] <= om["final_drift"] else "ORB"
    return winner, f"RMSE benzer — düşük drift seçildi (DROID={dm['final_drift']:.3f} ORB={om['final_drift']:.3f})"


def format_report(
    droid_m: dict,
    orb_m: dict,
    droid_rt: dict | None,
    orb_rt: dict | None,
    final_model: str,
    reason: str,
) -> str:
    rows_def = [
        ("RMSE_x (m)",       "rmse_x"),
        ("RMSE_y (m)",       "rmse_y"),
        ("RMSE_2D (m)",      "rmse_2d"),
        ("RMSE_3D (m)",      "rmse_3d"),
        ("Max Drift (m)",    "max_drift"),
        ("Final Drift (m)",  "final_drift"),
        ("Recovery-5 (m)",   "recovery_5"),
        ("Recovery-10 (m)",  "recovery_10"),
        ("Health=0 frames",  "n_dead"),
        ("Transitions",      "n_transitions"),
    ]

    lines = [
        "=" * 60,
        "DROID vs ORB — Online Estimator Karşılaştırma",
        "=" * 60,
        "",
        f"{'Metrik':<22} {'DROID':>12} {'ORB':>12} {'Daha İyi':>10}",
        "-" * 60,
    ]
    for label, key in rows_def:
        dv = droid_m.get(key, float("nan"))
        ov = orb_m.get(key, float("nan"))
        if isinstance(dv, float) and isinstance(ov, float) and not np.isnan(dv) and not np.isnan(ov):
            if key in ("n_dead", "n_transitions"):
                better = ""
            elif dv < ov:
                better = "DROID ✓"
            elif ov < dv:
                better = "ORB ✓"
            else:
                better = "Eşit"
        else:
            better = ""
        lines.append(f"{label:<22} {fmt(dv):>12} {fmt(ov):>12} {better:>10}")

    if droid_rt or orb_rt:
        lines += ["", "Runtime:", "-" * 40]
        dfps = droid_rt.get("fps", "?") if droid_rt else "?"
        ofps = orb_rt.get("fps", "?") if orb_rt else "?"
        dms  = droid_rt.get("max_ms", "?") if droid_rt else "?"
        oms  = orb_rt.get("max_ms", "?") if orb_rt else "?"
        lines.append(f"  Avg FPS     : DROID={dfps}  ORB={ofps}")
        lines.append(f"  Max latency : DROID={dms}ms  ORB={oms}ms")

    # Health=0 segment detayı
    d_segs = droid_m.get("h0_segments", [])
    o_segs = orb_m.get("h0_segments", [])
    if d_segs or o_segs:
        lines += ["", "Health=0 Segment Detayı (DROID):", "-" * 60,
                  f"  {'#':<4} {'Başlangıç':>9} {'Bitiş':>7} {'Uzunluk':>8} {'RMSE':>9} {'MaxErr':>9}"]
        for idx, seg in enumerate(d_segs, 1):
            lines.append(
                f"  {idx:<4} {seg['start']:>9} {seg['end']:>7} "
                f"{seg['length']:>8} {seg['rmse']:>9.4f} {seg['max_err']:>9.4f}"
            )
        if o_segs:
            lines += ["", "Health=0 Segment Detayı (ORB):", "-" * 60,
                      f"  {'#':<4} {'Başlangıç':>9} {'Bitiş':>7} {'Uzunluk':>8} {'RMSE':>9} {'MaxErr':>9}"]
            for idx, seg in enumerate(o_segs, 1):
                lines.append(
                    f"  {idx:<4} {seg['start']:>9} {seg['end']:>7} "
                    f"{seg['length']:>8} {seg['rmse']:>9.4f} {seg['max_err']:>9.4f}"
                )

    lines += [
        "",
        "=" * 60,
        f"FINAL MODEL : {final_model}",
        f"Gerekçe     : {reason}",
        "=" * 60,
    ]
    return "\n".join(lines)


def plot_trajectory(
    droid_rows: list[dict],
    orb_rows:   list[dict],
    out_path:   str,
) -> None:
    """
    GT vs DROID vs ORB yörünge karşılaştırması.
    Health=0 frame'ler kesik çizgi ile gösterilir.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [WARN] matplotlib yok — trajectory_overlay.png atlanıyor")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    def _split_segments(rows, key_x, key_y, health_val):
        """health==health_val olan ardışık segmentleri döndür."""
        segs = []
        cur_x, cur_y = [], []
        for r in rows:
            if r["health"] == health_val:
                cur_x.append(r[key_x])
                cur_y.append(r[key_y])
            else:
                if cur_x:
                    segs.append((cur_x[:], cur_y[:]))
                    cur_x, cur_y = [], []
        if cur_x:
            segs.append((cur_x, cur_y))
        return segs

    # GT (yalnızca bir kez, droid_rows'dan alınır)
    gt_x = [r["gt_x"] for r in droid_rows]
    gt_y = [r["gt_y"] for r in droid_rows]
    ax.plot(gt_x, gt_y, "k-", lw=2.5, label="Ground Truth", zorder=5)
    ax.plot(gt_x[0], gt_y[0], "k^", ms=10, zorder=6)   # başlangıç
    ax.plot(gt_x[-1], gt_y[-1], "ks", ms=10, zorder=6)  # bitiş

    colors = {"DROID": ("#1f77b4", "#aec7e8"), "ORB": ("#d62728", "#f7b6d2")}
    all_model_rows = [("DROID", droid_rows), ("ORB", orb_rows)]

    for name, rows in all_model_rows:
        col_h1, col_h0 = colors[name]
        # health=1 segmentler — düz çizgi
        for seg_x, seg_y in _split_segments(rows, "est_x", "est_y", 1):
            ax.plot(seg_x, seg_y, "-", color=col_h1, lw=1.5,
                    label=f"{name} (h=1)" if seg_x is _split_segments(rows, "est_x", "est_y", 1)[0][0] else "")
        # health=0 segmentler — kesik nokta
        for seg_x, seg_y in _split_segments(rows, "est_x", "est_y", 0):
            ax.plot(seg_x, seg_y, "--", color=col_h0, lw=1.2)

    # Lejant
    patches = [
        mpatches.Patch(color="k",         label="Ground Truth"),
        mpatches.Patch(color="#1f77b4",    label="DROID health=1"),
        mpatches.Patch(color="#aec7e8",    label="DROID health=0 (kesik)"),
        mpatches.Patch(color="#d62728",    label="ORB health=1"),
        mpatches.Patch(color="#f7b6d2",    label="ORB health=0 (kesik)"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory Overlay — GT vs DROID vs ORB")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Grafik kaydedildi: {out_path}")


def plot_error_vs_time(
    droid_rows: list[dict],
    orb_rows:   list[dict],
    out_path:   str,
) -> None:
    """
    Frame bazlı 2D konum hatası + health=0 bölgeleri gölgeli.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib yok — error_vs_time.png atlanıyor")
        return

    def _errors(rows):
        frames = np.array([r["frame"]  for r in rows])
        health = np.array([r["health"] for r in rows])
        err    = np.sqrt(
            (np.array([r["est_x"] for r in rows]) - np.array([r["gt_x"] for r in rows]))**2 +
            (np.array([r["est_y"] for r in rows]) - np.array([r["gt_y"] for r in rows]))**2
        )
        return frames, health, err

    d_f, d_h, d_e = _errors(droid_rows)
    o_f, o_h, o_e = _errors(orb_rows)

    fig, ax = plt.subplots(figsize=(13, 5))

    # Health=0 bölgelerini gri gölge ile işaretle (DROID verisi referans)
    in_dead = False
    dead_start = 0
    for i, (f, h) in enumerate(zip(d_f, d_h)):
        if h == 0 and not in_dead:
            dead_start = f
            in_dead = True
        elif h == 1 and in_dead:
            ax.axvspan(dead_start, f, color="gray", alpha=0.18, lw=0)
            in_dead = False
    if in_dead:
        ax.axvspan(dead_start, d_f[-1], color="gray", alpha=0.18, lw=0,
                   label="Health=0 bölgesi")

    ax.plot(d_f, d_e, "#1f77b4", lw=1.2, label="DROID hata (m)")
    ax.plot(o_f, o_e, "#d62728", lw=1.0, alpha=0.8, label="ORB hata (m)")

    # Ortalama çizgiler
    ax.axhline(float(np.sqrt(np.mean(d_e**2))), color="#1f77b4",
               ls="--", lw=0.9, alpha=0.6, label=f"DROID RMSE={np.sqrt(np.mean(d_e**2)):.3f}m")
    ax.axhline(float(np.sqrt(np.mean(o_e**2))), color="#d62728",
               ls="--", lw=0.9, alpha=0.6, label=f"ORB RMSE={np.sqrt(np.mean(o_e**2)):.3f}m")

    ax.set_xlabel("Frame")
    ax.set_ylabel("2D Pozisyon Hatası (m)")
    ax.set_title("Error vs Time — DROID vs ORB  [gri=health=0]")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Grafik kaydedildi: {out_path}")


def load_runtime(path: str) -> dict:
    """runtime.txt'den model bazlı FPS/latency oku."""
    result = {}
    if not os.path.exists(path):
        return result
    current = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("[DROID]"):
                current = "droid"
                result[current] = {}
            elif line.startswith("[ORB]"):
                current = "orb"
                result[current] = {}
            elif current and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k == "avg_fps":
                    result[current]["fps"] = v
                elif k == "max_latency":
                    result[current]["max_ms"] = v.replace(" ms", "")
    return result


def main():
    parser = argparse.ArgumentParser(description="DROID vs ORB evaluasyon")
    parser.add_argument("--droid", default=os.path.join(RESULTS_DIR, "est_droid.csv"))
    parser.add_argument("--orb",   default=os.path.join(RESULTS_DIR, "est_orb.csv"))
    parser.add_argument("--out",   default=OUT_TXT)
    parser.add_argument("--plots", action="store_true",
                        help="Trajectory overlay + error vs time grafikleri üret")
    parser.add_argument("--plots-dir", default=RESULTS_DIR,
                        help="Grafik çıktı dizini (varsayılan: competition/results/)")
    args = parser.parse_args()

    for p, label in [(args.droid, "DROID"), (args.orb, "ORB")]:
        if not os.path.exists(p):
            print(f"HATA: {label} sonucu bulunamadı: {p}", file=sys.stderr)
            print("Önce çalıştır:", file=sys.stderr)
            print("  python3 competition/run_estimator_on_dataset.py --model droid", file=sys.stderr)
            print("  python3 competition/run_estimator_on_dataset.py --model orb", file=sys.stderr)
            sys.exit(1)

    droid_rows = load_est(args.droid)
    orb_rows   = load_est(args.orb)

    droid_m = compute_metrics(droid_rows)
    orb_m   = compute_metrics(orb_rows)

    runtime_all = load_runtime(os.path.join(RESULTS_DIR, "runtime.txt"))
    droid_rt = runtime_all.get("droid")
    orb_rt   = runtime_all.get("orb")

    final_model, reason = select_model(droid_m, orb_m)
    report = format_report(droid_m, orb_m, droid_rt, orb_rt, final_model, reason)

    print(report)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(f"\nRapor kaydedildi: {args.out}")

    # Grafikler
    if args.plots:
        os.makedirs(args.plots_dir, exist_ok=True)
        print("\nGrafikler üretiliyor...")
        plot_trajectory(
            droid_rows, orb_rows,
            os.path.join(args.plots_dir, "trajectory_overlay.png"),
        )
        plot_error_vs_time(
            droid_rows, orb_rows,
            os.path.join(args.plots_dir, "error_vs_time.png"),
        )

    # FINAL_MODEL ortam değişkenini stdout'a yaz (scripts tarafından okunabilir)
    print(f"\nFINAL_MODEL={final_model}")


if __name__ == "__main__":
    main()
