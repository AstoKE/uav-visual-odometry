#!/usr/bin/env bash
# =============================================================================
# run_realism_benchmark.sh — v4 vs v5 world SLAM karşılaştırma benchmark
#
# Akış:
#   1. v4 realistic world ile dataset topla → SLAM çalıştır → metrikleri kaydet
#   2. v5 realistic world ile dataset topla → SLAM çalıştır → metrikleri kaydet
#   3. RMSE / drift karşılaştır → evaluation/metrics/realism_comparison.txt
#
# Kullanım:
#   bash slam/scripts/run_realism_benchmark.sh
#   bash slam/scripts/run_realism_benchmark.sh --skip-collect   # var olan dataset kullan
#   bash slam/scripts/run_realism_benchmark.sh --seed 99        # farklı v5 dünyası
#   bash slam/scripts/run_realism_benchmark.sh --dry-run
#
# Çıktılar:
#   dataset/raw_bench_v4/              — v4 ham görüntüler
#   dataset/raw_bench_v5/              — v5 ham görüntüler
#   dataset/small_motion_bench_v4/     — v4 işlenmiş
#   dataset/small_motion_bench_v5/     — v5 işlenmiş
#   competition/results/bench_v4.csv   — v4 SLAM sonucu
#   competition/results/bench_v5.csv   — v5 SLAM sonucu
#   evaluation/metrics/realism_comparison.txt
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
TS() { date '+%H:%M:%S'; }
info()  { echo -e "${CYAN}[$(TS) INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[$(TS)  OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[$(TS) WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[$(TS) FAIL]${NC}  $*"; exit 1; }
step()  { echo; echo -e "${BOLD}━━━ $* ━━━${NC}"; }

# ── Parametreler ──────────────────────────────────────────────────────────────
V5_SEED="42"
N_FRAMES="900"
STRIDE="3"
SKIP_COLLECT=""
DRY_RUN=""
CONDA_ENV="droid_clean"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)         V5_SEED="$2";    shift 2 ;;
    --frames)       N_FRAMES="$2";   shift 2 ;;
    --skip-collect) SKIP_COLLECT=1;  shift ;;
    --dry-run)      DRY_RUN=1;       shift ;;
    -h|--help)
      grep '^#' "$0" | head -25 | sed 's/^# \?//'; exit 0 ;;
    *) fail "Bilinmeyen argüman: $1" ;;
  esac
done

RUN() {
  if [[ -n "$DRY_RUN" ]]; then
    echo -e "  ${YELLOW}[DRY-RUN]${NC} $*"
  else
    eval "$@"
  fi
}

LOG_DIR="$REPO/evaluation/metrics/benchmark_logs"
METRICS_DIR="$REPO/evaluation/metrics"
mkdir -p "$LOG_DIR" "$METRICS_DIR"

DROID_CKPT="$REPO/DROID-SLAM/checkpoints/droid.pth"
DROID_SCRIPT="$REPO/slam/scripts/run_droid_figure8_v3.sh"

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║   Realism Benchmark  —  v4 vs v5 World                  ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
info "V5 seed   : $V5_SEED"
info "Frames    : $N_FRAMES  stride=$STRIDE"
[[ -n "$DRY_RUN"      ]] && warn "DRY-RUN modu"
[[ -n "$SKIP_COLLECT" ]] && warn "--skip-collect: var olan datasetler kullanılacak"

# ═════════════════════════════════════════════════════════════════════════════
# ADIM 0 — v5 world dosyasını üret (her çalıştırmada seed ile)
# ═════════════════════════════════════════════════════════════════════════════
step "ADIM 0 — v5 World Üretimi (seed=$V5_SEED)"
V5_SDF="$REPO/sim/worlds/slam_world_v5_realistic.sdf"
RUN "python3 '$REPO/sim/scripts/gen_realistic_world_v5.py' \
     --seed '$V5_SEED' --out '$V5_SDF' \
     > '$LOG_DIR/gen_v5.log' 2>&1"
ok "v5 SDF hazır: $V5_SDF"

# ═════════════════════════════════════════════════════════════════════════════
# Yardımcı: tek bir world için veri topla + SLAM çalıştır
# ═════════════════════════════════════════════════════════════════════════════
run_world_pipeline() {
  local version="$1"    # "v4" veya "v5"
  local world_name="$2" # "slam_world_realistic" veya "slam_world_v5_realistic"

  local raw_dir="$REPO/dataset/raw_bench_${version}"
  local proc_dir="$REPO/dataset/small_motion_bench_${version}"
  local slam_out="$REPO/competition/results/bench_${version}.csv"
  local gt_csv="$REPO/evaluation/ground_truth_bench_${version}.csv"

  step "WORLD ${version^^} — ${world_name}"

  # ── Gazebo + recorder + hareket ──────────────────────────────────────────
  if [[ -z "$SKIP_COLLECT" ]]; then
    info "Gazebo başlatılıyor (${version})..."
    RUN "WORLD_NAME=$world_name bash '$REPO/sim/scripts/run_gazebo.sh' \
         > '$LOG_DIR/gazebo_${version}.log' 2>&1 &"
    [[ -z "$DRY_RUN" ]] && sleep 8

    mkdir -p "$raw_dir"
    info "Recorder başlatılıyor..."
    RECORDER_PID=""
    if [[ -z "$DRY_RUN" ]]; then
      python3 "$REPO/sim/scripts/export_ground_truth.py" \
        --out "$gt_csv" --frames "$N_FRAMES" \
        > "$LOG_DIR/recorder_${version}.log" 2>&1 &
      RECORDER_PID=$!
    else
      echo -e "  ${YELLOW}[DRY-RUN]${NC} python3 export_ground_truth.py --out $gt_csv --frames $N_FRAMES &"
    fi

    info "Kamera hareketi (competition)..."
    RUN "PATTERN=competition HEIGHT=4.0 WORLD_NAME=$world_name \
         python3 '$REPO/sim/scripts/move_camera.py' \
         2>&1 | tee '$LOG_DIR/move_${version}.log'"

    [[ -n "${RECORDER_PID:-}" ]] && kill "$RECORDER_PID" 2>/dev/null || true

    # Gazebo kapat
    pkill -x gz 2>/dev/null || true
    [[ -z "$DRY_RUN" ]] && sleep 2
    ok "Gazebo kapatıldı."
  else
    info "Veri toplama atlandı (--skip-collect)"
  fi

  # ── Dataset üret ──────────────────────────────────────────────────────────
  if [[ -d "$raw_dir" ]] && [[ -n "$(ls -A "$raw_dir" 2>/dev/null)" ]]; then
    info "Dataset oluşturuluyor..."
    RUN "python3 '$REPO/dataset/make_small_motion_v3.py' \
         --src '$raw_dir' --dst '$proc_dir' --stride $STRIDE \
         > '$LOG_DIR/dataset_${version}.log' 2>&1"
    ok "Dataset hazır: $proc_dir"
  else
    warn "Ham dizin boş: $raw_dir — dataset adımı atlanıyor."
  fi

  # ── DROID-SLAM ────────────────────────────────────────────────────────────
  if [[ ! -f "$DROID_CKPT" ]]; then
    warn "DROID checkpoint yok — SLAM atlanıyor ($version)"
    return
  fi
  if ! command -v conda &>/dev/null; then
    warn "conda yok — SLAM atlanıyor ($version)"
    return
  fi

  info "DROID-SLAM çalıştırılıyor ($version)..."
  RUN "conda run -n '$CONDA_ENV' bash '$DROID_SCRIPT' \
       --images '$proc_dir' --out '$slam_out' \
       > '$LOG_DIR/droid_${version}.log' 2>&1"
  ok "SLAM tamamlandı → $slam_out"
}

# ═════════════════════════════════════════════════════════════════════════════
# Her iki world'ü çalıştır
# ═════════════════════════════════════════════════════════════════════════════
run_world_pipeline "v4" "slam_world_realistic"
run_world_pipeline "v5" "slam_world_v5_realistic"

# ═════════════════════════════════════════════════════════════════════════════
# Karşılaştırma raporu
# ═════════════════════════════════════════════════════════════════════════════
step "Karşılaştırma Raporu"

COMPARE_SCRIPT="$REPO/evaluation/compare_realism.py"

# Inline Python karşılaştırma
python3 - <<'PYEOF'
import csv, os, sys
import numpy as np

REPO    = os.path.expanduser("~/code/uav-visual-odometry")
RESULTS = os.path.join(REPO, "competition/results")
METRICS = os.path.join(REPO, "evaluation/metrics")

def load_csv(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows

def rmse_2d(rows):
    if not rows:
        return float("nan")
    err = np.array([
        (r["est_x"] - r["gt_x"])**2 + (r["est_y"] - r["gt_y"])**2
        for r in rows if r.get("health", 0) == 0
    ])
    return float(np.sqrt(np.mean(err))) if len(err) else float("nan")

def final_drift(rows):
    if not rows:
        return float("nan")
    last = rows[-1]
    return float(np.sqrt((last["est_x"]-last["gt_x"])**2 + (last["est_y"]-last["gt_y"])**2))

def max_drift(rows):
    if not rows:
        return float("nan")
    errs = [np.sqrt((r["est_x"]-r["gt_x"])**2+(r["est_y"]-r["gt_y"])**2)
            for r in rows]
    return float(max(errs))

versions = ["v4", "v5"]
results  = {}
for v in versions:
    p = os.path.join(RESULTS, f"bench_{v}.csv")
    results[v] = load_csv(p)

lines = [
    "=" * 65,
    "Realism Benchmark — v4 vs v5 World Karşılaştırması",
    "=" * 65,
    "",
    f"{'Metrik':<24} {'v4 (realistic)':>18} {'v5 (v5_realistic)':>18} {'Fark':>8}",
    "-" * 65,
]

metrics_fn = [
    ("RMSE_2D (m)",    rmse_2d),
    ("Final Drift (m)", final_drift),
    ("Max Drift (m)",   max_drift),
]
for label, fn in metrics_fn:
    d4 = fn(results["v4"])
    d5 = fn(results["v5"])
    if not (d4 != d4) and not (d5 != d5):
        diff = d5 - d4
        sign = "↓" if diff < 0 else "↑"
        diff_str = f"{sign}{abs(diff):.3f}"
    else:
        diff_str = "N/A"
    v4_str = f"{d4:.4f}" if d4 == d4 else "N/A"
    v5_str = f"{d5:.4f}" if d5 == d5 else "N/A"
    lines.append(f"{label:<24} {v4_str:>18} {v5_str:>18} {diff_str:>8}")

lines += [
    "",
    "Not: ↓ = v5 daha iyi (daha düşük hata)",
    "     ↑ = v4 daha iyi",
    "",
    "v5 Yenilikleri:",
    "  - Lens distortion (k1=-0.12, k2=0.015)",
    "  - Noise stddev 0.010 → 0.015–0.022 (seed-bağımlı)",
    "  - Ambient 0.40 → 0.18–0.25 (3-light setup)",
    "  - 5 yüksek kule (parallax)",
    "  - 10 çok katlı platform",
    "  - 8 dinamik Actor (SLAM robustness testi)",
    "  - Zemin kir lekeleri (feature diversity)",
    "=" * 65,
]

report = "\n".join(lines)
print(report)

out_path = os.path.join(METRICS, "realism_comparison.txt")
os.makedirs(METRICS, exist_ok=True)
with open(out_path, "w") as f:
    f.write(report + "\n")
print(f"\nRapor: {out_path}")
PYEOF

echo
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Benchmark tamamlandı!${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo
echo "  Çıktılar:"
echo "    v4 SLAM     : $REPO/competition/results/bench_v4.csv"
echo "    v5 SLAM     : $REPO/competition/results/bench_v5.csv"
echo "    Rapor       : $REPO/evaluation/metrics/realism_comparison.txt"
echo "    Loglar      : $LOG_DIR/"
echo
info "v5 farklı seed ile test:"
echo "    bash slam/scripts/run_realism_benchmark.sh --seed 99 --skip-collect"
