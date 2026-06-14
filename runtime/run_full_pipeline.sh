#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh — Gazebo → SLAM → Estimator → Evaluation tam akışı
#
# Kullanım:
#   bash runtime/run_full_pipeline.sh
#   bash runtime/run_full_pipeline.sh --pattern competition --height 4.0
#   bash runtime/run_full_pipeline.sh --skip-gazebo   # sadece SLAM + eval
#   bash runtime/run_full_pipeline.sh --skip-slam     # sadece eval
#   bash runtime/run_full_pipeline.sh --dry-run       # komutları yazdır, çalıştırma
#
# Adımlar:
#   1. Gazebo başlat (arka planda)
#   2. Kamera recorder başlat
#   3. move_camera çalıştır (waypoints)
#   4. Dataset oluştur (raw → small_motion)
#   5. DROID-SLAM çalıştır (conda: droid_clean)
#   6. Estimator pipeline çalıştır
#   7. Evaluation + grafik çıktısı
#
# Çıktılar:
#   dataset/raw_competition/          — ham görüntüler
#   dataset/small_motion_competition/ — işlenmiş dataset
#   competition/results/              — SLAM + estimator çıktıları
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ── Renkli log ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
TS() { date '+%H:%M:%S'; }
info()  { echo -e "${CYAN}[$(TS) INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[$(TS)  OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[$(TS) WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[$(TS) FAIL]${NC}  $*"; exit 1; }
step()  { echo; echo -e "${BOLD}━━━ $* ━━━${NC}"; }

# ── Varsayılan parametreler ───────────────────────────────────────────────────
PATTERN="competition"
HEIGHT="4.0"
WORLD="slam_world"
SKIP_GAZEBO=""
SKIP_SLAM=""
DRY_RUN=""
CONDA_ENV="droid_clean"
DATASET_NAME="competition"
N_FRAMES=2250        # yarışma: 5 dakika × 7.5fps = 2250 frame
STRIDE=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pattern)     PATTERN="$2";      shift 2 ;;
    --height)      HEIGHT="$2";       shift 2 ;;
    --world)       WORLD="$2";        shift 2 ;;
    --dataset)     DATASET_NAME="$2"; shift 2 ;;
    --frames)      N_FRAMES="$2";     shift 2 ;;
    --stride)      STRIDE="$2";       shift 2 ;;
    --skip-gazebo) SKIP_GAZEBO=1;     shift ;;
    --skip-slam)   SKIP_SLAM=1;       shift ;;
    --dry-run)     DRY_RUN=1;         shift ;;
    -h|--help)
      grep '^#' "$0" | head -30 | sed 's/^# \?//'
      exit 0 ;;
    *)
      fail "Bilinmeyen argüman: $1" ;;
  esac
done

RUN() {
  if [[ -n "$DRY_RUN" ]]; then
    echo -e "  ${YELLOW}[DRY-RUN]${NC} $*"
  else
    eval "$@"
  fi
}

RAW_DIR="$REPO/dataset/raw_${DATASET_NAME}"
PROCESSED_DIR="$REPO/dataset/small_motion_${DATASET_NAME}"
SLAM_OUT="$REPO/competition/results/slam_${DATASET_NAME}.csv"
GT_CSV="$REPO/evaluation/ground_truth_${DATASET_NAME}.csv"
LOG_DIR="$REPO/competition/results/pipeline_logs"

mkdir -p "$LOG_DIR" "$REPO/competition/results"

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║   UAV VO Full Pipeline  —  $(date '+%Y-%m-%d %H:%M')       ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
info "Pattern   : $PATTERN"
info "Height    : ${HEIGHT}m"
info "Dataset   : $DATASET_NAME  (n=$N_FRAMES  stride=$STRIDE)"
info "Raw dir   : $RAW_DIR"
info "Proc dir  : $PROCESSED_DIR"
[[ -n "$DRY_RUN"    ]] && warn "DRY-RUN modu — komutlar çalıştırılmayacak"
[[ -n "$SKIP_GAZEBO"]] && warn "Gazebo adımı atlanıyor"
[[ -n "$SKIP_SLAM"  ]] && warn "SLAM adımı atlanıyor"

# ─────────────────────────────────────────────────────────────────────────────
# ADIM 1 — Gazebo başlat
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$SKIP_GAZEBO" ]]; then
  step "ADIM 1/7 — Gazebo Başlatma"

  if pgrep -x "gz" > /dev/null 2>&1; then
    warn "Gazebo zaten çalışıyor, atlanıyor."
  else
    info "Gazebo arka planda başlatılıyor..."
    RUN "WORLD_NAME=$WORLD bash '$REPO/sim/scripts/run_gazebo.sh' \
         > '$LOG_DIR/gazebo.log' 2>&1 &"
    GAZEBO_PID=$!
    info "Gazebo PID: $GAZEBO_PID — 8s bekleniyor..."
    [[ -z "$DRY_RUN" ]] && sleep 8
    ok "Gazebo başlatıldı."
  fi
else
  info "ADIM 1 atlandı (--skip-gazebo)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# ADIM 2 — Recorder başlat
# ─────────────────────────────────────────────────────────────────────────────
step "ADIM 2/7 — Recorder Başlatma"

mkdir -p "$RAW_DIR"

if [[ -z "$SKIP_GAZEBO" ]]; then
  info "Gazebo topic recorder başlatılıyor → $RAW_DIR"
  RUN "python3 '$REPO/sim/scripts/export_ground_truth.py' \
       --out '$GT_CSV' \
       --frames $N_FRAMES \
       > '$LOG_DIR/recorder.log' 2>&1 &"
  RECORDER_PID=$!
  info "Recorder PID: $RECORDER_PID"
  [[ -z "$DRY_RUN" ]] && sleep 2
else
  info "ADIM 2 atlandı (--skip-gazebo)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# ADIM 3 — Kamera hareket ettir
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$SKIP_GAZEBO" ]]; then
  step "ADIM 3/7 — Kamera Hareketi ($PATTERN)"
  info "move_camera çalıştırılıyor (PATTERN=$PATTERN HEIGHT=$HEIGHT)..."
  RUN "PATTERN=$PATTERN HEIGHT=$HEIGHT WORLD_NAME=$WORLD \
       python3 '$REPO/sim/scripts/move_camera.py' \
       2>&1 | tee '$LOG_DIR/move_camera.log'"
  ok "Hareket tamamlandı."

  # Recorder'ı durdur
  if [[ -n "${RECORDER_PID:-}" ]]; then
    info "Recorder durduruluyor (PID=$RECORDER_PID)..."
    kill "$RECORDER_PID" 2>/dev/null || true
  fi
else
  info "ADIM 3 atlandı (--skip-gazebo)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# ADIM 4 — Dataset oluştur
# ─────────────────────────────────────────────────────────────────────────────
step "ADIM 4/7 — Dataset Oluşturma"

if [[ -d "$RAW_DIR" ]] && [[ -n "$(ls -A "$RAW_DIR" 2>/dev/null)" ]]; then
  info "Dataset dönüştürülüyor: $RAW_DIR → $PROCESSED_DIR"
  RUN "python3 '$REPO/dataset/make_small_motion_v3.py' \
       --src '$RAW_DIR' \
       --dst '$PROCESSED_DIR' \
       --stride $STRIDE \
       > '$LOG_DIR/dataset.log' 2>&1"
  ok "Dataset oluşturuldu: $PROCESSED_DIR"
else
  warn "Ham dizin boş veya yok: $RAW_DIR — dataset adımı atlanıyor."
fi

# ─────────────────────────────────────────────────────────────────────────────
# ADIM 5 — DROID-SLAM çalıştır
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$SKIP_SLAM" ]]; then
  step "ADIM 5/7 — DROID-SLAM"

  DROID_CKPT="$REPO/DROID-SLAM/checkpoints/droid.pth"
  DROID_CALIB="$REPO/dataset/meta/calib.txt"

  if [[ ! -f "$DROID_CKPT" ]]; then
    warn "DROID checkpoint bulunamadı: $DROID_CKPT"
    warn "SLAM adımı atlanıyor — ORB-VO fallback kullanılacak."
    SKIP_SLAM=1
  elif ! command -v conda &>/dev/null; then
    warn "conda bulunamadı — SLAM atlanıyor."
    SKIP_SLAM=1
  else
    info "DROID-SLAM başlatılıyor (conda env: $CONDA_ENV)..."
    RUN "conda run -n '$CONDA_ENV' python '$REPO/DROID-SLAM/demo.py' \
         --imagedir '$PROCESSED_DIR' \
         --calib '$DROID_CALIB' \
         --weights '$DROID_CKPT' \
         --trajectory_path '$SLAM_OUT' \
         --disable_vis \
         > '$LOG_DIR/droid.log' 2>&1"
    ok "DROID-SLAM tamamlandı → $SLAM_OUT"
  fi
else
  info "ADIM 5 atlandı (--skip-slam)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# ADIM 6 — Estimator Pipeline
# ─────────────────────────────────────────────────────────────────────────────
step "ADIM 6/7 — Estimator Pipeline"

CALIB_FILE="$REPO/dataset/meta/calib.txt"
if [[ ! -f "$CALIB_FILE" ]]; then
  warn "Kalibrasyon dosyası bulunamadı: $CALIB_FILE — varsayılan 457 457 320 240 kullanılıyor"
  FX=457; FY=457; CX=320; CY=240
else
  read -r FX FY CX CY < "$CALIB_FILE"
fi

info "Estimator çalıştırılıyor..."
RUN "python3 '$REPO/competition/run_teknofest_test.py' \
     --images '$PROCESSED_DIR' \
     --gt     '${GT_CSV:-$REPO/evaluation/ground_truth_figure8_v3_frames900.csv}' \
     --calib  '$CALIB_FILE' \
     --out    '$REPO/competition/results' \
     > '$LOG_DIR/estimator.log' 2>&1"
ok "Estimator tamamlandı."

# ─────────────────────────────────────────────────────────────────────────────
# ADIM 7 — Evaluation + Rapor
# ─────────────────────────────────────────────────────────────────────────────
step "ADIM 7/7 — Evaluation & Rapor"

info "Evaluation çalıştırılıyor..."
RUN "python3 '$REPO/competition/evaluate_online.py' \
     --plots --plots-dir '$REPO/competition/results' \
     > '$LOG_DIR/evaluation.log' 2>&1" || warn "evaluate_online.py başarısız (dosya yok olabilir)"

# Otomatik docs üret
info "Performans özeti üretiliyor..."
RUN "python3 '$REPO/docs/gen_performance_summary.py' \
     > '$LOG_DIR/docs.log' 2>&1" || true

# ── Özet ──────────────────────────────────────────────────────────────────────
echo
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Pipeline tamamlandı!${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo
echo "  Çıktılar:"
echo "    Dataset       : $PROCESSED_DIR"
echo "    SLAM sonucu   : $SLAM_OUT"
echo "    Estimator CSV : $REPO/competition/results/teknofest_test_results.csv"
echo "    Grafikler     : $REPO/competition/results/"
echo "    Pipeline logs : $LOG_DIR/"
echo
info "Çalıştırma komutu:"
echo "    cat $LOG_DIR/estimator.log | tail -40"
