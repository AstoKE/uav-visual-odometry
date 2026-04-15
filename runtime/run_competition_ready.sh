#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_competition_ready.sh — Yarışma günü tek komutla sistem ayağa kaldırma
#
# Kullanım:
#   bash runtime/run_competition_ready.sh
#   bash runtime/run_competition_ready.sh --url http://SERVER:PORT --token TOKEN
#   bash runtime/run_competition_ready.sh --no-slam      # LK fallback modu
#   bash runtime/run_competition_ready.sh --validate     # önce validation koş
#
# Ortam değişkenleri (override için):
#   COMP_URL   — sunucu adresi   (varsayılan: http://localhost:8080)
#   COMP_TOKEN — auth token      (varsayılan: boş)
#   COMP_CALIB — "fx fy cx cy"   (varsayılan: "457 457 320 240")
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ── Renkli çıktı ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║   UAV Visual Odometry — Competition Ready v1.0  ║"
echo "  ║   FINAL_MODEL = DROID-SLAM                      ║"
echo "  ║   FINAL_WORLD = v4  (slam_world_realistic.sdf)  ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Argümanları işle ──────────────────────────────────────────────────────────
URL="${COMP_URL:-http://localhost:8080}"
TOKEN="${COMP_TOKEN:-}"
CALIB="${COMP_CALIB:-457 457 320 240}"
NO_SLAM=""
RUN_VALIDATE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)       URL="$2";     shift 2 ;;
    --token)     TOKEN="$2";   shift 2 ;;
    --calib)     CALIB="$2";   shift 2 ;;
    --no-slam)   NO_SLAM="--no-slam"; shift ;;
    --validate)  RUN_VALIDATE="1";    shift ;;
    -h|--help)
      echo "Kullanım: bash $0 [--url URL] [--token TOKEN] [--no-slam] [--validate]"
      exit 0 ;;
    *)
      error "Bilinmeyen argüman: $1" ;;
  esac
done

# ── 1. Ortam kontrolü ─────────────────────────────────────────────────────────
info "Ortam kontrol ediliyor..."

PYTHON=$(command -v python3 || true)
[[ -z "$PYTHON" ]] && error "python3 bulunamadı."

$PYTHON -c "import numpy" 2>/dev/null  || error "numpy yok: pip install numpy"
$PYTHON -c "import cv2"   2>/dev/null  || warn  "opencv-python yok — LK fallback çalışmaz"

ok "Python: $($PYTHON --version)"

# ── 2. Repo yapısı kontrolü ───────────────────────────────────────────────────
info "Repo yapısı kontrol ediliyor..."

REQUIRED_FILES=(
  "competition/estimator.py"
  "competition/slam_pose_estimator.py"
  "competition/client.py"
  "runtime/run_final_system.py"
  "runtime/final_config.yaml"
)

for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$REPO/$f" ]]; then
    error "Dosya eksik: $f"
  fi
done
ok "Gerekli dosyalar mevcut."

# ── 3. Validation suite (isteğe bağlı) ───────────────────────────────────────
if [[ -n "$RUN_VALIDATE" ]]; then
  info "Validation suite çalıştırılıyor (--validate)..."
  $PYTHON competition/run_validation_suite.py \
    --frames 300 \
    --out competition/results/validate_preflight.csv
  ok "Validation tamamlandı → competition/results/validate_preflight.csv"
fi

# ── 4. Results dizini ─────────────────────────────────────────────────────────
mkdir -p "$REPO/competition/results"
info "Sonuç dizini: $REPO/competition/results/"

# ── 5. Final sistem çalıştır ──────────────────────────────────────────────────
info "Final sistem başlatılıyor..."
info "  URL    : $URL"
info "  Calib  : $CALIB"
info "  Mode   : ${NO_SLAM:-SLAM+SLAMPoseEstimator}"
echo

TOKEN_ARG=""
[[ -n "$TOKEN" ]] && TOKEN_ARG="--token $TOKEN"

CALIB_ARG="--calib $CALIB"

# shellcheck disable=SC2086
exec $PYTHON runtime/run_final_system.py \
  --url  "$URL"       \
  $TOKEN_ARG          \
  $CALIB_ARG          \
  $NO_SLAM
