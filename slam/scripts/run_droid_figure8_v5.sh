#!/usr/bin/env bash
# run_droid_figure8_v5.sh — Figure-8 v5 (realistic world v5) ile DROID-SLAM
#
# v5 realistic world:
#   - Lens distortion k1=-0.12, k2=0.015 → gerçekçi geniş açı
#   - Kamera noise stddev=0.015–0.022 (seed-bağımlı) → daha güçlü sim2real
#   - FPS 20 (v4: 15) → daha sık frame
#   - 3-light setup, ambient 0.18–0.25 → derin gölgeler
#   - 8 dinamik Actor → SLAM outlier rejection testi
#   - 5 yüksek kule + 10 platform → parallax çeşitliliği
#
# Kullanım:
#   bash ~/code/uav-visual-odometry/slam/scripts/run_droid_figure8_v5.sh

set -e

REPO_ROOT="$HOME/code/uav-visual-odometry"
DROID_DIR="$REPO_ROOT/DROID-SLAM"
IMAGEDIR="$REPO_ROOT/dataset/small_motion_v5"
CALIB="$REPO_ROOT/dataset/meta/calib.txt"
WEIGHTS="$DROID_DIR/checkpoints/droid.pth"
TRAJ_OUT="$REPO_ROOT/slam/outputs/trajectory_figure8_v5.csv"

echo "[run_droid_v5] ============================================"
echo "[run_droid_v5] DROID-SLAM — Figure-8 v5 (realistic world)"
echo "[run_droid_v5] imagedir : $IMAGEDIR"
echo "[run_droid_v5] calib    : $CALIB  (fx=457)"
echo "[run_droid_v5] traj out : $TRAJ_OUT"
echo "[run_droid_v5] ============================================"

if [ ! -d "$IMAGEDIR" ] || [ -z "$(ls -A "$IMAGEDIR"/*.png 2>/dev/null)" ]; then
    echo "[run_droid_v5] HATA: dataset/small_motion_v5 boş veya yok."
    echo "  Önce çalıştır:"
    echo "    python3 $REPO_ROOT/dataset/make_small_motion_v5.py"
    exit 1
fi
[ ! -f "$CALIB"   ] && { echo "HATA: calib.txt yok: $CALIB";    exit 1; }
[ ! -f "$WEIGHTS" ] && { echo "HATA: model ağırlıkları yok: $WEIGHTS"; exit 1; }

frame_count=$(ls "$IMAGEDIR"/*.png 2>/dev/null | wc -l)
echo "[run_droid_v5] Dataset frame sayısı : $frame_count"
echo "[run_droid_v5] Beklenen SLAM frame  : $((frame_count / 3))  (stride=3)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate droid_clean
echo "[run_droid_v5] Python: $(which python)"

cd "$DROID_DIR"
echo "[run_droid_v5] demo.py çalıştırılıyor (stride=3)..."

python demo.py \
    --imagedir        "$IMAGEDIR" \
    --calib           "$CALIB"    \
    --weights         "$WEIGHTS"  \
    --trajectory_path "$TRAJ_OUT" \
    --disable_vis

echo "[run_droid_v5] Tamamlandı."
if [ -f "$TRAJ_OUT" ]; then
    echo "[run_droid_v5] --- İlk 5 satır ---"
    head -6 "$TRAJ_OUT"
fi
echo ""
echo "[run_droid_v5] Sonraki adım:"
echo "  python3 $REPO_ROOT/slam/scripts/evaluate_figure8_v5.py"
