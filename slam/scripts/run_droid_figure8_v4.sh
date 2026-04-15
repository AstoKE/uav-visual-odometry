#!/usr/bin/env bash
# run_droid_figure8_v4.sh — Figure-8 v4 (realistic world) ile DROID-SLAM
#
# v4 realistic world:
#   - 12×12 siyah-beyaz checkerboard zemin
#   - cast_shadows, kamera Gaussian noise
#   - Aynı kamera parametreleri: 4m, FOV=1.2217, fx=fy=457
#
# Kullanım:
#   bash ~/code/uav-visual-odometry/slam/scripts/run_droid_figure8_v4.sh

set -e

REPO_ROOT="$HOME/code/uav-visual-odometry"
DROID_DIR="$REPO_ROOT/DROID-SLAM"
IMAGEDIR="$REPO_ROOT/dataset/small_motion_v4"
CALIB="$REPO_ROOT/dataset/meta/calib.txt"
WEIGHTS="$DROID_DIR/checkpoints/droid.pth"
TRAJ_OUT="$REPO_ROOT/slam/outputs/trajectory_figure8_v4.csv"

echo "[run_droid_v4] ============================================"
echo "[run_droid_v4] DROID-SLAM — Figure-8 v4 (realistic world)"
echo "[run_droid_v4] imagedir : $IMAGEDIR"
echo "[run_droid_v4] calib    : $CALIB  (fx=457)"
echo "[run_droid_v4] traj out : $TRAJ_OUT"
echo "[run_droid_v4] ============================================"

if [ ! -d "$IMAGEDIR" ] || [ -z "$(ls -A "$IMAGEDIR"/*.png 2>/dev/null)" ]; then
    echo "[run_droid_v4] HATA: dataset/small_motion_v4 boş veya yok."
    echo "  Önce çalıştır:"
    echo "    python3 $REPO_ROOT/dataset/make_small_motion_v4.py"
    exit 1
fi
[ ! -f "$CALIB"   ] && { echo "HATA: calib.txt yok: $CALIB";    exit 1; }
[ ! -f "$WEIGHTS" ] && { echo "HATA: model ağırlıkları yok: $WEIGHTS"; exit 1; }

frame_count=$(ls "$IMAGEDIR"/*.png 2>/dev/null | wc -l)
echo "[run_droid_v4] Dataset frame sayısı : $frame_count"
echo "[run_droid_v4] Beklenen SLAM frame  : $((frame_count / 3))  (stride=3)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate droid_clean
echo "[run_droid_v4] Python: $(which python)"

cd "$DROID_DIR"
echo "[run_droid_v4] demo.py çalıştırılıyor (stride=3)..."

python demo.py \
    --imagedir        "$IMAGEDIR" \
    --calib           "$CALIB"    \
    --weights         "$WEIGHTS"  \
    --trajectory_path "$TRAJ_OUT" \
    --disable_vis

echo "[run_droid_v4] Tamamlandı."
if [ -f "$TRAJ_OUT" ]; then
    echo "[run_droid_v4] --- İlk 5 satır ---"
    head -6 "$TRAJ_OUT"
fi
echo ""
echo "[run_droid_v4] Sonraki adım:"
echo "  python3 $REPO_ROOT/slam/scripts/evaluate_figure8_v4.py"
