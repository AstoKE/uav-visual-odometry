#!/usr/bin/env bash
# run_droid_figure8.sh — Figure-8 dataset ile DROID-SLAM çalıştırır.
#
# Kullanım:
#   bash ~/code/uav-visual-odometry/slam/scripts/run_droid_figure8.sh
#
# Ortam değişkenleri:
#   N_SLAM_FRAMES : (bilgi amaçlı, script otomatik sayar)

set -e

REPO_ROOT="$HOME/code/uav-visual-odometry"
DROID_DIR="$REPO_ROOT/DROID-SLAM"
IMAGEDIR="$REPO_ROOT/dataset/small_motion_v2"
CALIB="$REPO_ROOT/dataset/meta/calib.txt"
WEIGHTS="$DROID_DIR/checkpoints/droid.pth"
TRAJ_OUT="$REPO_ROOT/slam/outputs/trajectory_figure8.csv"

echo "[run_droid_figure8] ============================================"
echo "[run_droid_figure8] DROID-SLAM — Figure-8 dataset"
echo "[run_droid_figure8] imagedir : $IMAGEDIR"
echo "[run_droid_figure8] traj out : $TRAJ_OUT"
echo "[run_droid_figure8] ============================================"

# Ön kontroller
if [ ! -d "$IMAGEDIR" ] || [ -z "$(ls -A "$IMAGEDIR"/*.png 2>/dev/null)" ]; then
    echo "[run_droid_figure8] HATA: dataset/small_motion_v2 boş veya yok."
    echo "  Önce çalıştır:"
    echo "    python3 $REPO_ROOT/dataset/make_small_motion_v2.py"
    exit 1
fi
[ ! -f "$CALIB"   ] && { echo "HATA: calib.txt bulunamadı: $CALIB";   exit 1; }
[ ! -f "$WEIGHTS" ] && { echo "HATA: model ağırlıkları yok: $WEIGHTS"; exit 1; }

frame_count=$(ls "$IMAGEDIR"/*.png 2>/dev/null | wc -l)
echo "[run_droid_figure8] Dataset frame sayısı: $frame_count"
echo "[run_droid_figure8] Beklenen SLAM frame : $((frame_count / 3))  (stride=3)"

# Conda ortamı
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate droid_clean
echo "[run_droid_figure8] Python: $(which python)"

cd "$DROID_DIR"
echo "[run_droid_figure8] demo.py çalıştırılıyor..."

python demo.py \
    --imagedir        "$IMAGEDIR" \
    --calib           "$CALIB"    \
    --weights         "$WEIGHTS"  \
    --trajectory_path "$TRAJ_OUT" \
    --disable_vis

echo "[run_droid_figure8] Tamamlandı."
echo "[run_droid_figure8] Trajectory: $TRAJ_OUT"

if [ -f "$TRAJ_OUT" ]; then
    echo "[run_droid_figure8] --- İlk 5 satır ---"
    head -6 "$TRAJ_OUT"
fi

echo ""
echo "[run_droid_figure8] Sonraki adım:"
echo "  python3 $REPO_ROOT/slam/scripts/evaluate_figure8.py"
