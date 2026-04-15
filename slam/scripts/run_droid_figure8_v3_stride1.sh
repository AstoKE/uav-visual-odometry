#!/usr/bin/env bash
# run_droid_figure8_v3_stride1.sh — Figure-8 v3 dataset ile DROID-SLAM (stride=1)
#
# Mevcut v3 datasetini (small_motion_v3, 900 frame) stride=1 ile işler.
# Tüm frameler sırayla kullanılır → ~900 SLAM çıktı pozu (stride=3 ile ~300).
#
# Kullanım:
#   bash ~/code/uav-visual-odometry/slam/scripts/run_droid_figure8_v3_stride1.sh

set -e

REPO_ROOT="$HOME/code/uav-visual-odometry"
DROID_DIR="$REPO_ROOT/DROID-SLAM"
IMAGEDIR="$REPO_ROOT/dataset/small_motion_v3"
CALIB="$REPO_ROOT/dataset/meta/calib.txt"
WEIGHTS="$DROID_DIR/checkpoints/droid.pth"
TRAJ_OUT="$REPO_ROOT/slam/outputs/trajectory_figure8_v3_stride1.csv"

echo "[run_droid_v3_s1] ============================================"
echo "[run_droid_v3_s1] DROID-SLAM — Figure-8 v3  stride=1"
echo "[run_droid_v3_s1] imagedir : $IMAGEDIR"
echo "[run_droid_v3_s1] calib    : $CALIB  (fx=457)"
echo "[run_droid_v3_s1] traj out : $TRAJ_OUT"
echo "[run_droid_v3_s1] NOT: stride=1 → tüm 900 frame işlenir (~900 SLAM pozu)"
echo "[run_droid_v3_s1] ============================================"

if [ ! -d "$IMAGEDIR" ] || [ -z "$(ls -A "$IMAGEDIR"/*.png 2>/dev/null)" ]; then
    echo "[run_droid_v3_s1] HATA: dataset/small_motion_v3 boş veya yok."
    echo "  Önce çalıştır:"
    echo "    python3 $REPO_ROOT/dataset/make_small_motion_v3.py"
    exit 1
fi
[ ! -f "$CALIB"   ] && { echo "HATA: calib.txt yok: $CALIB";    exit 1; }
[ ! -f "$WEIGHTS" ] && { echo "HATA: model ağırlıkları yok: $WEIGHTS"; exit 1; }

frame_count=$(ls "$IMAGEDIR"/*.png 2>/dev/null | wc -l)
echo "[run_droid_v3_s1] Dataset frame sayısı : $frame_count"
echo "[run_droid_v3_s1] Beklenen SLAM frame  : $frame_count  (stride=1, hepsi)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate droid_clean
echo "[run_droid_v3_s1] Python: $(which python)"

cd "$DROID_DIR"
echo "[run_droid_v3_s1] demo.py çalıştırılıyor (stride=1)..."

python demo.py \
    --imagedir        "$IMAGEDIR" \
    --calib           "$CALIB"    \
    --weights         "$WEIGHTS"  \
    --trajectory_path "$TRAJ_OUT" \
    --stride          1           \
    --disable_vis

echo "[run_droid_v3_s1] Tamamlandı."
if [ -f "$TRAJ_OUT" ]; then
    echo "[run_droid_v3_s1] --- İlk 5 satır ---"
    head -6 "$TRAJ_OUT"
fi
echo ""
echo "[run_droid_v3_s1] Sonraki adım:"
echo "  python3 $REPO_ROOT/slam/scripts/evaluate_figure8_v3_stride1.py"
