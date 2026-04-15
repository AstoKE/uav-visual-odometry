#!/usr/bin/env bash
# run_droid_small.sh — DROID-SLAM'i küçük dataset ile çalıştırır
# Kullanım: bash ~/code/uav-visual-odometry/slam/scripts/run_droid_small.sh
#
# Ön koşul:
#   - conda activate droid_clean yapılmış OLMAMALI (script kendisi yapar)
#   - DROID-SLAM derlenmiş ve droid_backends import edilebilir olmalı
#   - dataset/small en az 8 PNG içermeli

set -e

REPO_ROOT="$HOME/code/uav-visual-odometry"
DROID_DIR="$REPO_ROOT/DROID-SLAM"
IMAGEDIR="$REPO_ROOT/dataset/small"
CALIB="$REPO_ROOT/dataset/meta/calib.txt"
WEIGHTS="$DROID_DIR/checkpoints/droid.pth"
TRAJ_OUT="$REPO_ROOT/slam/outputs/trajectory.csv"

echo "[run_droid_small] ============================================"
echo "[run_droid_small] DROID-SLAM kucuk dataset kosu"
echo "[run_droid_small] imagedir : $IMAGEDIR"
echo "[run_droid_small] calib    : $CALIB"
echo "[run_droid_small] weights  : $WEIGHTS"
echo "[run_droid_small] traj out : $TRAJ_OUT"
echo "[run_droid_small] ============================================"

# Ön kontroller
if [ ! -d "$IMAGEDIR" ] || [ -z "$(ls -A "$IMAGEDIR"/*.png 2>/dev/null)" ]; then
    echo "[run_droid_small] HATA: dataset/small bos veya yok."
    echo "  Once calistir: bash $REPO_ROOT/dataset/make_small_dataset.sh"
    exit 1
fi

if [ ! -f "$CALIB" ]; then
    echo "[run_droid_small] HATA: calib.txt bulunamadi: $CALIB"
    exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
    echo "[run_droid_small] HATA: Model dosyasi bulunamadi: $WEIGHTS"
    exit 1
fi

# Frame sayısını raporla
frame_count=$(ls "$IMAGEDIR"/*.png 2>/dev/null | wc -l)
echo "[run_droid_small] Dataset'teki toplam frame: $frame_count"

# conda ortamını aktive et
echo "[run_droid_small] conda activate droid_clean..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate droid_clean

echo "[run_droid_small] Python: $(which python)"

# DROID-SLAM dizinine gir
cd "$DROID_DIR"

echo "[run_droid_small] demo.py calistiriliyor..."
python demo.py \
    --imagedir        "$IMAGEDIR" \
    --calib           "$CALIB"    \
    --weights         "$WEIGHTS"  \
    --trajectory_path "$TRAJ_OUT" \
    --disable_vis

echo "[run_droid_small] Tamamlandi."
echo "[run_droid_small] Trajectory: $TRAJ_OUT"

if [ -f "$TRAJ_OUT" ]; then
    echo "[run_droid_small] --- Ilk 5 satir ---"
    head -6 "$TRAJ_OUT"
fi
