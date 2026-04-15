#!/usr/bin/env bash
# run_gazebo_realistic.sh — Realistic v4 world ile Gazebo başlatır
#
# Değişiklikler (v3 → v4 realistic):
#   - slam_world_realistic.sdf (checkerboard zemin, gölgeler, kamera noise)
#
# Kullanım:
#   bash ~/code/uav-visual-odometry/sim/scripts/run_gazebo_realistic.sh

set -e

WORLD_FILE="$HOME/code/uav-visual-odometry/sim/worlds/slam_world_realistic.sdf"

echo "[run_gazebo_realistic] Conda/venv ortamı temizleniyor..."
conda deactivate 2>/dev/null || true
deactivate 2>/dev/null || true

unset LD_LIBRARY_PATH
unset PYTHONPATH
unset CUDA_HOME
unset VIRTUAL_ENV
unset CONDA_DEFAULT_ENV
unset CONDA_PREFIX

echo "[run_gazebo_realistic] ROS2 Jazzy sourcelaniyor..."
source /opt/ros/jazzy/setup.bash

echo "[run_gazebo_realistic] World: $WORLD_FILE"
echo "[run_gazebo_realistic] Özellikler:"
echo "  - 12×12 siyah-beyaz checkerboard zemin (144 tile)"
echo "  - cast_shadows=true, ambient=0.40 (derinlik/edge)"
echo "  - Kamera Gaussian noise stddev=0.010"
echo "  - 196 model toplam (12 pillar + 4 disk + 12 kutu + 16 stripe + 8 köşe)"

if [ ! -f "$WORLD_FILE" ]; then
    echo "[run_gazebo_realistic] HATA: World dosyası bulunamadı."
    echo "  Önce çalıştır: python3 ~/code/uav-visual-odometry/sim/scripts/gen_realistic_world.py"
    exit 1
fi

echo "[run_gazebo_realistic] Gazebo başlatılıyor..."
exec gz sim -r "$WORLD_FILE"
