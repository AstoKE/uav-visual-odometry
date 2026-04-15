#!/usr/bin/env bash
# run_gazebo.sh — Gazebo Sim'i temiz ortamda başlatır
# Kullanım: bash ~/code/uav-visual-odometry/sim/scripts/run_gazebo.sh

set -e

WORLD_FILE="$HOME/code/uav-visual-odometry/sim/worlds/slam_world.sdf"

echo "[run_gazebo] Conda/venv ortamı temizleniyor..."
conda deactivate 2>/dev/null || true
deactivate 2>/dev/null || true

unset LD_LIBRARY_PATH
unset PYTHONPATH
unset CUDA_HOME
unset VIRTUAL_ENV
unset CONDA_DEFAULT_ENV
unset CONDA_PREFIX

echo "[run_gazebo] ROS2 Jazzy sourcelaniyor..."
source /opt/ros/jazzy/setup.bash

echo "[run_gazebo] World: $WORLD_FILE"

if [ ! -f "$WORLD_FILE" ]; then
    echo "[run_gazebo] HATA: World dosyasi bulunamadi: $WORLD_FILE"
    exit 1
fi

echo "[run_gazebo] Gazebo baslatiliyor..."
exec gz sim -r "$WORLD_FILE"
