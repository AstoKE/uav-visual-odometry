#!/usr/bin/env bash
# run_recorder.sh — Dataset kayıt node'unu sistem Python ile başlatır
# Kullanım: bash ~/code/uav-visual-odometry/ros/launch/run_recorder.sh
#
# Ön koşul:
#   - ROS2 Jazzy kurulu (/opt/ros/jazzy)
#   - ros_gz_bridge çalışıyor olmalı (run_bridge.sh)
#   - python3-opencv ve ros-jazzy-cv-bridge kurulu olmalı

set -e

NODE_SCRIPT="$HOME/code/uav-visual-odometry/ros/nodes/image_recorder.py"

echo "[run_recorder] ROS2 Jazzy sourcelaniyor..."
source /opt/ros/jazzy/setup.bash

if [ ! -f "$NODE_SCRIPT" ]; then
    echo "[run_recorder] HATA: Node script bulunamadi: $NODE_SCRIPT"
    exit 1
fi

echo "[run_recorder] image_recorder.py baslatiliyor..."
echo "[run_recorder] Durdurmak icin Ctrl+C"
exec /usr/bin/python3 "$NODE_SCRIPT"
