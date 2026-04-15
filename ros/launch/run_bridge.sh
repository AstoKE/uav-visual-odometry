#!/usr/bin/env bash
# run_bridge.sh — Gazebo <-> ROS2 kamera köprüsünü başlatır
# Kullanım: bash ~/code/uav-visual-odometry/ros/launch/run_bridge.sh
#
# Ön koşul: Gazebo (run_gazebo.sh) zaten çalışıyor olmalı.

set -e

echo "[run_bridge] ROS2 Jazzy sourcelaniyor..."
source /opt/ros/jazzy/setup.bash

echo "[run_bridge] ros_gz_bridge baslatiliyor: /down_camera (Image)..."
exec ros2 run ros_gz_bridge parameter_bridge \
    /down_camera@sensor_msgs/msg/Image@gz.msgs.Image
