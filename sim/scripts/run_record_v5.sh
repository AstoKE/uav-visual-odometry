#!/usr/bin/env bash
# run_record_v5.sh — v5 realistic world ile Gazebo başlatır ve veri kaydeder
#
# Akış:
#   Terminal 1: bash sim/scripts/run_record_v5.sh          ← bu script (Gazebo)
#   Terminal 2: bash sim/scripts/run_record_v5.sh --record ← recorder + kamera
#
# Ya da tek komutla (arka planda Gazebo, ön planda hareket):
#   bash sim/scripts/run_record_v5.sh --auto
#
# Çıktı:
#   dataset/raw_v5/  →  make_small_motion_v5.py  →  dataset/small_motion_v5/
#
# Kullanım:
#   bash ~/code/uav-visual-odometry/sim/scripts/run_record_v5.sh
#   bash ~/code/uav-visual-odometry/sim/scripts/run_record_v5.sh --seed 99
#   bash ~/code/uav-visual-odometry/sim/scripts/run_record_v5.sh --auto
#   bash ~/code/uav-visual-odometry/sim/scripts/run_record_v5.sh --auto --frames 900

set -e

REPO="$HOME/code/uav-visual-odometry"
V5_SDF="$REPO/sim/worlds/slam_world_v5_realistic.sdf"
RAW_DIR="$REPO/dataset/raw_v5"
SEED="42"
N_FRAMES="900"
MODE="gazebo"   # gazebo | record | auto

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)   SEED="$2";     shift 2 ;;
    --frames) N_FRAMES="$2"; shift 2 ;;
    --record) MODE="record"; shift ;;
    --auto)   MODE="auto";   shift ;;
    -h|--help)
      grep '^#' "$0" | head -20 | sed 's/^# \?//'; exit 0 ;;
    *) echo "Bilinmeyen argüman: $1"; exit 1 ;;
  esac
done

# ── v5 SDF yoksa üret ─────────────────────────────────────────────────────────
if [ ! -f "$V5_SDF" ]; then
  echo "[run_record_v5] v5 SDF bulunamadı, üretiliyor (seed=$SEED)..."
  python3 "$REPO/sim/scripts/gen_realistic_world_v5.py" \
    --seed "$SEED" --out "$V5_SDF"
  echo "[run_record_v5] SDF hazır: $V5_SDF"
fi

# ─────────────────────────────────────────────────────────────────────────────
# MOD: gazebo — sadece Gazebo'yu başlat (Terminal 1)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$MODE" == "gazebo" ]]; then
  echo "[run_record_v5] Conda ortamı temizleniyor..."
  conda deactivate 2>/dev/null || true
  deactivate 2>/dev/null || true
  unset LD_LIBRARY_PATH PYTHONPATH CUDA_HOME VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_PREFIX

  echo "[run_record_v5] ROS2 Jazzy sourcelaniyor..."
  source /opt/ros/jazzy/setup.bash

  echo "[run_record_v5] ============================================"
  echo "[run_record_v5] Gazebo v5 başlatılıyor"
  echo "[run_record_v5] World  : $V5_SDF"
  echo "[run_record_v5] Kayıt  : $RAW_DIR"
  echo "[run_record_v5] ============================================"
  echo "[run_record_v5] Gazebo başladıktan sonra Terminal 2'de çalıştır:"
  echo "  bash $REPO/sim/scripts/run_record_v5.sh --record"
  echo ""
  exec gz sim -r "$V5_SDF"
fi

# ─────────────────────────────────────────────────────────────────────────────
# MOD: record — recorder + move_camera (Terminal 2)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$MODE" == "record" ]]; then
  echo "[run_record_v5] ROS2 Jazzy sourcelaniyor..."
  source /opt/ros/jazzy/setup.bash

  mkdir -p "$RAW_DIR"
  echo "[run_record_v5] Recorder başlatılıyor → $RAW_DIR"

  # image_recorder artık SAVE_DIR env'i destekliyor
  SAVE_DIR="$RAW_DIR" /usr/bin/python3 \
    "$REPO/ros/nodes/image_recorder.py" &
  RECORDER_PID=$!
  echo "[run_record_v5] Recorder PID: $RECORDER_PID"

  sleep 3
  echo "[run_record_v5] Kamera hareketi başlıyor (competition pattern)..."
  PATTERN=competition HEIGHT=4.0 \
    python3 "$REPO/sim/scripts/move_camera.py"

  echo "[run_record_v5] Hareket tamamlandı. Recorder durduruluyor..."
  kill "$RECORDER_PID" 2>/dev/null || true
  wait "$RECORDER_PID" 2>/dev/null || true

  FRAME_COUNT=$(ls "$RAW_DIR"/*.png 2>/dev/null | wc -l)
  echo ""
  echo "[run_record_v5] ============================================"
  echo "[run_record_v5] Kayıt tamamlandı!"
  echo "[run_record_v5] Kaydedilen frame : $FRAME_COUNT"
  echo "[run_record_v5] Dizin            : $RAW_DIR"
  echo "[run_record_v5] ============================================"
  echo ""
  echo "Sonraki adım:"
  echo "  python3 $REPO/dataset/make_small_motion_v5.py"
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# MOD: auto — Gazebo arka planda, recorder + hareket ön planda
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$MODE" == "auto" ]]; then
  echo "[run_record_v5] Conda ortamı temizleniyor..."
  conda deactivate 2>/dev/null || true
  deactivate 2>/dev/null || true
  unset LD_LIBRARY_PATH PYTHONPATH CUDA_HOME VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_PREFIX

  echo "[run_record_v5] ROS2 Jazzy sourcelaniyor..."
  source /opt/ros/jazzy/setup.bash

  echo "[run_record_v5] AUTO mod — Gazebo arka planda başlatılıyor..."
  gz sim -r "$V5_SDF" > /tmp/gazebo_v5.log 2>&1 &
  GAZEBO_PID=$!
  echo "[run_record_v5] Gazebo PID: $GAZEBO_PID (log: /tmp/gazebo_v5.log)"
  echo "[run_record_v5] 10 saniye bekleniyor (Gazebo yüklensin)..."
  sleep 10

  mkdir -p "$RAW_DIR"
  echo "[run_record_v5] Recorder başlatılıyor → $RAW_DIR"
  SAVE_DIR="$RAW_DIR" /usr/bin/python3 \
    "$REPO/ros/nodes/image_recorder.py" > /tmp/recorder_v5.log 2>&1 &
  RECORDER_PID=$!
  echo "[run_record_v5] Recorder PID: $RECORDER_PID"

  sleep 3
  echo "[run_record_v5] Kamera hareketi başlıyor..."
  PATTERN=competition HEIGHT=4.0 \
    python3 "$REPO/sim/scripts/move_camera.py" 2>&1 | tee /tmp/move_v5.log

  echo "[run_record_v5] Hareket tamamlandı. Kapatılıyor..."
  kill "$RECORDER_PID" 2>/dev/null || true
  kill "$GAZEBO_PID"   2>/dev/null || true
  pkill -x gz 2>/dev/null || true
  wait "$RECORDER_PID" 2>/dev/null || true

  FRAME_COUNT=$(ls "$RAW_DIR"/*.png 2>/dev/null | wc -l)
  echo ""
  echo "[run_record_v5] ============================================"
  echo "[run_record_v5] AUTO kayıt tamamlandı!"
  echo "[run_record_v5] Kaydedilen frame : $FRAME_COUNT"
  echo "[run_record_v5] Dizin            : $RAW_DIR"
  echo "[run_record_v5] ============================================"
  echo ""
  echo "Sonraki adımlar:"
  echo "  python3 $REPO/dataset/make_small_motion_v5.py"
  echo "  bash $REPO/slam/scripts/run_droid_figure8_v5.sh"
  exit 0
fi
