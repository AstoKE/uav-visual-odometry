#!/usr/bin/env python3
"""
image_recorder.py — /down_camera topic'inden görüntü alıp PNG olarak kaydeder.

Kayıt dizini: ~/code/uav-visual-odometry/dataset/raw/
Dosya isimleri: 000000.png, 000001.png, ...

Çalıştırma:
    /usr/bin/python3 ~/code/uav-visual-odometry/ros/nodes/image_recorder.py

Ön koşul:
    - ROS2 Jazzy source edilmiş olmalı
    - ros_gz_bridge /down_camera topic'ini yayıyor olmalı
"""

import os
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# cv_bridge ROS2'nin sistem kütüphanesiyle gelir
try:
    from cv_bridge import CvBridge
except ImportError:
    print("[image_recorder] HATA: cv_bridge bulunamadi.")
    print("  Cozum: sudo apt install ros-jazzy-cv-bridge")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("[image_recorder] HATA: opencv-python bulunamadi.")
    print("  Cozum: sudo apt install python3-opencv")
    sys.exit(1)

# ── Sabit ayarlar ──────────────────────────────────────────────────────────────
TOPIC_NAME   = "/down_camera"
_DEFAULT_DIR = os.path.expanduser("~/code/uav-visual-odometry/dataset/raw")
SAVE_DIR     = os.environ.get("SAVE_DIR", _DEFAULT_DIR)
IMAGE_FORMAT = "png"
# ──────────────────────────────────────────────────────────────────────────────


class ImageRecorder(Node):
    def __init__(self):
        super().__init__("image_recorder")

        os.makedirs(SAVE_DIR, exist_ok=True)

        self.bridge   = CvBridge()
        self.counter  = self._find_start_index()

        self.subscription = self.create_subscription(
            Image,
            TOPIC_NAME,
            self._callback,
            10,
        )

        self.get_logger().info(
            f"ImageRecorder baslatildi. Topic: {TOPIC_NAME} | Kayit dizini: {SAVE_DIR}"
        )
        self.get_logger().info(f"Baslangic frame numarasi: {self.counter:06d}")

    def _find_start_index(self) -> int:
        """Mevcut PNG'lerin sayısını bularak devam numarasını döndürür."""
        if not os.path.isdir(SAVE_DIR):
            return 0
        existing = [
            f for f in os.listdir(SAVE_DIR)
            if f.endswith(f".{IMAGE_FORMAT}") and f[:-4].isdigit()
        ]
        if not existing:
            return 0
        return max(int(f[:-4]) for f in existing) + 1

    def _callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"imgmsg_to_cv2 hatasi: {e}")
            return

        filename = f"{self.counter:06d}.{IMAGE_FORMAT}"
        filepath = os.path.join(SAVE_DIR, filename)

        success = cv2.imwrite(filepath, cv_image)
        if not success:
            self.get_logger().error(f"Yazma hatasi: {filepath}")
            return

        if self.counter % 50 == 0:
            self.get_logger().info(f"Kaydedildi: {filename}  (toplam: {self.counter + 1})")

        self.counter += 1


def main() -> None:
    rclpy.init()
    node = ImageRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            f"Durduruldu. Toplam kaydedilen frame: {node.counter}"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
