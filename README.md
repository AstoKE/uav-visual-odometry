UAV Visual Position Estimation (GPS-Denied)
TEKNOFEST / Havacılıkta Yapay Zeka Yarışması – Görev 2
1. Proje Amacı

Bu proje, uydu tabanlı konumlandırma sistemlerinin (GPS vb.) devre dışı kaldığı durumlarda görsel veriye dayalı pozisyon kestirimi yapılmasını amaçlamaktadır.

Yarışma kapsamında hava aracının alt-görüş kamerasından alınan görüntüler kullanılarak aracın referans koordinat sistemindeki konum değişimleri hesaplanacaktır.

Pozisyon kestirimi için DROID-SLAM tabanlı bir Visual SLAM pipeline kullanılacaktır.

2. Problem Tanımı

Yarışma senaryosu:

Zaman	Durum
İlk 450 frame	Referans pozisyon sağlıklı
Sonraki 1800 frame	Referans pozisyon sağlıksız

Amaç:

Camera images → Position estimation → ΔX ΔY ΔZ

Sistem:

GPS olmadan çalışmalı

Kamera görüntüsünden konum kestirmeli

Referans pozisyon ile karşılaştırılabilir olmalı

3. Sistem Mimarisi
Genel Pipeline
Gazebo Simulation
        ↓
Downward Camera
        ↓
ROS2 Image Topic (/down_camera)
        ↓
Image Processing Pipeline
        ↓
DROID-SLAM
        ↓
Camera Trajectory
        ↓
ΔX ΔY ΔZ Computation
        ↓
Ground Truth Comparison
4. Sistem Bileşenleri
4.1 Simülasyon Ortamı

Kullanılan araçlar:

Tool	Amaç
Gazebo Sim	Drone simülasyonu
ROS2 Jazzy	Middleware
ros_gz_bridge	Gazebo-ROS bağlantısı

Simülasyonda:

downward camera

hareket eden drone

ground truth pose

yayınlanacaktır.

4.2 Görüntü Kaynağı

ROS2 Topic:

/down_camera

Frame rate:

≈ 30 Hz

Kamera:

Downward facing

RGB image

4.3 Visual SLAM

Algoritma:

DROID-SLAM

Neden DROID-SLAM:

State-of-the-art monocular SLAM

Robust pose estimation

Deep learning + optimization

Input:

image sequence

Output:

camera trajectory
5. Pozisyon Hesaplama

DROID-SLAM çıktısı:

Tcw = [R | t]

Buradan elde edilir:

x
y
z

İlk frame referans kabul edilir:

x0 y0 z0

Pozisyon farkı:

ΔX = x - x0
ΔY = y - y0
ΔZ = z - z0
6. Ground Truth

Simülasyon gerçek pozisyonu:

Gazebo PosePublisher

Topic:

/world/cam_world/pose/info

Bu veri kullanılarak:

ATE
RMSE

hesaplanacaktır.

7. Test Pipeline

Test aşamaları:

Aşama 1 – Simülasyon
Gazebo
↓
camera stream
Aşama 2 – Dataset oluşturma
ROS image topic
↓
image sequence
Aşama 3 – SLAM
images → DROID-SLAM → trajectory
Aşama 4 – Değerlendirme
estimated pose
↓
ground truth comparison
8. Repository Yapısı

Önerilen proje yapısı:

project_root
│
├── sim
│   ├── gazebo_world
│   └── drone_model
│
├── ros
│   ├── camera_bridge
│   └── ground_truth_node
│
├── slam
│   ├── droid_slam
│   └── slam_runner
│
├── dataset
│   ├── images
│   └── calibration
│
├── evaluation
│   ├── trajectory_compare
│   └── metrics
│
└── README.md
9. Kurulum (Overview)

Sistem gereksinimleri:

Ubuntu 24.04
ROS2 Jazzy
Gazebo Sim
CUDA GPU
PyTorch

Ana bileşenler:

ROS2
Gazebo
DROID-SLAM
10. Geliştirme Yol Haritası
Faz 1 – Simülasyon

Gazebo drone ortamı

downward camera

ROS image bridge

Faz 2 – Dataset üretimi
ROS camera → image sequence
Faz 3 – SLAM entegrasyonu
image sequence → DROID-SLAM
Faz 4 – Pozisyon hesaplama
trajectory → ΔX ΔY ΔZ
Faz 5 – Değerlendirme
estimate vs ground truth

Metricler:

RMSE

ATE

drift

11. Gelecek Çalışmalar

real-time ROS2 DROID-SLAM entegrasyonu

IMU fusion (Visual-Inertial SLAM)

multi-camera system

loop closure optimization

12. Referanslar

DROID-SLAM

https://github.com/princeton-vl/DROID-SLAM

ROS2

https://docs.ros.org

Gazebo

https://gazebosim.org
13. Takım Notları

Bu proje TEKNOFEST Havacılıkta Yapay Zeka Yarışması kapsamında geliştirilmiştir.

Amaç:

GPS-denied UAV navigation
