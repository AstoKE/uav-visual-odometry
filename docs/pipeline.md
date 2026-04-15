# UAV Visual Odometry — Pipeline

## Genel Akış

```
Gazebo Sim  →  ros_gz_bridge  →  image_recorder  →  DROID-SLAM  →  evaluate
(slam_world.sdf)   (/down_camera)   (dataset/raw_*)    (trajectory.csv)   (RMSE)
```

---

## v3 Kurulum Özeti (Güncel)

| Parametre        | v1 / v2           | **v3 (güncel)**          |
|------------------|-------------------|--------------------------|
| Kamera yüksekliği | 10 m             | **4 m**                  |
| Kalibrasyon fx=fy | 452              | **457**                  |
| Hareket deseni   | lawnmower / fig-8 | **figure-8**             |
| Figure-8 Ax × Ay | 6.0m × 5.0m      | **4.5m × 3.5m**          |
| Tur süresi / sayı | 30s × 3          | **25s × 3.5**            |
| Sahne             | 12 obje          | **77 obje + 41 zemin tile** |
| En iyi RMSE_2D    | ~5.52 m          | *kayıt sonrası güncellenecek* |

---

## Adım Adım Çalıştırma (v3)

Her adım **ayrı terminal**de çalıştırılır.

---

### Terminal 1 — Gazebo

```bash
bash ~/code/uav-visual-odometry/sim/scripts/run_gazebo.sh
```

- Zengin sahne: 77 obje + renkli zemin döşemeleri
- Kamera `down_cam_rig` 4m yüksekte, zemine bakar (`/down_camera` @ 15Hz)
- FOV=1.2217 rad → fx=fy=457

---

### Terminal 2 — ROS2 Bridge

```bash
bash ~/code/uav-visual-odometry/ros/launch/run_bridge.sh
```

---

### Terminal 3 — Dataset Recorder (raw_v3)

```bash
OUTPUT_DIR=~/code/uav-visual-odometry/dataset/raw_v3 \
  python3 ~/code/uav-visual-odometry/ros/nodes/image_recorder.py
```

---

### Terminal 4 — Kamera Hareketi (Figure-8 v3)

```bash
PATTERN=figure8 python3 ~/code/uav-visual-odometry/sim/scripts/move_camera.py
```

- Ax=4.5m, Ay=3.5m, T=25s × 3.5 tur ≈ 87.5s
- `~1312` hareket frame'i @ 15Hz
- Biterken otomatik durur; recorder'ı Ctrl+C ile durdur

---

### Adım 5 — Small Dataset Oluştur

```bash
python3 ~/code/uav-visual-odometry/dataset/make_small_motion_v3.py
```

- `dataset/raw_v3/` → frame 20-919 → `dataset/small_motion_v3/` (900 frame)
- SLAM stride=3 → ~300 efektif frame

---

### Adım 6 — DROID-SLAM

```bash
bash ~/code/uav-visual-odometry/slam/scripts/run_droid_figure8_v3.sh
```

- `conda activate droid_clean` otomatik yapılır
- Çıktı: `slam/outputs/trajectory_figure8_v3.csv`

---

### Adım 7 — Evaluation

```bash
python3 ~/code/uav-visual-odometry/slam/scripts/evaluate_figure8_v3.py
```

Üretilen dosyalar:

| Dosya | Açıklama |
|-------|----------|
| `evaluation/metrics/rmse_figure8_v3.txt` | RMSE raporu |
| `evaluation/plots/trajectory_figure8_v3.png` | GT vs SLAM XY |
| `evaluation/plots/rmse_comparison_v3.png` | Tüm sürümler karşılaştırması |
| `evaluation/plots/gt_figure8_v3_shape.png` | GT figure-8 şekli |

---

## Hızlı Kontroller

| Kontrol | Komut |
|---------|-------|
| Gazebo çalışıyor mu? | `gz topic -l \| grep down_camera` |
| Kaç frame kaydedildi? | `ls dataset/raw_v3/*.png \| wc -l` |
| DROID-SLAM env hazır? | `conda activate droid_clean && python -c "import droid_backends; print('OK')"` |
| Trajectory var mı? | `head -3 slam/outputs/trajectory_figure8_v3.csv` |

---

## Dosya Yapısı (v3 güncel)

```
uav-visual-odometry/
├── sim/
│   ├── worlds/slam_world.sdf           # 77 obje + 41 zemin tile, 4m kamera
│   └── scripts/
│       ├── run_gazebo.sh               # Gazebo başlatıcı
│       ├── move_camera.py              # figure-8 (varsayılan) / lawnmower
│       └── export_ground_truth.py      # SLAM GT üretici (pattern env ile)
├── ros/
│   ├── nodes/image_recorder.py         # ROS2 image → PNG (OUTPUT_DIR env)
│   └── launch/
│       ├── run_bridge.sh
│       └── run_recorder.sh
├── dataset/
│   ├── raw/                            # Orijinal lawnmower kaydı
│   ├── raw_v3/                         # Figure-8 v3 kaydı (yeni)
│   ├── small_motion/                   # Lawnmower SLAM dataset
│   ├── small_motion_v3/                # Figure-8 v3 SLAM dataset (yeni)
│   ├── meta/calib.txt                  # 457 457 320 240  (v3: fx=457)
│   ├── make_small_motion_v3.py         # raw_v3 → small_motion_v3
│   └── make_small_motion_v2.py         # raw → small_motion_v2
├── slam/
│   ├── scripts/
│   │   ├── run_droid_figure8_v3.sh     # v3 SLAM çalıştırıcı
│   │   ├── run_droid_figure8.sh        # v2 SLAM çalıştırıcı
│   │   ├── run_droid_small.sh          # orijinal SLAM çalıştırıcı
│   │   ├── evaluate_figure8_v3.py      # v3 tam evaluation pipeline
│   │   ├── evaluate_figure8.py         # v2 evaluation
│   │   ├── transform_to_world.py       # koordinat dönüşümü
│   │   ├── apply_scale.py              # scale uygulama
│   │   └── trajectory_to_delta.py      # delta hesaplama
│   └── outputs/
│       ├── trajectory_figure8_v3.csv   # (SLAM sonrası)
│       └── ...
├── evaluation/
│   ├── ground_truth_figure8_v3.csv     # (eval sonrası)
│   ├── metrics/
│   └── plots/
├── DROID-SLAM/
└── docs/pipeline.md                    # Bu dosya
```

---

## Kalibrasyon Notu (v3)

`dataset/meta/calib.txt` formatı: `fx fy cx cy`

```
horizontal_fov = 1.2217 rad
fx = fy = (width/2) / tan(fov/2)
         = 320 / tan(0.61085)
         = 320 / 0.70021
         ≈ 457

calib.txt → 457 457 320 240
```

Kamera yüksekliği (4m vs 10m) kalibrasyon değerlerini değiştirmez —
fx/fy yalnızca FOV ve çözünürlüğe bağlıdır.

---

## RMSE Geçmişi

| Sürüm | Sahne | Yükseklik | Pattern | RMSE_2D |
|-------|-------|-----------|---------|---------|
| Baseline | 12 obje | 10m | lawnmower | 12.98 m |
| + world frame | 12 obje | 10m | lawnmower | 7.85 m |
| + axis scale  | 12 obje | 10m | lawnmower | 5.69 m |
| figure-8 v2   | 12 obje | 10m | figure-8  | ~5.52 m |
| **figure-8 v3** | **77 obje, 4m** | **4m** | **figure-8** | *TBD* |
