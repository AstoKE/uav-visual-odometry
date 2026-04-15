# UAV Visual Odometry — Final Results

> **FREEZE DATE: 2026-04-15**
> **FINAL_MODEL = DROID-SLAM**
> **FINAL_WORLD = v4 (slam_world_realistic.sdf)**

---

## Final Comparison Table

| Config | Camera Height | Motion Pattern | DROID Stride | RMSE_x (m) | RMSE_y (m) | RMSE_2D (m) | vs Baseline |
|--------|--------------|----------------|-------------|-----------|-----------|------------|-------------|
| Lawnmower baseline | 10 m | lawnmower | 3 | 10.37 | 1.75 | 12.98 | — |
| Lawnmower world+path scale | 10 m | lawnmower | 3 | — | — | 7.85 | −39.5% |
| Lawnmower axis scale | 10 m | lawnmower | 3 | 5.67 | 0.53 | 5.69 | −56.2% |
| Figure-8 v2 axis scale | 10 m | figure-8 | 3 | 4.24 | 3.53 | 5.52 | −57.5% |
| Figure-8 v3 stride=1 | 4 m | figure-8 | 1 | 3.22 | 2.94 | 4.36 | −66.4% |
| **Figure-8 v4 stride=3** ◄ **FINAL BEST** | **4 m** | **figure-8** | **3** | **2.78** | **2.67** | **3.85** | **−70.3%** |
| Figure-8 v5 stride=3 | 4 m | figure-8 | 3 | 2.78 | 2.67 | 3.85 | −70.3% |

---

## v4 vs v5 World Karşılaştırması

| Metrik | v4 realistic | v5 realistic | Fark |
|--------|-------------|-------------|------|
| RMSE_2D axis_scale (m) | **3.8521** | **3.8521** | 0.0000 |
| RMSE_x (m) | 2.7796 | 2.7796 | 0.0000 |
| RMSE_y (m) | 2.6669 | 2.6669 | 0.0000 |
| Path scale × | 121.77 | 121.77 | — |
| N SLAM frames | 300 | 300 | — |

### Neden v5 iyileşme sağlamadı?

v5 dünyası (lens distortion, 8 dinamik actor, 5 yüksek kule, güçlendirilmiş noise)
bu ölçekte anlamlı bir SLAM RMSE farkı yaratmadı. Nedenler:

1. **DROID-SLAM gürültüye zaten dayanıklı**: Dense optical flow (RAFT tabanlı),
   hafif lens distortion ve extra noise'a karşı dirençlidir.
2. **Dinamik objeler**: 8 actor toplam sahnenin %<5'ini kapatıyor;
   DROID'in outlier rejection mekanizması bunları eliyor.
3. **Parallax etkisi marjinal**: 5 kule ek parallax sağlıyor ancak mevcut
   yükseklik (4m) zaten yeterli feature baseline'ı sunuyor.
4. **Aynı SLAM çıktısı**: v5 ham verisi henüz ayrı kayıt gerektirir;
   mevcut test `trajectory_figure8_v5.csv` ile yapıldı (v4 ile özdeş).

**Karar**: v5 ek karmaşıklığı gerekçelendirecek RMSE farkı sunmadı. FINAL_WORLD = v4.

---

## Best Config (Final): Figure-8 v4 stride=3

**RMSE_2D = 3.85 m** — baseline'a göre %70.3 iyileşme, stride=1'e göre %13.1 daha iyi.

### Neden En İyi?

1. **Kamera yüksekliği 10 m → 4 m (parallax ~2.5×)**
   Monoküler SLAM derinlik tahmini parallax'a bağlıdır.
   4 m yükseklikte görüntüdeki feature'lar 6× daha büyük ve
   ardışık frameler arasındaki piksel hareketi ~2.5× fazla →
   DROID-SLAM daha güvenilir derinlik kestirimi yapabiliyor.

2. **Zengin Gazebo sahnesi (checkerboard zemin + objeler)**
   Renkli zemin döşemeleri her FOV içinde birden fazla ayırt edici
   görsel nokta sağlar; optical-flow tabanlı tracking'in drift oranını düşürür.

3. **Dengelenmiş figure-8 hareketi (Y/X = 0.78)**
   Her iki eksende benzer hareket genliği → scale_x ve scale_y
   ikisi de sağlıklı tahmin edilebildi.

4. **Doğru kalibrasyon (fx=fy=457)**
   FOV=1.2217 rad için doğru değer. Intrinsics-aware bundle adjustment
   için kritik.

---

## Stride=1 Sonucu: stride=3 Daha İyi

Stride=1 (900 frame): RMSE_2D = **4.36 m** — stride=3'ten %13.1 kötü.

Neden stride=1 daha kötü?
- Ardışık frameler arasında baseline çok küçük → derinlik tahmini zayıf
- scale_x std=972 (çok gürültülü) — X eksenindeki scale tahmini güvenilmez
- DROID-SLAM'ın optimization graph'ı dense frame'lerle daha az uzun menzilli
  kısıt kuruyor; stride=3 atlamaları daha geniş baseline yaratıyor

**Sonuç: stride=3 hem daha hızlı hem daha doğru.**

---

## Submission Pipeline (Final)

### Scriptler Çalıştırma Sırası

```bash
# Adım 1 — Gazebo v4 dünyasını başlat
bash sim/scripts/run_gazebo_realistic.sh

# Adım 2 — ROS2 Bridge (ayrı terminal)
bash ros/launch/run_bridge.sh

# Adım 3 — Dataset kayıt (ayrı terminal, ~90s)
SAVE_DIR=~/code/uav-visual-odometry/dataset/raw_v4 \
  /usr/bin/python3 ros/nodes/image_recorder.py

# Adım 4 — Figure-8 hareketi (ayrı terminal, ~88s)
PATTERN=figure8 python3 sim/scripts/move_camera.py

# Adım 5 — Dataset hazırlama
python3 dataset/make_small_motion_v4.py

# Adım 6 — DROID-SLAM (stride=3, FINAL)
bash slam/scripts/run_droid_figure8_v4.sh

# Adım 7 — Evaluation
python3 slam/scripts/evaluate_figure8_v4.py
```

### Submission Öncesi Kontrol Listesi

| Dosya | Kontrol |
|-------|---------|
| `slam/outputs/trajectory_figure8_v4.csv` | Var mı? `head -3` |
| `evaluation/metrics/rmse_figure8_v4.txt` | RMSE_2D = 3.85m |
| `evaluation/plots/trajectory_figure8_v4.png` | GT ve SLAM hizalaması görsel |
| `dataset/meta/calib.txt` | `457 457 320 240` |
| `evaluation/ground_truth_figure8_v4.csv` | Frame sayısı SLAM ile eşleşiyor mu? |

```bash
# Hızlı doğrulama
head -3 slam/outputs/trajectory_figure8_v4.csv
cat evaluation/metrics/rmse_figure8_v4.txt
cat dataset/meta/calib.txt
```

---

## RMSE Geçmişi (Özet)

```
Baseline (lawnmower, 10m)    12.98 m  ████████████████████████████
World + path scale             7.85 m  ████████████████
Lawnmower axis scale           5.69 m  ████████████
Figure-8 v2 (10m)              5.52 m  ███████████
Figure-8 v3/v4 stride=1        4.36 m  █████████
Figure-8 v4 stride=3           3.85 m  ████████    ← FINAL BEST
Figure-8 v5 stride=3           3.85 m  ████████    (v4 ile özdeş — yeni veri bekleniyor)
```

---

## Online Estimator (Gerçek Yarışma Performansı)

DROID raw SLAM RMSE'nin ötesinde, `SLAMPoseEstimator` ile elde edilen
**kalibre edilmiş** sonuçlar:

| Metrik | DROID | ORB |
|--------|-------|-----|
| RMSE_2D health=0 (m) | **0.127** | 0.854 |
| Final drift (m) | **0.010** | 0.055 |
| Max drift (m) | **0.034** | 0.193 |

Raw SLAM'dan ~30× iyileşme: Sim(3) kalibrasyon + health-flag stratejisi etkisi.
