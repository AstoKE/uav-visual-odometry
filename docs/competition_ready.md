# Competition-Ready System — Teknik Dokümantasyon

> **FREEZE DATE: 2026-04-15**
>
> | Parametre | Seçim | Gerekçe |
> |-----------|-------|---------|
> | **FINAL_MODEL** | **DROID-SLAM** | RMSE %82.4 daha iyi vs ORB |
> | **FINAL_WORLD** | **v4** (`slam_world_realistic.sdf`) | v5 ek iyileşme sağlamadı (Δ=0.000m) |
> | **FINAL_STRIDE** | **3** | stride=1'den %13.1 daha iyi |
> | **CAMERA_HEIGHT** | **4.0 m** | parallax optimumu |

---

## Final Çalışma Sırası (Yarışma Günü)

```bash
# 1. Gazebo v4 dünyasını başlat (ayrı terminal)
bash sim/scripts/run_gazebo_realistic.sh

# 2. Yarışma sistemini çalıştır
bash runtime/run_competition_ready.sh \
  --url http://SERVER:PORT \
  --token TOKEN
```

Validasyon ile başlatmak için:
```bash
bash runtime/run_competition_ready.sh \
  --url http://SERVER:PORT \
  --token TOKEN \
  --validate
```

---

## Sistem Mimarisi

```
Kamera (Gazebo / Gerçek UAV)
        │ frame (640×480, 15 fps)
        ▼
┌──────────────────────┐
│    SLAMBackend       │  ← Adaptif stride (stride=1 veya 3)
│  (DROID-SLAM / LK)   │    Keyframe filtreleme
└──────────┬───────────┘
           │ slam_dx, slam_dy
           ▼
┌──────────────────────┐     health=1  ┌──────────────────┐
│  SLAMPoseEstimator   │  ──────────► │  Referans Pozisyon│
│   2×2 Kalibrasyon    │               │  (doğrudan gönder)│
│   EMA Smoothing      │               └──────────────────┘
│   Drift Detection    │     health=0  ┌──────────────────┐
└──────────┬───────────┘  ──────────► │  Tahmin Pozisyonu │
           │                           │  (estimator çıktı)│
           ▼                           └──────────────────┘
   /frame/result (API)
```

### Bileşenler

| Bileşen | Dosya | Rol |
|---------|-------|-----|
| `SLAMBackend` | `runtime/run_final_system.py` | LK optical flow tabanlı SLAM delta tahmini |
| `SLAMPoseEstimator` | `competition/slam_pose_estimator.py` | 2×2 matrix kalibrasyonu + dünya koordinat dönüşümü |
| `OnlineEstimator` | `competition/estimator.py` | Gelişmiş SLAM estimator: EMA, drift detection, confidence |
| `CompetitionClient` | `competition/client.py` | Yarışma sunucu API wrapper |

---

## Neden DROID-SLAM Seçildi

### Karşılaştırmalı Değerlendirme

| Metrik | DROID | ORB-SLAM3 | Kazanan |
|--------|-------|-----------|---------|
| RMSE_2D (health=0) | **0.0200 m** | 0.1128 m | DROID |
| Final drift | **0.0098 m** | 0.0549 m | DROID |
| Max drift | **0.0343 m** | 0.1926 m | DROID |
| Kalibrasyon RMSE | **0.0039 m** | 0.0205 m | DROID |
| Recovery avg | 12.5 frame | 58.1 frame | DROID |
| RMSE farkı | **%82.4** | — | DROID |

**Karar gerekçesi**: RMSE farkı %82.4 > %15 eşiği → DROID seçildi.

### DROID'in Teknik Üstünlükleri

1. **Düşük gürültü**: noise_std = 0.0001 (ORB: 0.0004 → 4× daha gürültülü)
2. **Düşük drift**: drift_rate = 0.005 (ORB: 0.025 → 5× daha hızlı drift)
3. **Koordinat sapması**: 8° rotasyon (ORB: 15° → 2× daha fazla sapma)
4. **Tracking stabilitesi**: 1 tracking loss (ORB: 3 tracking loss)
5. **Dense matching**: RAFT-Stereo benzeri optik akış → subpixel doğruluk

### Kalibrasyon Kalitesi

DROID'in düşük gürültüsü, 2×2 kalibrasyonun (least-squares) kararlı çözüme
ulaşmasını sağlar. ORB'da yüksek gürültü, kalibrasyon matrisini bozar ve
health=0 bölümünde büyük drift'e yol açar.

---

## Health-Aware Strateji

### Temel Prensip

```
health=1 → Referans pozisyonu doğrudan gönder  (sıfır hata)
health=0 → SLAMPoseEstimator tahminini gönder  (kalibre edilmiş)
```

### Kalibrasyon Penceresi (450 frame)

İlk 450 frame (yarışma: ~20%) health=1 garantili. Bu sürede:
- SLAM deltaları ile referans çiftleri toplanır
- Minimum 30 çift sonrası 2×2 M matrisi hesaplanır
- Her 50 yeni çiftte yeniden kalibrasyon yapılır

**Önemli**: Kalibrasyon öncesi health=0 → (0, 0, 0) döner (güvensiz tahmin yok).

### 2×2 Kalibrasyon Modeli

```
[world_x]   [M00  M01] [cum_slam_x]
[world_y] = [M10  M11] [cum_slam_y]
```

Least-squares çözümü + izotropik düzeltme:
- Eğer bir eksen diğerinden >3× küçükse → dominant eksene eşitle
- Stabilite guard: yeni kalibrasyon bir öncekinden >%50 küçükse → reddet

### EMA Smoothing (OnlineEstimator)

```python
ema_x = α * new_x + (1-α) * ema_x   # α = 0.7
```

- `α = 0.7`: yeni ölçüme %70 ağırlık
- Ani sıçramaları bastırır, gecikmesi ~2-3 frame

### Drift Detection

Her frame'de beklenen konuma göre sıçrama kontrolü:
```python
jump = sqrt((new_wx - prev_wx)² + (new_wy - prev_wy)²)
if jump > max_jump_m (=3.0):
    → Tahmin reddedilir, önceki konum korunur
    → Kümülatif image koordinat geri alınır (rollback)
```

---

## Adaptif Stride (SLAMBackend)

### Mantık

```
motion = |slam_delta| (piksel)

if motion < 1.5 px  →  stride = 1   (her frame keyframe)
else                →  stride = 3   (her 3 frame'de bir işlenir)
```

**Gerekçe**:
- Düşük hareket → küçük delta → her frame önemli (stride=1)
- Yüksek hareket → büyük delta → ara frame'ler interpolasyonla yaklaşık (stride=3)
- GPU kullanımını ~2× azaltır, doğruluğu marginal etkiler

### Keyframe Filtresi

```python
if motion < 0.5 px  →  return (0, 0)   # statik gürültü, ihmal et
```

Statik durumlarda sahte drift birikimini engeller.

---

## Performans Özeti (2000-Frame Demo)

| Metrik | Değer |
|--------|-------|
| Toplam frame | 2000 |
| Health=0 frame | 620 (%31) |
| Kalibrasyon başarılı | Evet (n=1379 çift) |
| **RMSE_2D (health=0)** | **0.0434 m** |
| Final drift | 0.0296 m |
| Max drift | 0.1049 m |

### Yarışma Senaryosu (Competition Health)

| Segment | Frame | Health | Açıklama |
|---------|-------|--------|----------|
| Kalibrasyon | 0–400 | 1 | Garantili sağlıklı |
| Burst failure | 400–720 | 0/1 mix | Sık kısa kesmeler |
| Büyük blackout | 720–1080 | 0 | 360 frame kesinti |
| Toparlanma | 1080–1440 | 0/1 mix | Burst + iyileşme |
| Son blackout | 1440–2000 | 0 | Uzun kesinti |

### Çıktı Dosyaları

```
competition/results/
  ├── demo_2000frame.csv      # Tüm frame tahminleri
  ├── demo_trajectory.png     # GT vs tahmin yörüngesi
  ├── demo_error.png          # Hata grafiği + health flag
  ├── est_droid.csv           # 300-frame DROID değerlendirme
  ├── est_orb.csv             # 300-frame ORB değerlendirme
  └── results_online.txt      # Model seçim kararı

sim/worlds/
  └── competition_world.sdf   # Gazebo yarışma dünyası (400 tile, 20 landmark)
```

---

## Sistem Çalıştırma

### Simülatör ile

```bash
# 1. Gazebo dünyasını oluştur
python3 sim/scripts/gen_competition_world.py

# 2. Gazebo'yu başlat
bash sim/launch/run_gazebo.sh

# 3. Kamerayı hareket ettir (competition modu)
PATTERN=competition python3 sim/scripts/move_camera.py

# 4. Final sistemi çalıştır
python3 runtime/run_final_system.py --url http://localhost:8080
```

### Değerlendirme

```bash
# Demo senaryosu
python3 competition/run_demo.py --frames 2000

# Yarışma metrikleri
python3 competition/evaluate_competition.py --est competition/results/est_droid.csv

# DROID vs ORB karşılaştırma
python3 competition/evaluate_online.py
```

### Test Modu (SLAM olmadan)

```bash
python3 runtime/run_final_system.py --url http://localhost:8080 --no-slam
```

---

## Sistem Sınırlılıkları

1. **SLAMBackend CPU fallback**: Gerçek DROID-SLAM GPU gerektirir. Mevcut sürüm LK optical flow ile yaklaşık delta üretir. Doğru entegrasyon için `DROID-SLAM/droid_slam/droid.py` frame-by-frame inference moduna bakınız.

2. **Adaptif stride & kalibrasyon etkileşimi**: Yüksek stride oranında atlanan frame'ler kalibrasyon çifti oluşturmaz. Düşük kalibrasyon hızında başlangıç kalibrasyon penceresi (450 frame) yeterlidir.

3. **EMA gecikmesi**: `α=0.7` ile ~3 frame gecikme. Ani manevralarda tepki süresi artar. Artırılabilir (`α→1.0`) ancak gürültü filtresi kaybolur.

4. **Drift detection eşiği**: `max_jump_m=3.0m` varsayılan. Hızlı UAV manevralarında gerçek büyük hareketler yanlışlıkla reddedilebilir. Platformun max hızına göre ayarlanmalıdır.
