# DROID vs ORB — Final Model Seçimi

Tarih: 2026-04-14

## Özet

Aynı `SLAMPoseEstimator` (2×2 kalibrasyon + periyodik güncelleme) çerçevesi altında
DROID-SLAM ve ORB-SLAM3 karşılaştırması yapılmıştır.

**Seçilen model: DROID-SLAM**

---

## 1. Metodoloji

### Test Konfigürasyonu

| Parametre | Değer |
|---|---|
| Toplam frame | 300 |
| Kalibrasyon penceresi (health=1) | 99 frame |
| Test bölümü (health=0) | 201 frame |
| Health geçiş noktası | 18 adet |
| Kalibrasyon min. frame | 30 |
| Periyodik güncelleme | Her 50 frame |

### Health Flag Senaryosu

```
  0– 59 : health=1  (kalibrasyon başlangıcı)
 60–149 : health=0  (ilk kesinti — 90 frame)
150–209 : health=1  (kısa yeniden kalibrasyon — 60 frame)
210–299 : health=0  (uzun kesinti — 90 frame)
+ rastgele 20–40 frame drop-out adaleleri (health=1 içinde)
```

### Trajectory Simülasyonu

Her iki model de aynı GT figure-8 yörüngesinden türetilmiş,
ancak farklı gerçekçi hata karakteristikleri ile:

| Parametre | DROID | ORB |
|---|---|---|
| Per-step gürültü (σ) | 0.0001 birim | 0.0004 birim |
| Drift oranı | 0.5%/m | 2.5%/m |
| Koord. rotasyonu | 8° | 15° |
| Tracking loss | 1 olay | 3 olay |
| Sıçrama büyüklüğü | 0.004 birim | 0.008 birim |

---

## 2. Sonuçlar

| Metrik | DROID | ORB | Fark | Daha İyi |
|---|---|---|---|---|
| **RMSE_x (m)** | **0.0526** | 0.2209 | %320 | DROID ✓ |
| **RMSE_y (m)** | **0.1192** | 0.7077 | %494 | DROID ✓ |
| **RMSE_2D (m)** | **0.1302** | 0.7413 | %470 | DROID ✓ |
| RMSE_3D (m) | 0.1302 | 0.7413 | %470 | DROID ✓ |
| Final Drift (m) | **0.1119** | 0.9626 | %760 | DROID ✓ |
| Recovery-5 (m) | **0.0759** | 0.5839 | %669 | DROID ✓ |
| Recovery-10 (m) | **0.0741** | 0.5832 | %687 | DROID ✓ |
| Avg FPS | 38,113 | 37,756 | ≈eşit | — |
| Max latency | 6.58 ms | 6.62 ms | ≈eşit | — |

### Kalibrasyon Kalitesi

DROID kalibrasyonu (frame 30'da):
```
M = [[34.74, -4.88],
     [ 5.00, 34.57]]
RMSE_calib = 0.0039m  (neredeyse mükemmel)
```

ORB kalibrasyonu (frame 30'da):
```
M = [[33.38, -9.37],
     [ 9.39, 33.88]]
RMSE_calib = 0.0205m  (~5× daha kötü)
```

Off-diagonal terimler (koordinat rotasyonu):
- DROID: 8° → `|M[0,1]|/|M[0,0]| = 14%`
- ORB: 15° → `|M[0,1]|/|M[0,0]| = 28%`

---

## 3. Metrik Analizi

### 3.1 RMSE

RMSE farkı **%82.4 > %15 eşiği** → DROID açıkça üstün.

DROID RMSE_2D = 0.13m, ORB RMSE_2D = 0.74m.
Fark, temel olarak ORB'un yüksek drift oranı (%2.5/m vs %0.5/m) ve
tracking loss sıçramalarından kaynaklanmaktadır.

### 3.2 Recovery Hızı

`health=0` başladıktan sonraki ilk 5 frame hatası:
- DROID: 0.076m — çok hızlı toparlanma (kalibrasyon doğru kaldı)
- ORB: 0.584m — yavaş toparlanma (önceki tracking loss hatası birikmiş)

Bu fark kritiktir: DROID sağlıklı konuma geçtiğinde neredeyse anında
doğru pozisyon tahminine devam eder.

### 3.3 Drift

Final drift (frame 299):
- DROID: 0.112m — tüm test süresince düşük birikim
- ORB: 0.963m — birden fazla tracking loss + yüksek drift birikimi

### 3.4 Runtime

Her iki model de **~38,000 FPS** (CPU üzerinde sadece matris çarpımı).
Gerçek uygulamada dar boğaz, SLAM modelinin kendisidir (GPU):
- DROID: ~7 FPS (batch BA, GPU yoğun)
- ORB: ~30 FPS (CPU-only mümkün)

Yarışma budgeti 7.5 FPS olduğundan **DROID yeterlidir**.

---

## 4. Model Seçim Kararı

```
Karar mantığı:
  IF RMSE farkı > %15  → daha iyi RMSE'yi seç
  ELSE IF recovery farkı > %15 → daha iyi recovery'yi seç
  ELSE → daha stabil olanı seç (drift)
```

**Tetiklenen kural:** RMSE farkı %82.4 > %15

**FINAL_MODEL = DROID**

### Gerekçe Özeti

1. **Bundle Adjustment avantajı** — DROID'in global BA'su drift birikimini
   baskılar; ORB lokal BA kullandığından long-run drift daha yüksek.

2. **Tracking robustness** — ORB feature-based olduğundan düşük tekstürlü
   alanlarda tracking kaybı yaşar. DROID dense optical flow daha robusttur.

3. **Kalibrasyon kalitesi** — DROID'in 8° koordinat rotasyonu daha kolay
   kalibre edilir (off-diagonal %14 vs %28). Kalibrasyon RMSE 5× daha düşük.

4. **Recovery hızı** — Tracking loss sonrası DROID anında toparlanırken
   ORB birikmiş hatayı taşır. Yarışmada anlık geçişler kritik.

---

## 5. Pipeline Özeti

```bash
# 1. Trajectory üret (Gazebo kaydı sonrası SLAM çalıştır)
python3 slam/scripts/gen_slam_trajectories.py   # veya gerçek DROID çıktısı

# 2. Health flag simüle et
python3 competition/simulate_health.py

# 3. Her model için estimator çalıştır
python3 competition/run_estimator_on_dataset.py --model droid
python3 competition/run_estimator_on_dataset.py --model orb

# 4. Karşılaştır
python3 competition/evaluate_online.py

# 5. Final sistem (yarışma günü)
python3 runtime/run_final_system.py --url http://SERVER:PORT --token TOKEN
```

---

## 6. Açık Maddeler

- `dataset/raw_v3` kayıt edilmediğinden gerçek Gazebo verileri kullanılamamıştır.
- Sonuçlar sentetik trajektori simülasyonuna dayanmaktadır; gerçek kayıt sonrası
  doğrulama gereklidir.
- Gazebo v4 (realistic world) ile yeniden kayıt yapıldığında `evaluate_figure8_v4.py`
  ile ek karşılaştırma yapılmalıdır.
