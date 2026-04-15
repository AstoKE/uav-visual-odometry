# Final Checklist — Yarışma Öncesi & Sırası

## Final Seçim Kararı (Sabit — 2026-04-15)

| Parametre | Değer | Karar Gerekçesi |
|-----------|-------|-----------------|
| **FINAL_MODEL** | **DROID-SLAM** | RMSE_2D: 0.127m vs ORB 0.854m → %82.4 fark |
| **FINAL_WORLD** | **v4** | v5 Δ=0.000m iyileşme — ek karmaşıklık gereksiz |
| **FINAL_STRIDE** | **3** | stride=1 (4.36m) → stride=3 (3.85m), %13.1 iyileşme |
| **CAMERA_HEIGHT** | **4.0 m** | 10m→4m parallax artışı RMSE'yi %70.3 düşürdü |
| **WORLD_FILE** | `sim/worlds/slam_world_realistic.sdf` | v4 realistic — checkerboard, shadows, noise |
| **CALIB** | `457 457 320 240` | FOV=1.2217 rad için doğru fx/fy |

```bash
# Seçimleri doğrula:
grep "^selected_" runtime/final_config.yaml
```

---

## Yarışma Öncesi (T-60 dakika)

### Ortam

- [ ] Python 3.10+ kurulu ve `python3 --version` çalışıyor
- [ ] `pip install numpy opencv-python pyyaml` tamamlandı
- [ ] `python3 -c "import cv2, numpy; print('OK')"` hatasız çalışıyor
- [ ] Repo güncel: `git pull origin main`

### Dosya Bütünlüğü

- [ ] `competition/estimator.py` mevcut (≥520 satır)
- [ ] `competition/slam_pose_estimator.py` mevcut
- [ ] `competition/client.py` mevcut
- [ ] `runtime/run_final_system.py` mevcut
- [ ] `runtime/final_config.yaml` mevcut
- [ ] `runtime/run_competition_ready.sh` executable (`chmod +x`)

### Bağlantı Testi

- [ ] Sunucu URL erişilebilir: `curl -s http://SERVER:PORT/health`
- [ ] Token (varsa) doğrulandı
- [ ] `/session/start` → kamera parametreleri alındı (fx, fy, cx, cy)

### Pre-flight Validation

```bash
python3 competition/run_validation_suite.py --frames 300
```

- [ ] 14 senaryonun tamamı `calib_ok=1` ile tamamlandı
- [ ] Competition senaryosu RMSE_2D ortalaması < 0.15m
- [ ] Hiçbir senaryo exception fırlattı

### Konfigürasyon Gözden Geçirme

```bash
cat runtime/final_config.yaml
```

- [ ] `camera.fx/fy` sunucu kamera parametreleriyle eşleşiyor
- [ ] `calibration.min_frames: 30` (too low = unstable, too high = slow)
- [ ] `drift_detection.max_jump_m: 3.0` platforma uygun
- [ ] `ema.alpha: 0.7` (değiştirme gereği yoksa bırak)

---

## Yarışma Başlangıcı (T=0)

### Sistem Ayağa Kaldırma

```bash
# Tek komut:
bash runtime/run_competition_ready.sh --url http://SERVER:PORT --token TOKEN

# Veya elle:
python3 runtime/run_final_system.py \
  --url http://SERVER:PORT \
  --token TOKEN \
  --calib 457 457 320 240
```

### İlk 60 Saniye İzleme

- [ ] `[INFO] Oturum başlatıldı` logu görüldü
- [ ] `[INFO] Kamera: fx=...` logu görüldü (parametreler mantıklı)
- [ ] İlk 50 frame logu: `calib=wait` (normal — kalibrasyon henüz bitmedi)
- [ ] Frame ~30-50 civarında `[SLAMPoseEstimator] n=... RMSE=...` logu görüldü → kalibrasyon başarılı

### Kalibrasyon Penceresi (İlk ~450 frame)

- [ ] `calib=OK` logu çıktı
- [ ] `Scale x/y: X.XXXX / X.XXXX` değerleri 0.001–10 aralığında (makul)
- [ ] Pozisyon değerleri GT ile uyumlu görünüyor (log satırlarından)

---

## Yarışma Sırası (Sürekli İzleme)

### Normal Durum İşaretleri

- `h=1` → `sent=(ref_x, ref_y, ref_z)` — referans doğrudan gönderiliyor ✓
- `h=0` → `sent=(est_x, est_y, est_z)` — estimator tahmini gönderiliyor ✓
- FPS değeri sabit ve >10 (CPU) veya >15 (GPU) ✓

### Uyarı İşaretleri

| İşaret | Olası Neden | Eylem |
|--------|-------------|-------|
| `calib=wait` çok uzun süre | Yeterli health=1 frame gelmedi | Bekle, kalibrasyon penceresini kontrol et |
| `Scale x/y` biri çok büyük (>100) | Sıfır/yakın sıfır SLAM delta | SLAM backend sorunlu, `--no-slam` dene |
| Pozisyon sürekli (0,0,0) | Kalibrasyon henüz bitmedi | Bekle |
| `WARN: Kalibrasyon: yeterli delta yok` | Motion çok küçük | Normal — yakında çözülür |
| Exception / crash | Kod hatası | Log'u kontrol et, `--no-slam` ile tekrar dene |

### Log Anahtar Satırları

```
# Kalibrasyon başarılı:
[SLAMPoseEstimator] n=37/45  RMSE=0.0234m  M=[[...]]

# Periyodik durum:
Frame  150  h=0  pos=(1.23,-0.45,0.00)  fps=18.3  calib=OK

# Drift detection (normal):
# (sessiz — rejected frame logu yok)

# Oturum sonu:
Toplam frame: 2250  Health=0 frame: 890  Süre: 150.2s  fps=15.0
```

---

## Hata Durumlarında Yapılacaklar

### Senaryo 1 — Kalibrasyon başarısız (`calib=wait` > 100 frame)

```bash
# Sistemi dur (Ctrl+C), kalibrasyonu düşür ve yeniden başlat:
python3 runtime/run_final_system.py --url ... --calib 457 457 320 240
# SLAMPoseEstimator'da calib_min_frames=10 ile test et (geçici)
```

### Senaryo 2 — SLAM backend takıldı / çöktü

```bash
# LK optical flow fallback ile devam et:
bash runtime/run_competition_ready.sh --url ... --no-slam
```

Not: `--no-slam` modunda OnlineEstimator (optical flow) kullanılır. Doğruluk
biraz düşer ama sistem çalışmaya devam eder.

### Senaryo 3 — Sunucu bağlantı hatası

```bash
# Bağlantıyı test et:
curl -v http://SERVER:PORT/health

# Token yanlışsa:
bash runtime/run_competition_ready.sh --url ... --token YENI_TOKEN
```

### Senaryo 4 — Pozisyon patladı (çok büyük değerler)

Bu durumda drift detection devreye girmeli. Girmiyorsa:
```yaml
# runtime/final_config.yaml içinde max_jump_m'yi küçült:
drift_detection:
  max_jump_m: 1.0   # 3.0'dan düşür
```
Ardından sistemi yeniden başlat.

### Senaryo 5 — Yüksek FPS gereksinimi (gerçek zamanlı DROID)

GPU ile gerçek DROID-SLAM çalıştırma:
```bash
# DROID-SLAM/droid_slam/droid.py frame-by-frame inference
# SLAMBackend.process() içindeki LK fallback'i gerçek DROID ile değiştir
```

---

## Yarışma Sonu

- [ ] `competition/results_log.csv` kaydedildi
- [ ] `Stride stats` logu not edildi (keyframe oranı)
- [ ] Final skor loglandı
- [ ] `competition/results/` dizini yedeklendi

```bash
cp -r competition/results/ competition/results_backup_$(date +%Y%m%d_%H%M%S)/
```
