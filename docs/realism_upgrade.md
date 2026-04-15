# Realism Upgrade — v3 → v4

## Özet

| Metrik | v3 (colored tiles) | v4 (realistic) | Fark |
|--------|-------------------|----------------|------|
| RMSE_x | 2.78 m | *TBD* | *TBD* |
| RMSE_y | 2.67 m | *TBD* | *TBD* |
| RMSE_2D | **3.85 m** | *TBD* | *TBD* |

> `python3 slam/scripts/evaluate_figure8_v4.py` çalıştırıldıktan sonra TBD alanları güncellenecek.

---

## Yapılan Değişiklikler

### 1. Zemin Texture: Siyah-Beyaz Checkerboard

**Öncesi (v3):** 41 renkli tile (kırmızı/mavi/sarı/yeşil/mor), 1m × 1m, koyu gri zemin üstünde.

**Sonrası (v4):** 12 × 12 = **144 tile**, siyah/beyaz alternating checkerboard, 0.88m × 0.88m (8 cm boşluk ile ayrılmış), 1m ızgara adımı.

- Beyaz tile: `ambient/diffuse = (0.91, 0.91, 0.91)` — parlak
- Siyah tile: `ambient/diffuse = (0.05, 0.05, 0.05)` — koyu

**Etki:** Renkli tile kontrastı ~0.3–0.5 (farklı hue, benzer brightness) iken siyah-beyaz
kontrast ~0.85–0.90. Checkerboard her 1m'de tekrarlayan keskin kenar üretir → optical-flow
feature extractor (DROID-SLAM'ın tracker'ı) bu kenarlarda çok daha fazla ve daha güçlü
nokta tespit eder. Ayrıca ızgara yapısı yer-altı simetrisi sayesinde homografi tahminine
yardımcı olur.

---

### 2. Işıklandırma: Gölgeler ve Azaltılmış Ambient

**Öncesi (v3):**
```xml
<ambient>1.0 1.0 1.0 1</ambient>
<shadows>false</shadows>
<cast_shadows>false</cast_shadows>
```

**Sonrası (v4):**
```xml
<ambient>0.40 0.40 0.40 1</ambient>
<shadows>true</shadows>

<!-- Güneş -->
<diffuse>1.0 0.98 0.92 1</diffuse>
<direction>-0.5 0.3 -1.0</direction>
<cast_shadows>true</cast_shadows>

<!-- Fill ışığı: zayıf, gölgesiz -->
<diffuse>0.30 0.32 0.35 1</diffuse>
```

**Etki:**
- Ambient=1.0 → her yüzey eşit aydınlık, kenar gradientleri sıfıra yakın.
- Ambient=0.4 + cast_shadows → objelerin gölgeleri zemine düşer, silindir/kutu kenarları
  aydınlık/karanlık geçiş üretir. Bu gradient'ler SLAM'ın depth estimation'ını güçlendirir.
- Güneş açısı (`-0.5 0.3 -1.0`): hafif yatay bileşen → checker tile'larda ince lateral
  gölge bandı → ek high-frequency texture.

---

### 3. Kamera Gaussian Noise

**Öncesi (v3):** Gürültüsüz, ideal kamera.

**Sonrası (v4):**
```xml
<noise>
  <type>gaussian</type>
  <mean>0.0</mean>
  <stddev>0.010</stddev>
</noise>
```

**Etki:**
- stddev=0.010 (1% piksel değeri) — görsel olarak hafif; PSNR ~40 dB civarı
- Gerçek kameralarda sensör gürültüsü bu seviyede → sim2real gap azalır
- SLAM robustness: gürültüye karşı dayanıklılık test edilir; model zaten gerçek
  dünya görselleriyle eğitildiği için hafif noise performansı artırabilir

**SLAM üzerindeki etki (teorik):**
- Pozitif: Gerçekçi gradientler, daha iyi feature matching güveni
- Negatif: Sub-piksel gradient hesaplarında hafif sapma

---

### 4. Materyal Çeşitliliği: Specular + Striped Objeler

**Öncesi (v3):** Tüm objeler sadece `ambient` ve `diffuse` (mat görünüm).

**Sonrası (v4):**
- Tüm objelere `specular` eklendi (değer: 0.15–0.40)
- 8 çift **striped yapı** (16 model): iki farklı renkli kutu üst üste
  → iki renk sınırında keskin yatay kenar → eşsiz feature nokta
- Köşe işaretçileri korundu

**Etki:** Speküler highlight → ışık açısına göre değişen parlaklık → hareket sırasında
dinamik texture değişimi → SLAM'ın optical-flow tahmini için daha ayırt edici işaretler.

---

### 5. Kamera Parametreleri (Değişmedi)

```
Yükseklik : 4 m
FOV       : 1.2217 rad (horizontal)
Çözünürlük: 640 × 480
fx = fy   : 457
calib.txt : 457 457 320 240
```

Kalibrasyon v3 ile aynı kaldığı için `dataset/meta/calib.txt` güncellenmedi.

---

## Hangi Değişiklik En Çok Etki Etti?

Beklenen etki sıralaması (yüksekten düşüğe):

1. **Checkerboard zemin** — En kritik. Düşük-kontrast renkli tile'dan yüksek-kontrast
   siyah-beyaz'a geçiş, birim alanda tespit edilen feature sayısını dramatik artırır.
   Feature density ve repeatability artışı SLAM drift'ini doğrudan azaltır.

2. **Cast shadows + azaltılmış ambient** — İkinci kritik. Flat-lit sahnede SLAM derinlik
   kestirimi zayıftır çünkü kenarlar belirsizleşir. Gölgeler ve lateral aydınlatma,
   3D yapıyı 2D görüntüye yansıtır → daha güvenilir depth estimation.

3. **Striped yapılar** — Orta etki. Sahnenin belirli bölgelerinde benzersiz landmark
   sağlar, loop closure kalitesini artırır.

4. **Specular materyal** — Küçük etki. Hareket sırasında dinamik texture; SLAM sistemleri
   genellikle Lambertian varsayımı yapar, spekülar highlight hafif bozucu olabilir.

5. **Kamera noise** — Belirsiz etki. Hafif noise gerçekçiliği artırır, ancak RMSE
   üzerinde ±0.1 m düzeyinde rastgele etki beklenir.

---

## Dosya Değişiklikleri

| Dosya | Değişiklik |
|-------|-----------|
| `sim/worlds/slam_world_realistic.sdf` | YENİ — realistic v4 world |
| `sim/scripts/gen_realistic_world.py` | YENİ — SDF üretici script |
| `sim/scripts/run_gazebo_realistic.sh` | YENİ — realistic Gazebo başlatıcı |
| `dataset/make_small_motion_v4.py` | YENİ — raw_v4 → small_motion_v4 |
| `slam/scripts/run_droid_figure8_v4.sh` | YENİ — v4 SLAM çalıştırıcı |
| `slam/scripts/evaluate_figure8_v4.py` | YENİ — v4 evaluation + v3 karşılaştırma |
| `dataset/meta/calib.txt` | Değişmedi (457 457 320 240) |
| `sim/worlds/slam_world.sdf` | Değişmedi (v3 korundu) |

---

## v4 Pipeline Çalıştırma

```bash
# Terminal 1 — Realistic Gazebo
bash sim/scripts/run_gazebo_realistic.sh

# Terminal 2 — ROS2 Bridge
bash ros/launch/run_bridge.sh

# Terminal 3 — Dataset kayıt → raw_v4
OUTPUT_DIR=~/code/uav-visual-odometry/dataset/raw_v4 \
  python3 ros/nodes/image_recorder.py

# Terminal 4 — Figure-8 hareketi (~88s, v3 ile aynı parametreler)
PATTERN=figure8 python3 sim/scripts/move_camera.py
# → Bittikten sonra Terminal 3'te Ctrl+C

# Dataset hazırlama
python3 dataset/make_small_motion_v4.py

# DROID-SLAM
bash slam/scripts/run_droid_figure8_v4.sh

# Evaluation + v3 karşılaştırma
python3 slam/scripts/evaluate_figure8_v4.py
```

---

## RMSE Geçmişi (Güncel)

```
Baseline (lawnmower, 10m)    12.98 m  ████████████████████████████
Lawnmower axis scale          5.69 m  ████████████
Figure-8 v2 (10m)             5.52 m  ███████████
Figure-8 v3 (4m, colored)     3.85 m  ████████
Figure-8 v4 (4m, realistic)    TBD    ← hedef: < 3.5 m
```
