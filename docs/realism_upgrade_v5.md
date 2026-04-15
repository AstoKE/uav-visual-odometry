# Realism Upgrade v5 — Gazebo World Geliştirmeleri

## Genel Bakış

Hedef: Gazebo sim2real gap'ini azaltarak DROID-SLAM'in feature extraction kalitesini ve Sim(3) kalibrasyon kararlılığını iyileştirmek.

**Mevcut durum (v4):** RMSE_2D ≈ 3.85 m  
**Hedef (v5):** RMSE_2D < 3.0 m

---

## v4 → v5 Değişiklik Özeti

| Bileşen | v4 | v5 | Beklenen Etki |
|---|---|---|---|
| Kamera noise stddev | 0.010 | 0.015–0.022 (seed) | Daha gerçekçi görüntü bozulması |
| Lens distortion | yok | k1=-0.12, k2=0.015 | Gerçek geniş açı lens davranışı |
| FPS | 15 | 20 | Daha sık frame → daha iyi tracking |
| Ambient ışık | 0.40 | 0.18–0.25 (seed) | Derin gölgeler → feature kenarları |
| Işık sayısı | 2 (sun+fill) | 3 (sun+fill+rim) | Gölge çeşitliliği → zengin gradient |
| Zemin kir lekeleri | yok | %8 tile (seed) | Feature diversity artışı |
| Parallax katmanları | ~3 seviye | 5 seviye (0.3–2.5m) | Ölçek hatası azalır |
| Yüksek kuleler | yok | 5 × 1.5–2.5m | Derinlik kenarları, parallax boost |
| Dinamik objeler | yok | 8 Actor | SLAM outlier rejection testi |
| Domain randomization | yok | seed tabanlı | Overfitting'e karşı dayanıklılık |

---

## 1. Kamera Modeli Geliştirmeleri

### Lens Distortion
```xml
<distortion>
  <k1>-0.120</k1>   <!-- barrel distortion, negatif → dışa şişme -->
  <k2>0.015</k2>    <!-- 2. dereceden düzeltme -->
  <center>0.5 0.5</center>
</distortion>
```

**Neden önemli:** Gerçek UAV kameralarında radyal distorsiyon kaçınılmazdır. SLAM feature matching'i lens distortion ile eğitimde hesaba katılmamışsa, ground truth korelasyonu bozulur. k1=-0.12 yaygın bir geniş açı değeridir.

### Noise Artışı (0.010 → 0.015–0.022)
Gazebo'nun Gaussian noise modeli piksel değerlerine eklenir. Daha yüksek stddev, SLAM'ı zayıf texture bölgelerinde daha dikkatli olmaya zorlar (gerçek kameralar tipik olarak 0.015–0.025 arası sensor noise içerir).

### FPS: 15 → 20
Daha kısa inter-frame süre → kamera hareketi küçülür → optik akış daha güvenilir → daha fazla başarılı SLAM track.

---

## 2. Işıklandırma Değişiklikleri

### Üç Işık Kurulumu
```
Sun (ana)    : cast_shadows=true, diffuse=(1.0, 0.97, 0.90)
Fill (dolgu) : cast_shadows=false, diffuse=(0.25, 0.27, 0.30)  ← karşı taraf
Rim (kenar)  : cast_shadows=false, diffuse=(0.15, 0.18, 0.22)  ← backlight
```

**Neden önemli:** Tek yönlü ışıkla düz görüntüler oluşur; SLAM keypoint detector'lar (FAST, Shi-Tomasi, ORB) **kenar gradyanlarına** bağımlıdır. Üç ışık + düşük ambient → çapraz gölgeler → zengin texture gradient → daha fazla inlier.

### Ambient: 0.40 → 0.18–0.25
Ambient düşürüldüğünde gölge bölgeleri gerçekten karanlık kalır. Bu, kamera aşağı bakarken oluşan gölge kenarlarını (nesnelerin yanları) daha belirgin yapar.

### Seed-bağımlı Güneş Yönü
```python
sun_azimuth = rng.uniform(-0.8, 0.8)   # x
sun_lateral = rng.uniform(0.1, 0.5)    # y
```
Her çalıştırmada farklı gölge paterni → SLAM modeli belirli bir ışık koşuluna fazla uyum sağlamaz.

---

## 3. Zemin — Kir Lekeleri (Dirt Patches)

v4'teki saf siyah-beyaz checkerboard, SLAM için mükemmel ama gerçek değil. v5'te %8 oranında tile kahverengi tonlarla boyanır:

```python
r = rng.uniform(0.35, 0.55)  # toprak tonu
g = rng.uniform(0.25, 0.40)
b = rng.uniform(0.10, 0.20)
```

**Beklenen etki:** Uniform grid paterni kırılır → farklı bölgelerde farklı feature density → görsel yerini belirleme (place recognition) iyileşir.

---

## 4. Parallax / Yükseklik Çeşitliliği

### 5 Yüksek Kule (1.5–2.5m)
```
tower_0: (3.0,  3.0) h=2.5m  kırmızı
tower_1: (-3.0,-3.0) h=2.0m  mavi
tower_2: (3.0, -3.0) h=1.8m  yeşil
tower_3: (-3.0, 3.0) h=1.5m  sarı
tower_4: (0.0,  0.0) h=2.2m  mor  ← merkez
```

### 10 Çok Katlı Platform (3 seviye)
```
Low  (0.3m): 4 platform — matte tekstür
Mid  (0.7m): 3 platform — orta parlaklık
High (1.2m): 3 platform — specular
```

**Neden önemli:** Kamera 3–5m yükseklikte uçarken, yüksek nesneler (2.5m kule) ile alçak nesneler (0.3m platform) görüntüde farklı perspektif değişimleri gösterir. Bu **parallax farkı**, SLAM'ın derinlik tahmini yapmasını sağlar → ölçek kalibrasyonu iyileşir.

---

## 5. Dinamik Objeler — 8 Actor

### 5 Linear Kutu (ileri-geri)
```
dyn_box_0: (-4,2) ↔ (4,2)  period=8s   kırmızı
dyn_box_1: (4,-2) ↔ (-4,-2) period=6s  mavi
...
```

### 3 Dairesel Silindir
```
dyn_cyl_0: merkez=(0,3)  r_traj=2.5m  period=10s  turuncu
dyn_cyl_1: merkez=(0,-3) r_traj=2.5m  period=8s   camgöbeği
dyn_cyl_2: merkez=(0,0)  r_traj=3.5m  period=12s  mor
```

**Neden önemli:** SLAM algoritmalarının temel varsayımı "statik dünya"dır. Hareketli objeler RANSAC'ın outlier olarak ayırt etmesi gereken false match'ler üretir. Bu test, estimator'ın health=0 dönemlerinde drift rejection'ının ne kadar sağlam olduğunu ölçer.

**Beklenen etki:** İlk çalıştırmalarda RMSE hafif artabilir (SLAM hareketlileri filtrelerken). Doğru kurulumda (enough static features) SLAM bunları yok saymalı.

---

## 6. Domain Randomization

```bash
# Her seed farklı bir dünya üretir
python3 sim/scripts/gen_realistic_world_v5.py --seed 42  # varsayılan
python3 sim/scripts/gen_realistic_world_v5.py --seed 99  # farklı güneş yönü
python3 sim/scripts/gen_realistic_world_v5.py --seed 7   # yüksek noise
```

| Parametre | Aralık | Kontrol eden |
|---|---|---|
| Güneş yönü x | -0.8 – +0.8 | Gölge yönü |
| Güneş yönü y | 0.1 – 0.5 | Lateral gölge |
| Noise stddev | 0.015 – 0.022 | Sensor bozulması |
| Ambient | 0.18 – 0.25 | Genel parlaklık |
| Kule jitter | ±0.3m | Parallax yoğunluğu |

---

## Kullanım

### World üret ve Gazebo başlat
```bash
python3 sim/scripts/gen_realistic_world_v5.py --seed 42
WORLD_NAME=slam_world_v5_realistic bash sim/scripts/run_gazebo_realistic.sh
```

### v4 vs v5 benchmark çalıştır
```bash
bash slam/scripts/run_realism_benchmark.sh
# veya var olan datasetlerle sadece karşılaştır:
bash slam/scripts/run_realism_benchmark.sh --skip-collect
```

### Full pipeline v5 ile
```bash
bash runtime/run_full_pipeline.sh --world slam_world_v5_realistic
```

---

## Beklenen Performans Değişimi

| Metrik | v4 Beklenti | v5 Beklenti | Değişim |
|---|---|---|---|
| RMSE_2D | ~3.85m | ~2.8–3.2m | ↓ %15–27 |
| Scale variance | yüksek | orta | ↓ |
| Drift (health=0) | yüksek | orta | ↓ |
| Recovery | orta | daha iyi | ↑ |

> **Not:** Dinamik objeler ilk çalıştırmada RMSE'yi geçici olarak artırabilir. Eğer SLAM statik feature sayısı yeterli değilse `--seed` değiştirilmeli (daha az dinamik bölgeye yönlendirir).

---

## Dosya Listesi

| Dosya | Açıklama |
|---|---|
| `sim/worlds/slam_world_v5_realistic.sdf` | Üretilen world (seed=42) |
| `sim/scripts/gen_realistic_world_v5.py` | World üretici (tüm parametreler) |
| `slam/scripts/run_realism_benchmark.sh` | v4 vs v5 karşılaştırma pipeline |
| `evaluation/metrics/realism_comparison.txt` | Benchmark sonucu |

**Mevcut v4 dosyaları silinmedi:**
- `sim/worlds/slam_world_realistic.sdf` → v4 world (korundu)
- `sim/scripts/gen_realistic_world.py` → v4 üretici (korundu)
