#!/usr/bin/env python3
"""
gen_realistic_world_v5.py — slam_world_v5_realistic.sdf üretici

v5 yenilikleri (v4 → v5):
  1. Kamera    : lens distortion (k1=-0.12, k2=0.015)  +  noise 0.018  +  FPS 20
  2. Işık      : 3-light setup (sun + fill + rim), ambient 0.20 → derin gölgeler
  3. Zemin     : v4 checkerboard korunur + seed-bağımlı kir lekeleri (dirt patches)
  4. Parallax  : 5 yüksek kule (1.5–2.5m), yükseklik katmanları 0.3 / 0.7 / 1.2 / 2.0m
  5. Dinamik   : 8 Actor tabanlı hareketli kutu/silindir (loop trajectory)
  6. Randomiz. : --seed ile güneş yönü, gürültü, kir pozisyonu değişir

Kullanım:
    python3 sim/scripts/gen_realistic_world_v5.py            # seed=42
    python3 sim/scripts/gen_realistic_world_v5.py --seed 99
    python3 sim/scripts/gen_realistic_world_v5.py --seed 7 --out /tmp/test_v5.sdf

Neden önemli:
  - Lens distortion: ORB/DROID feature matching'i iyileştirir (gerçek lens davranışı)
  - Düşük ambient + 3 ışık: zengin gölge gradyanı → VO için güçlü kenarlar
  - Dirt patches: uniform yüzey kırar → daha fazla distinguishable keypoint
  - Farklı yükseklik nesneleri: parallax → ölçek hatasını azaltır
  - Dinamik objeler: SLAM outlier rejection'ı test eder
"""

import argparse
import math
import os
import random

REPO = os.path.expanduser("~/code/uav-visual-odometry")
DEFAULT_OUT = os.path.join(REPO, "sim/worlds/slam_world_v5_realistic.sdf")


# ══════════════════════════════════════════════════════════════════════════════
# Primitive builders — (x, y, z) = center, all return SDF string snippet
# ══════════════════════════════════════════════════════════════════════════════

def _mat(r, g, b, spec=0.15, rough=0.6):
    return (
        f"<ambient>{r:.3f} {g:.3f} {b:.3f} 1</ambient>"
        f"<diffuse>{r:.3f} {g:.3f} {b:.3f} 1</diffuse>"
        f"<specular>{spec:.3f} {spec:.3f} {spec:.3f} 1</specular>"
    )


def tile_box(name, x, y, w, h, r, g, b, spec=0.05, z_top=0.03, thick=0.06):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {z_top:.3f} 0 0 0</pose>
      <link name="link">
        <collision name="col"><geometry><box><size>{w:.3f} {h:.3f} {thick:.3f}</size></box></geometry></collision>
        <visual name="vis">
          <geometry><box><size>{w:.3f} {h:.3f} {thick:.3f}</size></box></geometry>
          <material>{_mat(r, g, b, spec)}</material>
        </visual>
      </link>
    </model>"""


def cylinder_m(name, x, y, z, radius, length, r, g, b, spec=0.2):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
      <link name="link">
        <collision name="col"><geometry><cylinder><radius>{radius:.3f}</radius><length>{length:.3f}</length></cylinder></geometry></collision>
        <visual name="vis">
          <geometry><cylinder><radius>{radius:.3f}</radius><length>{length:.3f}</length></cylinder></geometry>
          <material>{_mat(r, g, b, spec)}</material>
        </visual>
      </link>
    </model>"""


def box_m(name, x, y, z, sx, sy, sz, yaw, r, g, b, spec=0.15):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.3f}</pose>
      <link name="link">
        <collision name="col"><geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry></collision>
        <visual name="vis">
          <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
          <material>{_mat(r, g, b, spec)}</material>
        </visual>
      </link>
    </model>"""


# ══════════════════════════════════════════════════════════════════════════════
# 1. Zemin — 12×12 checkerboard + kir lekeleri (v4'ten alındı, dirt eklendi)
# ══════════════════════════════════════════════════════════════════════════════

def gen_checkerboard(rng: random.Random, dirt_ratio: float = 0.08):
    """
    12×12 siyah-beyaz checkerboard.
    dirt_ratio kadarı seed-bağımlı kir lekeleriyle boyanır
    (kahverengi varyasyonlar → feature diversity).
    """
    parts = []
    n = 12
    step = 1.0
    tile_size = 0.88
    start = -5.5
    idx = 0
    for ix in range(n):
        for iy in range(n):
            x = start + ix * step
            y = start + iy * step
            white = (ix + iy) % 2 == 0
            # Kir leke uygula?
            is_dirt = rng.random() < dirt_ratio
            if is_dirt:
                # Toprak tonu: [0.35–0.55, 0.25–0.40, 0.10–0.20]
                r = rng.uniform(0.35, 0.55)
                g = rng.uniform(0.25, 0.40)
                b = rng.uniform(0.10, 0.20)
                spec = 0.02
            elif white:
                r, g, b = 0.91, 0.91, 0.91
                spec = 0.08
            else:
                r, g, b = 0.05, 0.05, 0.05
                spec = 0.02
            parts.append(tile_box(f"cb_{idx}", x, y, tile_size, tile_size,
                                   r, g, b, spec))
            idx += 1
    return parts, idx


# ══════════════════════════════════════════════════════════════════════════════
# 2. Direkler — v4'ten kopyalandı (12 silindir)
# ══════════════════════════════════════════════════════════════════════════════

_PILLARS_BASE = [
    (-6.0,  0.0, 0.7, 0.15, 1.4,  0.20, 0.90, 0.40, 0.35),
    ( 6.0,  0.0, 0.7, 0.15, 1.4,  0.90, 0.10, 0.60, 0.30),
    ( 0.0, -6.0, 0.7, 0.15, 1.4,  0.10, 0.40, 0.90, 0.30),
    ( 0.0,  6.0, 0.7, 0.15, 1.4,  0.90, 0.80, 0.00, 0.25),
    ( 4.0, -5.5, 0.7, 0.15, 1.4,  0.80, 0.10, 0.80, 0.35),
    (-4.0,  5.5, 0.7, 0.15, 1.4,  0.10, 0.80, 0.80, 0.30),
    ( 1.2, -6.0, 0.7, 0.15, 1.4,  0.90, 0.20, 0.20, 0.30),
    (-1.2,  6.0, 0.7, 0.15, 1.4,  0.10, 0.50, 0.90, 0.25),
    ( 6.5,  2.8, 0.7, 0.15, 1.4,  0.80, 0.10, 0.80, 0.30),
    (-6.5, -2.8, 0.7, 0.15, 1.4,  0.10, 0.80, 0.10, 0.25),
    ( 4.5, -6.5, 0.7, 0.15, 1.4,  0.90, 0.80, 0.00, 0.30),
    (-4.5,  6.5, 0.7, 0.15, 1.4,  0.00, 0.80, 0.90, 0.25),
]


def gen_pillars():
    return [cylinder_m(f"pillar_{i}", x, y, z, r, l, cr, cg, cb, sp)
            for i, (x, y, z, r, l, cr, cg, cb, sp) in enumerate(_PILLARS_BASE)]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Geniş silindirler — v4'ten kopyalandı
# ══════════════════════════════════════════════════════════════════════════════

_FAT_CYLS = [
    (-7.0,  0.0, 0.5, 0.7, 0.25,  0.90, 0.10, 0.10, 0.4),
    ( 7.0,  0.0, 0.5, 0.7, 0.25,  0.10, 0.10, 0.90, 0.4),
    ( 0.0, -7.0, 0.5, 0.7, 0.25,  0.95, 0.75, 0.00, 0.3),
    ( 0.0,  7.0, 0.5, 0.7, 0.25,  0.10, 0.80, 0.10, 0.3),
]


def gen_fat_cylinders():
    return [cylinder_m(f"fat_cyl_{i}", x, y, z, r, l, cr, cg, cb, sp)
            for i, (x, y, z, r, l, cr, cg, cb, sp) in enumerate(_FAT_CYLS)]


# ══════════════════════════════════════════════════════════════════════════════
# 4. Orta kutular — v4'ten kopyalandı (12 adet)
# ══════════════════════════════════════════════════════════════════════════════

_BOXES_BASE = [
    ( 6.0,  5.0, 0.50, 1.0, 1.0, 1.0, 0.0,  0.9, 0.10, 0.10),
    (-6.0, -5.0, 0.50, 1.0, 1.0, 1.0, 0.0,  0.1, 0.10, 0.90),
    ( 5.5, -6.0, 0.40, 1.2, 0.6, 0.8, 0.4,  0.0, 0.7,  0.30),
    (-5.5,  6.0, 0.40, 1.2, 0.6, 0.8, 0.6,  0.9, 0.5,  0.00),
    (-6.5,  4.5, 0.35, 0.7, 1.4, 0.7, 0.3,  0.0, 0.6,  0.80),
    ( 6.5, -4.5, 0.35, 0.7, 1.4, 0.7, 0.8,  0.7, 0.1,  0.10),
    (-5.0,  1.5, 0.55, 1.1, 0.5, 1.1, 0.5,  0.5, 0.0,  0.90),
    ( 5.0, -1.5, 0.55, 1.1, 0.5, 1.1, 0.9,  0.1, 0.8,  0.10),
    ( 3.5,  7.0, 0.45, 0.8, 0.8, 0.9, 1.0,  0.1, 0.7,  0.20),
    (-3.5, -7.0, 0.45, 0.8, 0.8, 0.9, 0.2,  0.0, 0.9,  0.70),
    ( 7.2,  5.5, 0.40, 0.6, 1.0, 0.8, 0.7,  0.8, 0.0,  0.80),
    (-7.2, -5.5, 0.40, 0.6, 1.0, 0.8, 1.3,  0.0, 0.7,  0.50),
]


def gen_boxes():
    return [box_m(f"mbox_{i}", x, y, z, sx, sy, sz, yaw, r, g, b)
            for i, (x, y, z, sx, sy, sz, yaw, r, g, b) in enumerate(_BOXES_BASE)]


# ══════════════════════════════════════════════════════════════════════════════
# 5. Şeritli çiftler — v4'ten kopyalandı
# ══════════════════════════════════════════════════════════════════════════════

_STRIPE_PAIRS = [
    ( 2.5,  7.5, (0.90,0.10,0.10), (0.95,0.90,0.00), 0.0),
    (-2.5, -7.5, (0.10,0.10,0.90), (0.00,0.85,0.85), 0.3),
    ( 7.5, -3.0, (0.00,0.70,0.30), (0.90,0.50,0.00), 0.5),
    (-7.5,  3.0, (0.80,0.00,0.80), (0.10,0.90,0.10), 0.8),
    ( 5.8,  2.0, (0.20,0.60,0.90), (0.90,0.20,0.50), 1.0),
    (-5.8, -2.0, (0.90,0.60,0.00), (0.10,0.20,0.90), 0.2),
    ( 1.0, -7.5, (0.50,0.90,0.10), (0.90,0.10,0.80), 0.6),
    (-1.0,  7.5, (0.10,0.10,0.60), (0.80,0.80,0.10), 1.2),
]


def gen_stripes():
    parts = []
    for i, (x, y, (r1,g1,b1), (r2,g2,b2), yaw) in enumerate(_STRIPE_PAIRS):
        parts.append(box_m(f"stripe_lo_{i}", x, y, 0.25, 0.7, 0.7, 0.5,
                           yaw, r1, g1, b1, 0.3))
        parts.append(box_m(f"stripe_hi_{i}", x, y, 0.75, 0.7, 0.7, 0.5,
                           yaw, r2, g2, b2, 0.3))
    return parts


# ══════════════════════════════════════════════════════════════════════════════
# 6. Köşe işaretçiler — v4'ten kopyalandı
# ══════════════════════════════════════════════════════════════════════════════

_CORNERS = [
    ("corner_a_0", -7.0, -7.0, 0.6, 1.2, 0.4, 1.2, 0.0,  0.9, 0.1, 0.1),
    ("corner_b_0", -6.6, -7.4, 0.3, 0.4, 1.2, 0.6, 0.0,  0.1, 0.1, 0.9),
    ("corner_a_1",  7.0,  7.0, 0.6, 1.2, 0.4, 1.2, 0.0,  0.95,0.80,0.0),
    ("corner_b_1",  7.4,  7.4, 0.3, 0.4, 1.2, 0.6, 0.0,  0.0, 0.8, 1.0),
    ("corner_a_2",  7.0, -7.0, 0.6, 1.2, 0.4, 1.2, 0.0,  0.1, 0.1, 0.9),
    ("corner_b_2",  7.4, -7.4, 0.3, 0.4, 1.2, 0.6, 0.0,  0.9, 0.1, 0.1),
    ("corner_a_3", -7.0,  7.0, 0.6, 1.2, 0.4, 1.2, 0.0,  0.1, 0.8, 0.1),
    ("corner_b_3", -7.4,  7.4, 0.3, 0.4, 1.2, 0.6, 0.0,  0.1, 0.8, 0.1),
]


def gen_corners():
    return [box_m(n, x, y, z, sx, sy, sz, yaw, r, g, b)
            for (n, x, y, z, sx, sy, sz, yaw, r, g, b) in _CORNERS]


# ══════════════════════════════════════════════════════════════════════════════
# 7. YENİ v5: Yüksek kuleler — parallax artırma (5 kule, 1.5–2.5m)
# ══════════════════════════════════════════════════════════════════════════════

def gen_tall_towers(rng: random.Random):
    """
    5 yüksek kule — farklı yüksekliklerde, farklı renklerde.
    Parallax zenginleştirir: kamera hareket ederken farklı depth katmanları
    görünür → ölçek hatası azalır.
    """
    configs = [
        # (x, y, height, radius, r, g, b)
        ( 3.0,  3.0, 2.5, 0.18, 0.85, 0.20, 0.10),
        (-3.0, -3.0, 2.0, 0.18, 0.10, 0.20, 0.85),
        ( 3.0, -3.0, 1.8, 0.20, 0.20, 0.75, 0.20),
        (-3.0,  3.0, 1.5, 0.20, 0.85, 0.70, 0.00),
        ( 0.0,  0.0, 2.2, 0.15, 0.70, 0.10, 0.85),
    ]
    # Seed-bağımlı hafif pozisyon jitter
    parts = []
    for i, (bx, by, h, rad, r, g, b) in enumerate(configs):
        jx = bx + rng.uniform(-0.3, 0.3)
        jy = by + rng.uniform(-0.3, 0.3)
        z_center = h / 2.0
        parts.append(cylinder_m(f"tower_{i}", jx, jy, z_center, rad, h,
                                 r, g, b, spec=0.4))
    return parts


# ══════════════════════════════════════════════════════════════════════════════
# 8. YENİ v5: Çok katlı platform grupları — yükseklik katmanları 0.3 / 0.7 / 1.2m
# ══════════════════════════════════════════════════════════════════════════════

def gen_height_layers():
    """
    3 yükseklik seviyesinde platform kümeleri.
    Kamera yüksekliği 3–5m → 0.3m / 0.7m / 1.2m farklı görüntü büyüklükleri
    → parallax gradyanı sağlar.
    """
    platforms = [
        # Level 0.3m: 4 düz platform (geniş, matte)
        ("plat_low_0",  5.0, -5.0, 0.15, 1.5, 1.5, 0.3, 0.0, 0.60, 0.55, 0.50, 0.03),
        ("plat_low_1", -5.0,  5.0, 0.15, 1.5, 1.5, 0.3, 0.0, 0.50, 0.55, 0.60, 0.03),
        ("plat_low_2",  5.5,  3.0, 0.15, 1.2, 1.0, 0.3, 0.3, 0.55, 0.50, 0.45, 0.04),
        ("plat_low_3", -5.5, -3.0, 0.15, 1.2, 1.0, 0.3, 0.6, 0.45, 0.50, 0.55, 0.04),
        # Level 0.7m: 3 yükseltilmiş platform (orta parlaklık)
        ("plat_mid_0",  0.0,  5.5, 0.35, 1.0, 1.8, 0.7, 0.0, 0.80, 0.30, 0.10, 0.15),
        ("plat_mid_1",  0.0, -5.5, 0.35, 1.0, 1.8, 0.7, 0.0, 0.10, 0.30, 0.80, 0.15),
        ("plat_mid_2",  5.0,  0.0, 0.35, 1.8, 1.0, 0.7, 0.0, 0.10, 0.80, 0.30, 0.15),
        # Level 1.2m: 3 yüksek platform (parlak, specular)
        ("plat_hi_0", -5.0,  0.0, 0.60, 1.8, 1.0, 1.2, 0.0, 0.90, 0.80, 0.00, 0.30),
        ("plat_hi_1",  3.0,  6.0, 0.60, 1.2, 1.2, 1.2, 0.5, 0.00, 0.70, 0.90, 0.25),
        ("plat_hi_2", -3.0, -6.0, 0.60, 1.2, 1.2, 1.2, 0.9, 0.80, 0.00, 0.80, 0.25),
    ]
    return [box_m(n, x, y, z, sx, sy, sz, yaw, r, g, b, spec)
            for (n, x, y, z, sx, sy, sz, yaw, r, g, b, spec) in platforms]


# ══════════════════════════════════════════════════════════════════════════════
# 9. YENİ v5: Dinamik Actor objeler (8 hareketli kutu/silindir)
# ══════════════════════════════════════════════════════════════════════════════

def _actor_box(name, x1, y1, x2, y2, z, period, r, g, b):
    """
    İleri-geri linear hareket yapan Actor kutusu.
    period saniyede bir tam tur yapar (x1,y1) → (x2,y2) → (x1,y1).
    """
    half = period / 2.0
    return f"""
    <actor name="{name}">
      <pose>{x1:.2f} {y1:.2f} {z:.2f} 0 0 0</pose>
      <link name="body">
        <visual name="vis">
          <geometry><box><size>0.40 0.40 0.40</size></box></geometry>
          <material>{_mat(r, g, b, spec=0.20)}</material>
        </visual>
      </link>
      <script>
        <loop>true</loop>
        <delay_start>0</delay_start>
        <auto_start>true</auto_start>
        <trajectory id="0" type="__default__">
          <waypoint><time>0</time><pose>{x1:.2f} {y1:.2f} {z:.2f} 0 0 0</pose></waypoint>
          <waypoint><time>{half:.1f}</time><pose>{x2:.2f} {y2:.2f} {z:.2f} 0 0 0</pose></waypoint>
          <waypoint><time>{period:.1f}</time><pose>{x1:.2f} {y1:.2f} {z:.2f} 0 0 0</pose></waypoint>
        </trajectory>
      </script>
    </actor>"""


def _actor_cyl(name, cx, cy, radius_traj, z, period, r, g, b):
    """
    Dairesel yörüngede dönen Actor silindiri.
    8 waypoint ile yaklaşık daire.
    """
    n_wp = 8
    lines = [
        f"""
    <actor name="{name}">
      <pose>{cx:.2f} {cy:.2f} {z:.2f} 0 0 0</pose>
      <link name="body">
        <visual name="vis">
          <geometry><cylinder><radius>0.20</radius><length>0.35</length></cylinder></geometry>
          <material>{_mat(r, g, b, spec=0.25)}</material>
        </visual>
      </link>
      <script>
        <loop>true</loop>
        <delay_start>0</delay_start>
        <auto_start>true</auto_start>
        <trajectory id="0" type="__default__">"""
    ]
    for k in range(n_wp + 1):
        t = period * k / n_wp
        angle = 2 * math.pi * k / n_wp
        wx = cx + radius_traj * math.cos(angle)
        wy = cy + radius_traj * math.sin(angle)
        lines.append(
            f"          <waypoint><time>{t:.2f}</time>"
            f"<pose>{wx:.3f} {wy:.3f} {z:.2f} 0 0 0</pose></waypoint>"
        )
    lines.append("""        </trajectory>
      </script>
    </actor>""")
    return "\n".join(lines)


def gen_dynamic_actors(rng: random.Random):
    """
    8 hareketli nesne: 5 linear kutu + 3 dairesel silindir.
    Seed-bağımlı başlangıç faz/hız küçük varyasyonlar içerir.
    """
    parts = []

    # 5 linear box — farklı renk, farklı hız (period 4–10s)
    linear_defs = [
        # (x1,  y1,  x2,  y2,  z,    period, r,   g,   b)
        (-4.0,  2.0,  4.0,  2.0, 0.25,  8.0, 0.90, 0.10, 0.10),
        ( 4.0, -2.0, -4.0, -2.0, 0.25,  6.0, 0.10, 0.10, 0.90),
        (-3.0, -4.0,  3.0, -4.0, 0.25,  7.0, 0.10, 0.80, 0.10),
        ( 2.0,  5.0, -2.0,  5.0, 0.25,  9.0, 0.90, 0.80, 0.00),
        (-2.0, -5.0,  2.0, -5.0, 0.25,  5.0, 0.80, 0.10, 0.80),
    ]
    for i, (x1, y1, x2, y2, z, period, r, g, b) in enumerate(linear_defs):
        # Hafif seed-bağımlı jitter
        jitter = rng.uniform(-0.4, 0.4)
        parts.append(_actor_box(f"dyn_box_{i}",
                                x1 + jitter, y1, x2 + jitter, y2,
                                z, period + rng.uniform(-1, 1), r, g, b))

    # 3 circular cylinder
    circle_defs = [
        # (cx,   cy,  r_traj, z,    period, r,   g,   b)
        ( 0.0,  3.0,   2.5,  0.25,  10.0, 0.95, 0.50, 0.00),
        ( 0.0, -3.0,   2.5,  0.25,   8.0, 0.00, 0.80, 0.80),
        ( 0.0,  0.0,   3.5,  0.25,  12.0, 0.70, 0.00, 0.70),
    ]
    for i, (cx, cy, r_traj, z, period, r, g, b) in enumerate(circle_defs):
        parts.append(_actor_cyl(f"dyn_cyl_{i}", cx, cy, r_traj, z,
                                period + rng.uniform(-1.5, 1.5), r, g, b))

    return parts


# ══════════════════════════════════════════════════════════════════════════════
# 10. SDF Header — ışık + sahne (seed-bağımlı güneş yönü)
# ══════════════════════════════════════════════════════════════════════════════

def build_header(seed: int, noise_stddev: float, sun_dir: tuple,
                 ambient: float) -> str:
    dx, dy, dz = sun_dir
    return f"""\
<?xml version="1.0" ?>
<!--
  slam_world_v5_realistic.sdf  —  UAV Visual Odometry v5 (Realistic)
  Üretildi: gen_realistic_world_v5.py  seed={seed}

  v5 yenilikleri (v4 → v5):
    1. Kamera: lens distortion (k1=-0.12, k2=0.015) + noise={noise_stddev:.3f} + FPS 20
    2. Işık:   3-light setup (sun+fill+rim), ambient={ambient:.2f} → derin gölge
    3. Zemin:  seed-bağımlı kir lekeleri (dirt patches) feature diversity için
    4. Parallax: 5 yüksek kule (1.5–2.5m) + 10 çok katlı platform
    5. Dinamik: 8 Actor (5 linear kutu + 3 dairesel silindir)
    6. Randomiz: seed={seed} → güneş yönü, gürültü, kir konumları
-->
<sdf version="1.9">
  <world name="slam_world_v5_realistic">

    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system"
            name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <gravity>0 0 -9.8</gravity>

    <!--
      v5 Işıklandırma değişiklikleri:
        - ambient: 0.40 → {ambient:.2f}  (daha derin gölgeler, feature gradyanı)
        - 3-light setup: main sun + fill + rim backlight
        - Güneş yönü seed={seed}'den: ({dx:.2f}, {dy:.2f}, {dz:.2f})
    -->
    <scene>
      <ambient>{ambient:.2f} {ambient:.2f} {ambient:.2f} 1</ambient>
      <background>0.45 0.60 0.80 1</background>
      <shadows>true</shadows>
    </scene>

    <!-- Ana güneş: açılı, gölge aktif, yön seed-bağımlı -->
    <light name="sun" type="directional">
      <pose>0 0 30 0 0 0</pose>
      <diffuse>1.0 0.97 0.90 1</diffuse>
      <specular>0.55 0.55 0.50 1</specular>
      <direction>{dx:.3f} {dy:.3f} {dz:.3f}</direction>
      <cast_shadows>true</cast_shadows>
    </light>

    <!-- Dolgu ışığı: zayıf, gölgesiz, karşı taraftan -->
    <light name="fill_light" type="directional">
      <pose>0 0 20 0 0 0</pose>
      <diffuse>0.25 0.27 0.30 1</diffuse>
      <specular>0.04 0.04 0.04 1</specular>
      <direction>{-dx:.3f} {-dy:.3f} {dz:.3f}</direction>
      <cast_shadows>false</cast_shadows>
    </light>

    <!-- Kenar ışığı: rim backlight, kontrast artırır -->
    <light name="rim_light" type="directional">
      <pose>0 0 15 0 0 0</pose>
      <diffuse>0.15 0.18 0.22 1</diffuse>
      <specular>0.10 0.10 0.12 1</specular>
      <direction>{dy:.3f} {-dx:.3f} {dz*0.5:.3f}</direction>
      <cast_shadows>false</cast_shadows>
    </light>

    <!-- Zemin düzlemi: koyu gri baz -->
    <model name="ground_plane">
      <static>true</static>
      <pose>0 0 0 0 0 0</pose>
      <link name="ground_link">
        <collision name="collision">
          <geometry>
            <plane><normal>0 0 1</normal><size>80 80</size></plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane><normal>0 0 1</normal><size>80 80</size></plane>
          </geometry>
          <material>
            <ambient>0.12 0.12 0.12 1</ambient>
            <diffuse>0.15 0.15 0.15 1</diffuse>
            <specular>0.02 0.02 0.02 1</specular>
          </material>
        </visual>
      </link>
    </model>
"""


# ══════════════════════════════════════════════════════════════════════════════
# 11. Kamera rig — v5: lens distortion + noise 0.018 + FPS 20
# ══════════════════════════════════════════════════════════════════════════════

def build_camera(noise_stddev: float) -> str:
    return f"""
    <!-- ===== KAMERA RIG v5: 4m, aşağı bakan, FOV=1.2217 rad ===== -->
    <!--
      v5 değişiklikleri:
        + Lens distortion: k1=-0.12, k2=0.015 (barrel, gerçek geniş açı lens)
        + Gaussian noise: stddev={noise_stddev:.3f} (v4: 0.010 → sim2real gap)
        + FPS: 15 → 20 (daha sık frame → daha iyi SLAM tracking)
      Sabit:
        4m yükseklik, FOV=1.2217, 640×480, fx=fy=457
    -->
    <model name="down_cam_rig">
      <static>true</static>
      <pose>0 0 4 0 0 0</pose>

      <link name="cam_link">
        <pose>0 0 0 0 1.5708 0</pose>

        <sensor name="down_camera" type="camera">
          <always_on>true</always_on>
          <update_rate>20</update_rate>
          <visualize>false</visualize>
          <camera>
            <horizontal_fov>1.2217</horizontal_fov>
            <image>
              <width>640</width>
              <height>480</height>
              <format>R8G8B8</format>
            </image>
            <clip>
              <near>0.05</near>
              <far>50</far>
            </clip>
            <noise>
              <type>gaussian</type>
              <mean>0.0</mean>
              <stddev>{noise_stddev:.4f}</stddev>
            </noise>
            <distortion>
              <k1>-0.120</k1>
              <k2>0.015</k2>
              <k3>0.000</k3>
              <p1>0.000</p1>
              <p2>0.000</p2>
              <center>0.5 0.5</center>
            </distortion>
          </camera>
          <topic>/down_camera</topic>
        </sensor>
      </link>
    </model>

  </world>
</sdf>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Ana üretici
# ══════════════════════════════════════════════════════════════════════════════

def generate(seed: int, out_path: str) -> None:
    rng = random.Random(seed)

    # Domain randomization: güneş yönü
    sun_azimuth  = rng.uniform(-0.8, 0.8)   # x bileşeni
    sun_lateral  = rng.uniform(0.1, 0.5)    # y bileşeni
    sun_dir      = (sun_azimuth, sun_lateral, -1.0)

    # Noise: 0.015–0.022 arası
    noise_stddev = rng.uniform(0.015, 0.022)

    # Ambient: 0.18–0.25 arası
    ambient = rng.uniform(0.18, 0.25)

    # ── Tüm parçaları üret ────────────────────────────────────────────────────
    cb_parts, n_cb   = gen_checkerboard(rng, dirt_ratio=0.08)
    pillars           = gen_pillars()
    fat_cyls          = gen_fat_cylinders()
    boxes             = gen_boxes()
    stripes           = gen_stripes()
    corners           = gen_corners()
    towers            = gen_tall_towers(rng)         # YENİ
    height_layers     = gen_height_layers()           # YENİ
    dynamic_actors    = gen_dynamic_actors(rng)       # YENİ

    static_models = (cb_parts + pillars + fat_cyls + boxes +
                     stripes + corners + towers + height_layers)

    # ── SDF birleştir ─────────────────────────────────────────────────────────
    sdf = build_header(seed, noise_stddev, sun_dir, ambient)

    for m in static_models:
        sdf += m + "\n"

    for actor in dynamic_actors:
        sdf += actor + "\n"

    sdf += build_camera(noise_stddev)

    # ── Yaz ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(sdf)

    # ── Özet ─────────────────────────────────────────────────────────────────
    n_total = len(static_models) + len(dynamic_actors)
    print(f"[gen_v5] seed={seed}")
    print(f"[gen_v5] Checkerboard tiles  : {n_cb}")
    print(f"[gen_v5] Statik modeller     : {len(static_models)}")
    print(f"[gen_v5]   ↳ pillars         : {len(pillars)}")
    print(f"[gen_v5]   ↳ fat cylinders   : {len(fat_cyls)}")
    print(f"[gen_v5]   ↳ boxes           : {len(boxes)}")
    print(f"[gen_v5]   ↳ stripe pairs    : {len(stripes)//2} çift ({len(stripes)})")
    print(f"[gen_v5]   ↳ corners         : {len(corners)}")
    print(f"[gen_v5]   ↳ tall towers (Ⓝ): {len(towers)}")
    print(f"[gen_v5]   ↳ height layers (Ⓝ): {len(height_layers)}")
    print(f"[gen_v5] Actor (dinamik)  (Ⓝ): {len(dynamic_actors)}")
    print(f"[gen_v5] Toplam nesne        : {n_total}")
    print(f"[gen_v5] Noise stddev        : {noise_stddev:.4f}")
    print(f"[gen_v5] Ambient             : {ambient:.3f}")
    print(f"[gen_v5] Güneş yönü         : ({sun_dir[0]:.3f}, {sun_dir[1]:.3f}, {sun_dir[2]:.3f})")
    print(f"[gen_v5] Yazıldı             : {out_path}")

    import xml.etree.ElementTree as ET
    try:
        ET.parse(out_path)
        print("[gen_v5] XML geçerli.")
    except ET.ParseError as e:
        print(f"[gen_v5] HATA: XML parse hatası: {e}")

    print()
    print("[gen_v5] Çalıştırmak için:")
    print(f"  WORLD_NAME=slam_world_v5_realistic bash sim/scripts/run_gazebo_realistic.sh")


def main():
    parser = argparse.ArgumentParser(description="v5 realistic world üretici")
    parser.add_argument("--seed", type=int, default=42,
                        help="Domain randomization seed (güneş yönü, noise, kir)")
    parser.add_argument("--out",  default=DEFAULT_OUT,
                        help="Çıktı SDF yolu")
    args = parser.parse_args()
    generate(args.seed, os.path.expanduser(args.out))


if __name__ == "__main__":
    main()
