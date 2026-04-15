#!/usr/bin/env python3
"""
gen_realistic_world.py  —  slam_world_realistic.sdf üretici

Değişiklikler (v3 → realistic):
  1. Zemin texture:  rengarenk tile'lar → 12×12 yüksek-kontrastlı siyah-beyaz checkerboard
  2. Işıklandırma:   cast_shadows=true, ambient azaltıldı (0.4), güneş açılı
  3. Kamera noise:   Gaussian stddev=0.010 eklendi
  4. Materyal çeşit: tüm objeler specular eklendi, striped pair objeler eklendi
  5. Kamera aynı:    4m yükseklik, FOV=1.2217, 640×480, fx=457

Kullanım:
    python3 ~/code/uav-visual-odometry/sim/scripts/gen_realistic_world.py
"""

import math, os

OUT = os.path.expanduser(
    "~/code/uav-visual-odometry/sim/worlds/slam_world_realistic.sdf"
)

# ─── Yardımcı: SDF model snippet'leri ────────────────────────────────────────

def tile(name, x, y, w, h, r, g, b, specular=0.05):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} 0.03 0 0 0</pose>
      <link name="link">
        <collision name="col"><geometry><box><size>{w:.2f} {h:.2f} 0.06</size></box></geometry></collision>
        <visual name="vis">
          <geometry><box><size>{w:.2f} {h:.2f} 0.06</size></box></geometry>
          <material>
            <ambient>{r:.3f} {g:.3f} {b:.3f} 1</ambient>
            <diffuse>{r:.3f} {g:.3f} {b:.3f} 1</diffuse>
            <specular>{specular:.3f} {specular:.3f} {specular:.3f} 1</specular>
          </material>
        </visual>
      </link>
    </model>"""


def cylinder(name, x, y, z, radius, length, r, g, b, spec=0.2):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.2f} {y:.2f} {z:.2f} 0 0 0</pose>
      <link name="link">
        <collision name="col"><geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry></collision>
        <visual name="vis">
          <geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>
          <material>
            <ambient>{r:.2f} {g:.2f} {b:.2f} 1</ambient>
            <diffuse>{r:.2f} {g:.2f} {b:.2f} 1</diffuse>
            <specular>{spec:.2f} {spec:.2f} {spec:.2f} 1</specular>
          </material>
        </visual>
      </link>
    </model>"""


def box(name, x, y, z, sx, sy, sz, yaw, r, g, b, spec=0.15):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.2f} {y:.2f} {z:.2f} 0 0 {yaw:.2f}</pose>
      <link name="link">
        <collision name="col"><geometry><box><size>{sx:.2f} {sy:.2f} {sz:.2f}</size></box></geometry></collision>
        <visual name="vis">
          <geometry><box><size>{sx:.2f} {sy:.2f} {sz:.2f}</size></box></geometry>
          <material>
            <ambient>{r:.2f} {g:.2f} {b:.2f} 1</ambient>
            <diffuse>{r:.2f} {g:.2f} {b:.2f} 1</diffuse>
            <specular>{spec:.2f} {spec:.2f} {spec:.2f} 1</specular>
          </material>
        </visual>
      </link>
    </model>"""


# ─── 1. Siyah-beyaz checkerboard (12×12, 1m spacing) ─────────────────────────

def gen_checkerboard():
    parts = []
    n = 12          # 12×12 grid
    step = 1.0      # 1m aralık
    tile_size = 0.88  # küçük boşluk → görsel ayrım
    start = -5.5    # -(n-1)/2 * step
    idx = 0
    for ix in range(n):
        for iy in range(n):
            x = start + ix * step
            y = start + iy * step
            white = (ix + iy) % 2 == 0
            if white:
                r, g, b = 0.91, 0.91, 0.91   # parlak beyaz
                spec = 0.08
            else:
                r, g, b = 0.05, 0.05, 0.05   # koyu siyah
                spec = 0.02
            parts.append(tile(f"cb_{idx}", x, y, tile_size, tile_size, r, g, b, spec))
            idx += 1
    return parts, idx


# ─── 2. İnce direkler (12 adet, yüksek kontrast, speküler) ───────────────────

PILLARS = [
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
    return [cylinder(f"pillar_{i}", x, y, z, r, l, cr, cg, cb, sp)
            for i, (x, y, z, r, l, cr, cg, cb, sp) in enumerate(PILLARS)]


# ─── 3. Geniş diskler / gözetleme taşları (4 adet) ───────────────────────────

FAT_CYLS = [
    (-7.0,  0.0, 0.5, 0.7, 0.25,  0.90, 0.10, 0.10, 0.4),
    ( 7.0,  0.0, 0.5, 0.7, 0.25,  0.10, 0.10, 0.90, 0.4),
    ( 0.0, -7.0, 0.5, 0.7, 0.25,  0.95, 0.75, 0.00, 0.3),
    ( 0.0,  7.0, 0.5, 0.7, 0.25,  0.10, 0.80, 0.10, 0.3),
]


def gen_fat_cylinders():
    return [cylinder(f"fat_cyl_{i}", x, y, z, r, l, cr, cg, cb, sp)
            for i, (x, y, z, r, l, cr, cg, cb, sp) in enumerate(FAT_CYLS)]


# ─── 4. Orta boy kutular (12 adet, çeşitli boyut/açı) ────────────────────────

BOXES = [
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
    return [box(f"mbox_{i}", x, y, z, sx, sy, sz, yaw, r, g, b)
            for i, (x, y, z, sx, sy, sz, yaw, r, g, b) in enumerate(BOXES)]


# ─── 5. Şeritli objeler: 2 renkli kutu çiftleri (8 çift = 16 model) ─────────

STRIPE_PAIRS = [
    # ( x,    y, lower_rgb,         upper_rgb,          yaw)
    ( 2.5,  7.5, (0.90,0.10,0.10),  (0.95,0.90,0.00),   0.0),
    (-2.5, -7.5, (0.10,0.10,0.90),  (0.00,0.85,0.85),   0.3),
    ( 7.5, -3.0, (0.00,0.70,0.30),  (0.90,0.50,0.00),   0.5),
    (-7.5,  3.0, (0.80,0.00,0.80),  (0.10,0.90,0.10),   0.8),
    ( 5.8,  2.0, (0.20,0.60,0.90),  (0.90,0.20,0.50),   1.0),
    (-5.8, -2.0, (0.90,0.60,0.00),  (0.10,0.20,0.90),   0.2),
    ( 1.0, -7.5, (0.50,0.90,0.10),  (0.90,0.10,0.80),   0.6),
    (-1.0,  7.5, (0.10,0.10,0.60),  (0.80,0.80,0.10),   1.2),
]


def gen_stripes():
    parts = []
    for i, (x, y, (r1,g1,b1), (r2,g2,b2), yaw) in enumerate(STRIPE_PAIRS):
        # Alt kutu: z=0.25
        parts.append(box(f"stripe_lo_{i}", x, y, 0.25, 0.7, 0.7, 0.5,
                         yaw, r1, g1, b1, 0.3))
        # Üst kutu: z=0.75 (tam üstüne)
        parts.append(box(f"stripe_hi_{i}", x, y, 0.75, 0.7, 0.7, 0.5,
                         yaw, r2, g2, b2, 0.3))
    return parts


# ─── 6. Köşe işaretçiler (4 köşe × 2 parça) ─────────────────────────────────

CORNERS = [
    ("corner_a_0", -7, -7, 0.6, 1.2, 0.4, 1.2, 0.0,  0.9, 0.1, 0.1),
    ("corner_b_0", -6.6,-7.4, 0.3, 0.4, 1.2, 0.6, 0.0,  0.1, 0.1, 0.9),
    ("corner_a_1",  7,  7, 0.6, 1.2, 0.4, 1.2, 0.0,  0.95,0.80,0.0),
    ("corner_b_1",  7.4, 7.4, 0.3, 0.4, 1.2, 0.6, 0.0,  0.0, 0.8, 1.0),
    ("corner_a_2",  7, -7, 0.6, 1.2, 0.4, 1.2, 0.0,  0.1, 0.1, 0.9),
    ("corner_b_2",  7.4,-7.4, 0.3, 0.4, 1.2, 0.6, 0.0,  0.9, 0.1, 0.1),
    ("corner_a_3", -7,  7, 0.6, 1.2, 0.4, 1.2, 0.0,  0.1, 0.8, 0.1),
    ("corner_b_3", -7.4, 7.4, 0.3, 0.4, 1.2, 0.6, 0.0,  0.1, 0.8, 0.1),
]


def gen_corners():
    return [box(n, x, y, z, sx, sy, sz, yaw, r, g, b)
            for (n, x, y, z, sx, sy, sz, yaw, r, g, b) in CORNERS]


# ─── Ana SDF şablonu ─────────────────────────────────────────────────────────

HEADER = """\
<?xml version="1.0" ?>
<!--
  slam_world_realistic.sdf  —  UAV Visual Odometry v4 (Realistic)

  Değişiklikler (v3 → realistic/v4):
    1. Zemin: 12×12 siyah-beyaz checkerboard (yüksek kontrast)
    2. Gölgeler: cast_shadows=true, ambient azaltıldı → derinlik algısı
    3. Kamera: Gaussian noise stddev=0.010 → sim2real gap azaltma
    4. Objeler: specular eklendi, 8 çift striped (2-renkli) yapı
    5. Kamera pos/calib değişmedi: 4m, FOV=1.2217, fx=fy=457
-->
<sdf version="1.9">
  <world name="slam_world_realistic">

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
      Işıklandırma değişiklikleri:
        - ambient: 1.0 → 0.40  (flat aydınlatmayı kırar, shadow görünür)
        - shadows: false → true (derinlik kenarları, SLAM için gradient)
        - Güneş: daha açılı yön → yatay gölgeler
    -->
    <scene>
      <ambient>0.40 0.40 0.40 1</ambient>
      <background>0.45 0.60 0.80 1</background>
      <shadows>true</shadows>
    </scene>

    <!-- Ana güneş: açılı, gölge aktif -->
    <light name="sun" type="directional">
      <pose>0 0 30 0 0 0</pose>
      <diffuse>1.0 0.98 0.92 1</diffuse>
      <specular>0.5 0.5 0.45 1</specular>
      <direction>-0.5 0.3 -1.0</direction>
      <cast_shadows>true</cast_shadows>
    </light>

    <!-- Dolgu ışığı: zayıf, gölgesiz -->
    <light name="fill_light" type="directional">
      <pose>0 0 20 0 0 0</pose>
      <diffuse>0.30 0.32 0.35 1</diffuse>
      <specular>0.05 0.05 0.05 1</specular>
      <direction>0.4 -0.3 -1.0</direction>
      <cast_shadows>false</cast_shadows>
    </light>

    <!-- Zemin düzlemi: koyu gri baz (checkerboard tile'ların altı) -->
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
            <ambient>0.15 0.15 0.15 1</ambient>
            <diffuse>0.18 0.18 0.18 1</diffuse>
            <specular>0.02 0.02 0.02 1</specular>
          </material>
        </visual>
      </link>
    </model>
"""

CAMERA = """
    <!-- ===== KAMERA RIG: 4m, aşağı bakan, FOV=1.2217 rad ===== -->
    <!--
      Değişiklikler (v3 → v4):
        + Gaussian noise stddev=0.010 → sim2real gap, SLAM robustness
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
          <update_rate>15</update_rate>
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
              <stddev>0.010</stddev>
            </noise>
          </camera>
          <topic>/down_camera</topic>
        </sensor>
      </link>
    </model>

  </world>
</sdf>
"""

FOOTER = """
  </world>
</sdf>
"""


def main():
    cb_parts, n_cb = gen_checkerboard()
    pillars  = gen_pillars()
    fatcyls  = gen_fat_cylinders()
    boxes    = gen_boxes()
    stripes  = gen_stripes()
    corners  = gen_corners()

    all_models = cb_parts + pillars + fatcyls + boxes + stripes + corners
    n_total = len(all_models)

    print(f"[gen_realistic] Checkerboard tile  : {n_cb}")
    print(f"[gen_realistic] Pillar             : {len(pillars)}")
    print(f"[gen_realistic] Fat cylinder       : {len(fatcyls)}")
    print(f"[gen_realistic] Medium box         : {len(boxes)}")
    print(f"[gen_realistic] Striped struct ×2  : {len(stripes)} ({len(stripes)//2} çift)")
    print(f"[gen_realistic] Corner marker      : {len(corners)}")
    print(f"[gen_realistic] Toplam model       : {n_total}")

    # Kamera snippet'ini ayrı tut (models içine gömme)
    sdf_body = HEADER
    for m in all_models:
        sdf_body += m + "\n"
    # Kamera son olarak, FOOTER yerine CAMERA kullan
    sdf_body = sdf_body.rstrip()
    # HEADER'ın sonundaki </world>\n</sdf> yok, sadece açık world var
    sdf_body += CAMERA   # CAMERA kendi </world></sdf> ile bitiyor

    out_path = os.path.expanduser(OUT)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(sdf_body)

    print(f"[gen_realistic] Yazıldı: {out_path}")

    # Hızlı XML doğrulama
    import xml.etree.ElementTree as ET
    try:
        ET.parse(out_path)
        print("[gen_realistic] XML geçerli.")
    except ET.ParseError as e:
        print(f"[gen_realistic] HATA: XML parse hatası: {e}")

    print()
    print("[gen_realistic] Sonraki adım:")
    print("  bash ~/code/uav-visual-odometry/sim/scripts/run_gazebo_realistic.sh")


if __name__ == "__main__":
    main()
