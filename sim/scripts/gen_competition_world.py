#!/usr/bin/env python3
"""
gen_competition_world.py — sim/worlds/competition_world.sdf üretici

Yarışma senaryosu dünyası:
  - 20×20m alan, 4 farklı texture zonu (asfalt/çim/beton/karışık)
  - Şehir benzeri ızgara → farklı bölgelerde farklı görsel çeşitlilik
  - Tekrar eden ama birebir aynı olmayan pattern
  - Unique landmark bloklar (belirli bölgelerde)
  - Kamera: aşağı bakıyor (pitch ≈ -80°), yükseklik 4m (3–6 arası değişebilir)
  - Gaussian noise stddev=0.015
  - cast_shadows=true, iki ışık kaynağı (güneş + fill light → exposure variation)
  - 640×480, fx=fy=457

Kullanım:
    python3 sim/scripts/gen_competition_world.py [--height 4.0] [--noise 0.015]
"""

import math, os, argparse

REPO = os.path.expanduser("~/code/uav-visual-odometry")
OUT  = os.path.join(REPO, "sim/worlds/competition_world.sdf")

# ── SDF snippet yardımcıları ──────────────────────────────────────────────────

def _mat(r, g, b, spec=0.08, emit=0.0):
    e = f"<emissive>{emit:.3f} {emit:.3f} {emit:.3f} 1</emissive>" if emit > 0 else ""
    return (f"<ambient>{r:.3f} {g:.3f} {b:.3f} 1</ambient>"
            f"<diffuse>{r:.3f} {g:.3f} {b:.3f} 1</diffuse>"
            f"<specular>{spec:.3f} {spec:.3f} {spec:.3f} 1</specular>{e}")


def floor_tile(name, x, y, w, h, r, g, b, spec=0.06):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} 0.025 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>{w:.2f} {h:.2f} 0.05</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>{w:.2f} {h:.2f} 0.05</size></box></geometry>
          <material>{_mat(r, g, b, spec)}</material></visual>
      </link>
    </model>"""


def landmark_box(name, x, y, z, sx, sy, sz, yaw, r, g, b):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.2f} {y:.2f} {z:.2f} 0 0 {yaw:.3f}</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>{sx:.2f} {sy:.2f} {sz:.2f}</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>{sx:.2f} {sy:.2f} {sz:.2f}</size></box></geometry>
          <material>{_mat(r, g, b, spec=0.3)}</material></visual>
      </link>
    </model>"""


def landmark_cyl(name, x, y, z, radius, length, r, g, b):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.2f} {y:.2f} {z:.2f} 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry></collision>
        <visual name="v"><geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>
          <material>{_mat(r, g, b, spec=0.4)}</material></visual>
      </link>
    </model>"""


# ── Texture zonu paleti ───────────────────────────────────────────────────────

# Her kare için renk: (r, g, b) + küçük varyasyon
def asphalt(row, col, vary):
    """Koyu gri asfalt: hafif ızgara çizgisi efekti."""
    base = 0.22
    # Izgara çizgileri: 4'e bölen satır/sütunlarda açık çizgi
    on_grid = (row % 4 == 0) or (col % 4 == 0)
    v = 0.04 if on_grid else vary * 0.05
    return base + v, base + v, base + v + 0.02


def grass(row, col, vary):
    """Yeşil çim: yama efekti."""
    r = 0.15 + vary * 0.08
    g = 0.45 + vary * 0.12
    b = 0.10 + vary * 0.05
    return r, g, b


def concrete(row, col, vary):
    """Açık gri beton: düzgün ama hafif doku."""
    base = 0.68 + vary * 0.06
    return base, base, base - 0.03


def mixed(row, col, vary):
    """Karışık: 3×3 checkerboard + renkli noktalar."""
    if (row // 3 + col // 3) % 2 == 0:
        base = 0.80 + vary * 0.04
        return base, base * 0.95, base * 0.88
    else:
        base = 0.30 + vary * 0.06
        return base * 0.9, base, base * 1.1


ZONES = {
    # (row_min, row_max, col_min, col_max): color_fn
    (0, 10, 0, 10):   asphalt,   # Alt-sol: asfalt
    (0, 10, 10, 20):  grass,     # Alt-sağ: çim
    (10, 20, 0, 10):  concrete,  # Üst-sol: beton
    (10, 20, 10, 20): mixed,     # Üst-sağ: karışık
}


def get_zone_fn(row, col):
    for (r0, r1, c0, c1), fn in ZONES.items():
        if r0 <= row < r1 and c0 <= col < c1:
            return fn
    return concrete


# ── Landmark konumları ve renkleri ────────────────────────────────────────────

LANDMARKS = [
    # (x, y, renk_r, renk_g, renk_b, tip)  — merkez (-10,-10) offset ile orijin merkezde
    ( 3.5,  3.5, 1.0, 0.1, 0.1, "box"),   # kırmızı kutu
    (-3.5,  3.5, 0.1, 0.1, 1.0, "box"),   # mavi kutu
    ( 3.5, -3.5, 1.0, 0.9, 0.0, "cyl"),   # sarı silindir
    (-3.5, -3.5, 0.0, 0.8, 0.2, "cyl"),   # yeşil silindir
    ( 7.5,  7.5, 1.0, 0.4, 0.0, "box"),   # turuncu
    (-7.5,  7.5, 0.7, 0.0, 0.9, "cyl"),   # mor
    ( 7.5, -7.5, 0.0, 0.9, 0.9, "box"),   # camgöbeği
    (-7.5, -7.5, 1.0, 1.0, 0.0, "cyl"),   # parlak sarı
    ( 0.0,  8.0, 0.9, 0.2, 0.5, "box"),   # pembe
    ( 8.0,  0.0, 0.2, 0.8, 0.8, "cyl"),   # teal
    (-8.0,  0.0, 0.8, 0.5, 0.1, "box"),   # kahverengi-turuncu
    ( 0.0, -8.0, 0.5, 0.5, 1.0, "cyl"),   # lavanta
    # Ekstra — iç bölge
    ( 2.0,  6.0, 1.0, 0.0, 0.5, "box"),
    (-2.0,  6.0, 0.0, 0.5, 1.0, "cyl"),
    ( 2.0, -6.0, 1.0, 0.6, 0.0, "box"),
    (-2.0, -6.0, 0.4, 1.0, 0.4, "cyl"),
    ( 6.0,  2.0, 0.2, 0.2, 0.9, "box"),
    (-6.0,  2.0, 0.9, 0.9, 0.1, "cyl"),
    ( 6.0, -2.0, 0.6, 0.0, 0.6, "box"),
    (-6.0, -2.0, 0.0, 0.7, 0.3, "cyl"),
]


def make_world(height: float = 4.0, noise: float = 0.015) -> str:
    import random
    rng = random.Random(77)

    models = []
    tile_count = 0

    # 20×20 zemin tile'ları (1×1m kare)
    # Dünya merkezi (0,0), tile'lar -10..+10 arası
    for row in range(20):
        for col in range(20):
            cx = (col - 9.5)       # -9.5 .. +9.5
            cy = (row - 9.5)
            vary = rng.uniform(0, 1)
            fn = get_zone_fn(row, col)
            r, g, b = fn(row, col, vary)
            name = f"t_{row:02d}_{col:02d}"
            models.append(floor_tile(name, cx, cy, 1.0, 1.0, r, g, b))
            tile_count += 1

    # Landmark'lar
    lm_count = 0
    for i, (lx, ly, lr, lg, lb, ltype) in enumerate(LANDMARKS):
        name = f"lm_{i:02d}"
        if ltype == "box":
            sz_x = rng.uniform(0.5, 1.2)
            sz_y = rng.uniform(0.5, 1.2)
            sz_z = rng.uniform(0.4, 1.5)
            yaw  = rng.uniform(0, math.pi)
            models.append(landmark_box(name, lx, ly, sz_z/2, sz_x, sz_y, sz_z, yaw, lr, lg, lb))
        else:
            r_cyl = rng.uniform(0.3, 0.6)
            h_cyl = rng.uniform(0.5, 1.8)
            models.append(landmark_cyl(name, lx, ly, h_cyl/2, r_cyl, h_cyl, lr, lg, lb))
        lm_count += 1

    # Küçük dekoratif bloklar (texture bölge sınırlarına)
    deco_positions = [
        (-10, 0), (0, -10), (10, 0), (0, 10),  # Sınır orta noktalar
        (-5, -5), (5, -5), (-5, 5), (5, 5),     # Köşe çeyrekler
    ]
    for i, (dx, dy) in enumerate(deco_positions):
        name = f"deco_{i:02d}"
        r = rng.uniform(0.4, 0.9); g = rng.uniform(0.4, 0.9); b = rng.uniform(0.4, 0.9)
        sz = rng.uniform(0.3, 0.7)
        models.append(landmark_box(name, dx, dy, sz/2, sz, sz, sz, 0, r, g, b))

    # Kamera modeli (down-facing, pitch = -80° = -1.396 rad)
    pitch_rad = -1.3963  # -80 deg from horizontal
    camera_model = f"""
    <model name="down_cam_rig">
      <static>true</static>
      <pose>0 0 {height:.2f} 0 0 0</pose>
      <link name="camera_link">
        <sensor name="down_cam" type="camera">
          <pose>0 0 0 0 {pitch_rad:.4f} 0</pose>
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
            <clip><near>0.1</near><far>100</far></clip>
            <noise>
              <type>gaussian</type>
              <mean>0.0</mean>
              <stddev>{noise:.4f}</stddev>
            </noise>
          </camera>
          <topic>/world/competition_world/model/down_cam_rig/link/camera_link/sensor/down_cam/image</topic>
        </sensor>
      </link>
    </model>"""

    models_xml = "\n".join(models) + camera_model

    world_xml = f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="competition_world">

    <!-- Sistem plugin'leri -->
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

    <scene>
      <ambient>0.35 0.35 0.35 1</ambient>
      <background>0.5 0.7 1.0 1</background>
      <shadows>true</shadows>
    </scene>

    <!-- Ana güneş ışığı: cast_shadows, açılı -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 20 0 0 0</pose>
      <direction>-0.6 0.25 -1.0</direction>
      <diffuse>1.0 0.96 0.88 1</diffuse>
      <specular>0.4 0.38 0.35 1</specular>
      <attenuation><range>1000</range><linear>0.001</linear></attenuation>
    </light>

    <!-- Fill light: yumuşak karşı aydınlatma (exposure variation simülasyonu) -->
    <light name="fill_light" type="directional">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 15 0 0 0</pose>
      <direction>0.5 -0.3 -0.8</direction>
      <diffuse>0.30 0.32 0.38 1</diffuse>
      <specular>0.05 0.05 0.05 1</specular>
      <attenuation><range>1000</range><linear>0.005</linear></attenuation>
    </light>

    {models_xml}

  </world>
</sdf>
"""

    return world_xml, tile_count, lm_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=float, default=4.0,
                        help="Kamera yüksekliği metre (varsayılan: 4.0)")
    parser.add_argument("--noise",  type=float, default=0.015,
                        help="Gaussian noise stddev (varsayılan: 0.015)")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    world_xml, tiles, lms = make_world(args.height, args.noise)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(world_xml)

    print(f"competition_world.sdf yazıldı: {args.out}")
    print(f"  Tile sayısı     : {tiles}")
    print(f"  Landmark sayısı : {lms}")
    print(f"  Kamera yüksekliği: {args.height}m  Noise: {args.noise}")
    print(f"  Pitch: -80° ({-80 * math.pi / 180:.4f} rad) — neredeyse düz aşağı bakış")


if __name__ == "__main__":
    main()
