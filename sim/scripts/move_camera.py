#!/usr/bin/env python3
"""
move_camera.py — Gazebo Sim içindeki down_cam_rig modelini hareket ettirir.

Dört hareket deseni desteklenir (PATTERN env var ile seçilir):

  figure8     (varsayılan) — Figure-8 / ∞ deseni
  lawnmower   — Biçerdöver deseni
  competition — 5 dakikalık yarışma senaryosu (PDF uyumlu):
                  Faz 1 (0–60s)  : Smooth kalibrasyon hareketi (figure-8)
                  Faz 2 (60–300s): Rastgele hız/yön değişimleri + kısa durmalar
                                   + yükseklik jitter (3–5m)
  spiral      — Artan yarıçaplı spiral (test amaçlı)

Kullanım:
    python3 ~/code/uav-visual-odometry/sim/scripts/move_camera.py
    PATTERN=competition python3 sim/scripts/move_camera.py

Seçenekler (ortam değişkeni ile):
    PATTERN        : figure8 | lawnmower | competition | spiral
    WORLD_NAME     : Gazebo world adı     (varsayılan: slam_world)
    MODEL_NAME     : Hareket ettirilecek model (varsayılan: down_cam_rig)
    STEP_DELAY     : Adımlar arası bekleme sn  (varsayılan: 0.08)
    HEIGHT         : Kamera yüksekliği         (varsayılan: 4.0)
    COMP_SEED      : Competition deseni rastlantı tohumu (varsayılan: 42)

  Figure-8 parametreleri:
    FIGURE8_AX     : X genliği metre           (varsayılan: 4.5)
    FIGURE8_AY     : Y genliği metre           (varsayılan: 3.5)
    FIGURE8_PERIOD : Bir tur süresi saniye      (varsayılan: 25.0)
    FIGURE8_CYCLES : Tur sayısı                (varsayılan: 3.5)

  Lawnmower parametreleri:
    STEP_SIZE      : Adım başına metre         (varsayılan: 0.30)

Ön koşul:
    - Gazebo Sim çalışıyor olmalı (run_gazebo.sh)
    - `gz` CLI kullanılabilir olmalı
"""

import math
import os
import subprocess
import time

# ── Ortak ayarlar ─────────────────────────────────────────────────────────────
PATTERN     = os.environ.get("PATTERN",     "figure8")
WORLD_NAME  = os.environ.get("WORLD_NAME",  "slam_world")
MODEL_NAME  = os.environ.get("MODEL_NAME",  "down_cam_rig")
STEP_DELAY  = float(os.environ.get("STEP_DELAY", "0.08"))
HEIGHT      = float(os.environ.get("HEIGHT",     "4.0"))
COMP_SEED   = int(os.environ.get("COMP_SEED",    "42"))

# ── Figure-8 parametreleri ────────────────────────────────────────────────────
FIGURE8_AX     = float(os.environ.get("FIGURE8_AX",     "4.5"))
FIGURE8_AY     = float(os.environ.get("FIGURE8_AY",     "3.5"))
FIGURE8_PERIOD = float(os.environ.get("FIGURE8_PERIOD", "25.0"))
FIGURE8_CYCLES = float(os.environ.get("FIGURE8_CYCLES", "3.5"))

# ── Lawnmower parametreleri ───────────────────────────────────────────────────
STEP_SIZE   = float(os.environ.get("STEP_SIZE", "0.30"))

SERVICE = f"/world/{WORLD_NAME}/set_pose"


def set_pose(x: float, y: float, z: float) -> bool:
    """gz service çağrısıyla modeli belirtilen konuma taşır."""
    req = (
        f'name: "{MODEL_NAME}", '
        f'position: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}}, '
        f'orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}'
    )
    cmd = [
        "gz", "service",
        "-s", SERVICE,
        "--reqtype", "gz.msgs.Pose",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "1000",
        "--req", req,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
    return result.returncode == 0


def generate_figure8(
    amplitude_x: float = FIGURE8_AX,
    amplitude_y: float = FIGURE8_AY,
    period:      float = FIGURE8_PERIOD,
    n_cycles:    float = FIGURE8_CYCLES,
    step_delay:  float = STEP_DELAY,
) -> list[tuple[float, float]]:
    """
    Figure-8 (∞) yolu üretir.

    Parametrik denklem (bir tam döngü için θ: 0 → 2π):
        x(θ) = A_x * sin(θ)
        y(θ) = A_y * sin(2θ)   ← iki kat frekans → 8 şekli

    Dünya sınırları içinde kalır: |x| ≤ A_x ≤ 6m, |y| ≤ A_y ≤ 5m
    """
    n_steps = int(round(n_cycles * period / step_delay))
    waypoints = []
    for i in range(n_steps):
        t = i * step_delay
        theta = 2.0 * math.pi * t / period
        x = amplitude_x * math.sin(theta)
        y = amplitude_y * math.sin(2.0 * theta)
        waypoints.append((round(x, 6), round(y, 6)))
    return waypoints


def generate_lawnmower(
    x_min: float = -8.0, x_max: float = 8.0,
    y_min: float = -8.0, y_max: float = 8.0,
    step:  float = STEP_SIZE,
) -> list[tuple[float, float]]:
    """
    Biçerdöver (lawnmower) yolu üretir.
    Çift satırlarda soldan sağa, tek satırlarda sağdan sola.
    """
    waypoints = []
    row = 0
    y = y_min
    while y <= y_max + 1e-6:
        xs = []
        x = x_min
        while x <= x_max + 1e-6:
            xs.append(round(x, 4))
            x += step
        if row % 2 == 1:
            xs = xs[::-1]
        for px in xs:
            waypoints.append((px, round(y, 4)))
        y += step * 4
        row += 1
    return waypoints


def generate_competition(
    total_time:  float = 300.0,
    step_delay:  float = STEP_DELAY,
    calib_time:  float = 60.0,
    amplitude_x: float = FIGURE8_AX,
    amplitude_y: float = FIGURE8_AY,
    period:      float = FIGURE8_PERIOD,
    height_base: float = HEIGHT,
    seed:        int   = COMP_SEED,
) -> list[tuple[float, float, float]]:
    """
    5 dakikalık gerçekçi yarışma senaryosu (z koordinatı dahil).

    Faz 1 — Kalibrasyon (0 – calib_time):
        Düzgün figure-8 hareketi, sabit yükseklik.
        Yarışma sunucusu bu sürede health=1 gönderir.

    Faz 2 — GPS-denied mod (calib_time – total_time):
        • Rastgele hız modülasyonu (0.3× – 2.0× nominal hız)
        • Ani yön değişimleri (velocity reversal, ~10–20s'de bir)
        • Kısa durmalar (2–5s, ~30s'de bir)
        • Yükseklik jitter: 3–5m aralığında düzgün geçişler
        • Sınır çerçevesi: ±7m × ±7m

    Returns: list of (x, y, z) tuples — her biri STEP_DELAY sn arayla
    """
    import random
    rng = random.Random(seed)

    n_total = int(round(total_time / step_delay))
    n_calib = int(round(calib_time  / step_delay))

    result: list[tuple[float, float, float]] = []

    # ── Faz 1: Düzgün figure-8 ────────────────────────────────────────────────
    for i in range(n_calib):
        t     = i * step_delay
        theta = 2.0 * math.pi * t / period
        x     = amplitude_x * math.sin(theta)
        y     = amplitude_y * math.sin(2.0 * theta)
        result.append((round(x, 5), round(y, 5), round(height_base, 4)))

    # ── Faz 2: Karmaşık hareket ───────────────────────────────────────────────
    # Durum başlatma — kalibrasyon sonundaki konumdan devam
    cur_x, cur_y = result[-1][0], result[-1][1]
    cur_z        = height_base

    # Hedef yön ve hız
    vel_x = rng.uniform(-0.3, 0.3)
    vel_y = rng.uniform(-0.3, 0.3)

    # Planlama takvimi (saniye cinsinden olaylar)
    next_direction_change = calib_time + rng.uniform(8.0, 18.0)
    next_stop             = calib_time + rng.uniform(25.0, 40.0)
    stop_end              = 0.0
    next_height_change    = calib_time + rng.uniform(10.0, 25.0)
    target_height         = height_base

    for i in range(n_calib, n_total):
        t = i * step_delay

        # — Duraklama modu
        if t < stop_end:
            result.append((round(cur_x, 5), round(cur_y, 5), round(cur_z, 4)))
            continue

        # — Ani yön değişimi tetikleyici
        if t >= next_direction_change:
            # Yeni rastgele hedef yön seç
            target_x = rng.uniform(-amplitude_x * 0.9, amplitude_x * 0.9)
            target_y = rng.uniform(-amplitude_y * 0.9, amplitude_y * 0.9)
            dist     = math.hypot(target_x - cur_x, target_y - cur_y) + 1e-6
            speed    = rng.uniform(0.08, 0.25)          # m/step
            vel_x    = (target_x - cur_x) / dist * speed
            vel_y    = (target_y - cur_y) / dist * speed
            next_direction_change = t + rng.uniform(8.0, 20.0)

        # — Duraklama tetikleyici
        if t >= next_stop:
            stop_duration = rng.uniform(1.5, 5.0)       # saniye
            stop_end      = t + stop_duration
            next_stop     = stop_end + rng.uniform(20.0, 45.0)
            result.append((round(cur_x, 5), round(cur_y, 5), round(cur_z, 4)))
            continue

        # — Yükseklik değişimi (3–5m arası yumuşak geçiş)
        if t >= next_height_change:
            target_height      = rng.uniform(3.0, 5.0)
            next_height_change = t + rng.uniform(15.0, 35.0)
        # Yüksekliğe doğru kademeli yaklaşma (0.01m/step)
        if abs(cur_z - target_height) > 0.02:
            cur_z += 0.01 * math.copysign(1.0, target_height - cur_z)

        # — Hız gürültüsü (küçük rastgele pertürbasyon)
        vel_x += rng.gauss(0.0, 0.005)
        vel_y += rng.gauss(0.0, 0.005)

        # — Sınır yansıması (duvardan sekme)
        next_x = cur_x + vel_x
        next_y = cur_y + vel_y
        if abs(next_x) > 7.0:
            vel_x  = -vel_x * rng.uniform(0.7, 1.0)
            next_x = max(-7.0, min(7.0, next_x))
        if abs(next_y) > 7.0:
            vel_y  = -vel_y * rng.uniform(0.7, 1.0)
            next_y = max(-7.0, min(7.0, next_y))

        cur_x, cur_y = next_x, next_y
        result.append((round(cur_x, 5), round(cur_y, 5), round(cur_z, 4)))

    return result


def generate_spiral(
    max_radius:  float = 6.0,
    height:      float = HEIGHT,
    step_delay:  float = STEP_DELAY,
    turns:       float = 4.0,
) -> list[tuple[float, float, float]]:
    """
    Artan yarıçaplı spiral yol (test ve kalibrasyon doğrulama için).
    """
    n = int(round(turns * 2 * math.pi / 0.05))
    result = []
    for i in range(n):
        theta  = i * 0.05
        radius = max_radius * theta / (turns * 2 * math.pi)
        x      = radius * math.cos(theta)
        y      = radius * math.sin(theta)
        result.append((round(x, 5), round(y, 5), round(height, 4)))
    return result


def main() -> None:
    # ── Waypoint üret ─────────────────────────────────────────────────────────
    # Tüm desenler (x, y, z) üçlüsü döndürür. Eski figure8/lawnmower z=HEIGHT sabit.
    if PATTERN == "figure8":
        raw = generate_figure8()
        waypoints = [(x, y, HEIGHT) for x, y in raw]
        pattern_info = (
            f"Figure-8  A_x={FIGURE8_AX}m  A_y={FIGURE8_AY}m  "
            f"T={FIGURE8_PERIOD}s  cycles={FIGURE8_CYCLES}"
        )
    elif PATTERN == "lawnmower":
        raw = generate_lawnmower()
        waypoints = [(x, y, HEIGHT) for x, y in raw]
        pattern_info = f"Lawnmower  step={STEP_SIZE}m"
    elif PATTERN == "competition":
        waypoints = generate_competition(seed=COMP_SEED)
        calib_n   = int(60.0 / STEP_DELAY)
        pattern_info = (
            f"Competition  300s  calib={calib_n}wp  "
            f"phase2={len(waypoints)-calib_n}wp  seed={COMP_SEED}"
        )
    elif PATTERN == "spiral":
        waypoints = generate_spiral()
        pattern_info = "Spiral  turns=4"
    else:
        raise ValueError(
            f"Bilinmeyen PATTERN: '{PATTERN}'. "
            "figure8 / lawnmower / competition / spiral olmalı."
        )

    total_time = len(waypoints) * STEP_DELAY

    print(f"[move_camera] World   : {WORLD_NAME}")
    print(f"[move_camera] Model   : {MODEL_NAME}")
    print(f"[move_camera] Height  : {HEIGHT} m")
    print(f"[move_camera] Pattern : {pattern_info}")
    print(f"[move_camera] Adım    : {STEP_DELAY} s/adım")
    print(f"[move_camera] Waypoint: {len(waypoints)}")
    print(f"[move_camera] Süre    : {total_time:.1f} s  (~{total_time/60:.1f} dk)")
    print("[move_camera] Hareket başlıyor... (durdurmak için Ctrl+C)")
    print()

    # ── Başlangıç konumuna git ────────────────────────────────────────────────
    sx, sy, sz = waypoints[0]
    if not set_pose(sx, sy, sz):
        print("[move_camera] UYARI: İlk pose ayarlanamadı. Gazebo çalışıyor mu?")

    time.sleep(1.0)

    total   = len(waypoints)
    success = 0
    fail    = 0

    for idx, (wx, wy, wz) in enumerate(waypoints):
        try:
            ok = set_pose(wx, wy, wz)
            if ok:
                success += 1
            else:
                fail += 1
                if fail > 5:
                    print("[move_camera] Çok fazla hata, durduruluyor.")
                    break

            if idx % 100 == 0 or idx == total - 1:
                pct     = (idx + 1) / total * 100
                elapsed = (idx + 1) * STEP_DELAY
                print(f"  [{idx+1:5d}/{total}] ({pct:5.1f}%)  "
                      f"x={wx:+7.3f}  y={wy:+7.3f}  z={wz:.2f}  "
                      f"t={elapsed:6.1f}s  ok={success}  fail={fail}")

            time.sleep(STEP_DELAY)

        except KeyboardInterrupt:
            print(f"\n[move_camera] Kullanıcı tarafından durduruldu. "
                  f"Tamamlanan: {idx}/{total}")
            break
        except subprocess.TimeoutExpired:
            fail += 1
            print(f"[move_camera] Timeout: waypoint {idx} atlandı.")

    print(f"\n[move_camera] Bitti. Başarılı: {success}  Başarısız: {fail}")


if __name__ == "__main__":
    main()
