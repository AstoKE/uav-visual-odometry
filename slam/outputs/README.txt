SLAM Çıktıları — Notlar
======================

Bu dizin DROID-SLAM çalışmalarının çıktılarını tutar.

Hedef dosya (sonraki aşamada üretilecek):
  slam/outputs/trajectory.csv
  Format: frame,x,y,z

---------------------------------------------------------------------------
POSE EXPORT İÇİN NEREYE MÜDAHALE EDİLECEK
---------------------------------------------------------------------------

DROID-SLAM, pose'ları `demo.py` sonunda `droid.video` nesnesinde tutar.
Trajectory verisi `DepthVideo` sınıfının `poses` tensöründedir.

>> demo.py içindeki müdahale noktası:

    # demo.py, yaklaşık satır ~60-80 (model.track() çağrısından sonra):
    from droid_slam.droid import Droid
    ...
    droid = Droid(args)
    ...
    for (t, image, intrinsics) in tqdm(image_stream(args)):
        droid.track(t, image, intrinsics=intrinsics)

    # Buradan sonra pose'lar `droid.video.poses` içinde:
    #   shape: [N, 7]  (quaternion + translation: qx qy qz qw x y z)
    #   dtype: torch.float32, device: cuda

>> Eklenecek export kodu (demo.py sonuna):

    import csv, torch
    poses = droid.video.poses[:droid.video.counter].cpu().numpy()
    out_csv = os.path.join(
        os.path.dirname(__file__),
        "../slam/outputs/trajectory.csv"
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "z"])
        for i, p in enumerate(poses):
            # p = [qx, qy, qz, qw, x, y, z]
            writer.writerow([i, float(p[4]), float(p[5]), float(p[6])])
    print(f"Trajectory kaydedildi: {out_csv}")

---------------------------------------------------------------------------
DİKKAT: demo.py'yi şu an değiştirme.
Önce küçük dataset ile `run_droid_small.sh` çalışıyor mu doğrula,
sonra yukarıdaki export kodu eklenebilir.
---------------------------------------------------------------------------

Dosyalar:
  trajectory.csv   — (henüz yok) pose export çalıştırılınca oluşacak
  README.txt       — bu dosya
