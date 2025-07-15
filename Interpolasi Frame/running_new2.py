import os
import glob
import cv2
import numpy as np
import tensorflow as tf
import subprocess
from eval import interpolator

# ===================== PENGATURAN =====================

model_path = 'pretrained_models/film_net/Style/saved_model'
input_folder = 'input/camera_brin'
output_frame_folder = 'output/interpolated_frames_camera_brin'
output_video_path = 'output/interpolated_output_canera_brin.mp4'

block_shape = (4, 4)          # Seperti --block_height / --block_width
align = 64                    # Padding agar dimensi sesuai model
times_to_interpolate = 1      # Sama dengan --times_to_interpolate (N)
fps = 22                       # Frame per second untuk video hasil

# ======================================================

os.makedirs(output_frame_folder, exist_ok=True)

# Load model
interpolator_model = interpolator.Interpolator(
    model_path,
    align=align,
    block_shape=block_shape
)

# Ambil semua frame input
frame_paths = sorted(glob.glob(os.path.join(input_folder, '*.png')))
assert len(frame_paths) >= 2, "❌ Minimal 2 frame input!"

idx = 0
for i in range(len(frame_paths) - 1):
    # Baca dan normalisasi input
    frame1 = cv2.imread(frame_paths[i])[:, :, ::-1] / 255.0
    frame2 = cv2.imread(frame_paths[i + 1])[:, :, ::-1] / 255.0

    # Validasi ukuran frame
    assert frame1.shape == frame2.shape, f"❌ Ukuran frame tidak sama: {frame_paths[i]} dan {frame_paths[i+1]}"

    # Simpan frame pertama
    cv2.imwrite(os.path.join(output_frame_folder, f'{idx:04d}.png'),
                (frame1 * 255).astype(np.uint8)[:, :, ::-1])
    idx += 1

    # Hitung waktu interpolasi (dt)
    N = times_to_interpolate
    dt_values = [i / (2 ** N) for i in range(1, 2 ** N, 2)]
    dt = np.array(dt_values, dtype=np.float32)

    # Interpolasi
    image_batch = interpolator_model.interpolate(
        tf.convert_to_tensor(frame1[np.newaxis, ...], dtype=tf.float32),
        tf.convert_to_tensor(frame2[np.newaxis, ...], dtype=tf.float32),
        dt=dt
    )

    # Ambil hasil
    image_tensor = image_batch['image'] if isinstance(image_batch, dict) else image_batch
    image_tensor = image_tensor.numpy() if isinstance(image_tensor, tf.Tensor) else image_tensor

    for j in range(len(dt)):
        frame_interp = np.clip(image_tensor[j], 0.0, 1.0)  # ⛑️ pastikan dalam range [0, 1]

        # Simpan frame interpolasi
        cv2.imwrite(os.path.join(output_frame_folder, f'{idx:04d}.png'),
                    (frame_interp * 255).astype(np.uint8)[:, :, ::-1])
        idx += 1

# Simpan frame terakhir
frame_last = cv2.imread(frame_paths[-1])[:, :, ::-1] / 255.0
cv2.imwrite(os.path.join(output_frame_folder, f'{idx:04d}.png'),
            (frame_last * 255).astype(np.uint8)[:, :, ::-1])

print("✅ Interpolasi selesai. Semua frame disimpan di:", output_frame_folder)

# Buat video dari frame (pakai ffmpeg)
ffmpeg_command = [
    'ffmpeg',
    '-y',
    '-framerate', str(fps),
    '-i', os.path.join(output_frame_folder, '%04d.png'),
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    output_video_path
]

print("🎬 Membuat video...")
subprocess.run(ffmpeg_command)
print("✅ Video disimpan di:", output_video_path)
