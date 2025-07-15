import os
import cv2
import numpy as np
import tensorflow as tf
from eval import interpolator

# ========== PENGATURAN INPUT ==========
frame1_path = '/home/interpolasi/Downloads/FRUC/FILM/input/12.png'   # Ganti dengan path gambar pertama
frame2_path = '/home/interpolasi/Downloads/FRUC/FILM/input/14.png'   # Ganti dengan path gambar kedua
model_path = 'pretrained_models/film_net/Style/saved_model'
output_path = 'input/13.png'

# ========== MUAT MODEL ==========
interpolator_model = interpolator.Interpolator(model_path)

# ========== BACA DAN SIAPKAN FRAME ==========
frame1 = cv2.imread(frame1_path)[:, :, ::-1] / 255.0  # BGR → RGB dan normalisasi
frame2 = cv2.imread(frame2_path)[:, :, ::-1] / 255.0

assert frame1.shape == frame2.shape, "❌ Ukuran dua gambar tidak sama!"

# ========== INTERPOLASI FRAME TENGAH ==========
dt = np.array([0.5], dtype=np.float32)  # waktu di tengah

image_batch = interpolator_model.interpolate(
    tf.convert_to_tensor(frame1[np.newaxis, ...], dtype=tf.float32),
    tf.convert_to_tensor(frame2[np.newaxis, ...], dtype=tf.float32),
    dt=dt
)

# ========== SIMPAN HASIL ==========
image_tensor = image_batch['image'] if isinstance(image_batch, dict) else image_batch
image_tensor = image_tensor.numpy() if isinstance(image_tensor, tf.Tensor) else image_tensor
interpolated = np.clip(image_tensor[0], 0.0, 1.0)

cv2.imwrite(output_path, (interpolated * 255).astype(np.uint8)[:, :, ::-1])  # RGB → BGR
print(f"✅ Interpolasi selesai. Hasil disimpan di: {output_path}")




