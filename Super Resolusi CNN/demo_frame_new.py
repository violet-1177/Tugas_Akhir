import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils.common import *
from model import SRCNN

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir',    type=str, default='frames/', help='Folder berisi gambar input (LR)')
parser.add_argument('--scale',        type=int, default=2, help='Upscale factor: 2, 3, or 4')
parser.add_argument('--architecture', type=str, default="915", help='SRCNN architecture: 915, 935, 955')
parser.add_argument("--ckpt-path",    type=str, default="", help='Path ke model checkpoint')

FLAGS = parser.parse_args()

# Validasi arsitektur dan scale
scale = FLAGS.scale
architecture = FLAGS.architecture
pad = int(architecture[1]) // 2 + 6
sigma = 0.3 if scale == 2 else 0.2

if FLAGS.ckpt_path == "" or FLAGS.ckpt_path == "default":
    ckpt_path = f"checkpoint/SRCNN{architecture}/SRCNN-{architecture}.h5"
else:
    ckpt_path = FLAGS.ckpt_path

# Aktifkan memory growth (Jetson safe)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"[ERROR] Cannot set memory growth: {e}")

# Load model
model = SRCNN(architecture)
model.load_weights(ckpt_path)

# Siapkan list untuk menyimpan nilai PSNR
psnr_scores = []
image_names = []

# Buat folder output
os.makedirs("output_sr", exist_ok=True)

# Proses semua gambar di folder
for filename in sorted(os.listdir(FLAGS.input_dir)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        filepath = os.path.join(FLAGS.input_dir, filename)
        image_names.append(filename)

        # Load gambar LR
        lr_image = read_image(filepath)

        # Resize otomatis jika terlalu besar
        MAX_RES = 320
        h, w = lr_image.shape[:2]
        if max(h, w) > MAX_RES:
            factor = MAX_RES / max(h, w)
            lr_image = tf.image.resize(lr_image, (int(h * factor), int(w * factor)), method='bicubic')
            lr_image = tf.cast(lr_image, tf.uint8)

        # Buat bicubic referensi
        bicubic_image = upscale(lr_image, scale)
        bicubic_cropped = bicubic_image[pad:-pad, pad:-pad]

        # Preprocess input
        blur = gaussian_blur(lr_image, sigma=sigma)
        bicubic_input = upscale(blur, scale)
        bicubic_input = rgb2ycbcr(bicubic_input)
        bicubic_input = norm01(bicubic_input)
        bicubic_input = tf.expand_dims(bicubic_input, axis=0)

        # Inference
        sr = model.predict(bicubic_input)[0]
        sr = denorm01(sr)
        sr = tf.cast(sr, tf.uint8)
        sr = ycbcr2rgb(sr)

        # Simpan output sebagai .png apapun inputnya
        base_name = os.path.splitext(filename)[0]  # hapus ekstensi
        output_filename = f"HR_{base_name}.png"
        out_path = os.path.join("output_sr", output_filename)
        write_image(out_path, sr)

        # Hitung PSNR
        sr_np = sr.numpy().astype("uint8")
        bicubic_np = bicubic_cropped.numpy().astype("uint8")
        psnr_val = psnr(bicubic_np, sr_np, data_range=255)
        psnr_scores.append(psnr_val)

        print(f"[INFO] {filename} → PSNR: {psnr_val:.2f} dB | disimpan: {output_filename}")

# Hitung rata-rata PSNR
avg_psnr = sum(psnr_scores) / len(psnr_scores)
print(f"\n[RATA-RATA] PSNR semua gambar: {avg_psnr:.2f} dB")

# Buat grafik PSNR
plt.figure(figsize=(10, 5))
plt.plot(image_names, psnr_scores, marker='o', linestyle='-', label='PSNR per frame')
plt.axhline(avg_psnr, color='red', linestyle='--', label=f'Rata-rata: {avg_psnr:.2f} dB')
plt.xticks(rotation=45)
plt.xlabel('Nama Gambar')
plt.ylabel('PSNR (dB)')
plt.title('PSNR per Gambar')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("psnr_plot.png")
plt.show()




