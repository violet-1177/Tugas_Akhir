import os
import glob
import cv2

# ========== PENGATURAN ==========
frame_folder = 'output/interpolated_frames_camera_brin'   # ← Ubah ke folder gambar kamu
output_video_path = 'output/interpolated_video_camera_brin.mp4'       # ← Nama file video yang dihasilkan
fps = 8                                                   # ← FPS video (frame per detik)

# Ambil semua file frame yang diurutkan
frame_paths = sorted(glob.glob(os.path.join(frame_folder, '*.png')))
assert frame_paths, f"Tidak ada file .png ditemukan di {frame_folder}"

# Baca dimensi dari frame pertama
first_frame = cv2.imread(frame_paths[0])
height, width, _ = first_frame.shape

# Siapkan penulis video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Tambahkan semua frame ke video
for frame_path in frame_paths:
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"⚠️ Gagal membaca {frame_path}, dilewati.")
        continue
    video_writer.write(frame)

video_writer.release()
print(f"✅ Video selesai dibuat: {output_video_path} (FPS: {fps})")



