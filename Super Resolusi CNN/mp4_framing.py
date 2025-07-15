import cv2
import os

def video_to_frames(video_path, output_folder):
    # Membuat folder output jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Membuka video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Menyimpan setiap frame sebagai file JPG
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")

# Contoh penggunaan
video_path = '/home/interpolasi/Downloads/SRCNN-TF/camera_brin.avi'
output_folder = '/home/interpolasi/Downloads/SRCNN-TF/camera_brin'
video_to_frames(video_path, output_folder)
