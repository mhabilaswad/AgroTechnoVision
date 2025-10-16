import streamlit as st
import cv2
import time
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# === Streamlit Setup ===
st.set_page_config(page_title="AgroTechnoVision", layout="wide")

# === Header ===
st.title("AgroTechnoVision")
st.subheader("Real-Time Fruit Detection and Classification (YOLOv12 + CNN-MTL)")

st.markdown(
    """
    **Author:** Muhammad Habil Aswad  
    *Universitas Syiah Kuala*
    """,
    unsafe_allow_html=True
)

# === Load models ===
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/YOLOv12n2.pt")
    cnn_model = load_model("model/CNN-MTL.keras")
    return yolo_model, cnn_model

yolo_model, cnn_model = load_models()

# === Label maps ===
fruit_idx2label = {
    0: "alpukat", 1: "durian", 2: "jeruk-siam", 3: "mangga", 4: "nanas",
    5: "nangka", 6: "pepaya", 7: "pisang", 8: "rambutan", 9: "salak"
}
freshness_idx2label = {0: "busuk", 1: "segar", 2: "tidak-segar"}
ripeness_idx2label = {0: "matang", 1: "mentah", 2: "setengah-matang"}

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.1
MAX_MISSED = 10
HISTORY_SIZE = 20

# === Utility ===
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = ((x2 - x1) * (y2 - y1)) + ((x2b - x1b) * (y2b - y1b)) - inter_area
    return inter_area / union_area if union_area > 0 else 0


# === Input Section ===
st.write("---")
source_type = st.radio("Pilih jenis input:", ["Webcam", "Upload Video", "Upload Gambar"], horizontal=True)

uploaded_file = None
if source_type == "Webcam":
    cam_index = st.number_input("Pilih kamera (0=default, 1=external, dst):", min_value=0, max_value=10, value=0, step=1)
elif source_type == "Upload Video":
    uploaded_file = st.file_uploader("Upload file video", type=["mp4", "mov", "avi"])
else:
    uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

col_start, col_stop = st.columns(2)
start_button = col_start.button("Mulai Deteksi", use_container_width=True)
stop_button = col_stop.button("Hentikan", use_container_width=True)
st.write("---")

# === Image Detection ===
if start_button and source_type == "Upload Gambar" and uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert("RGB"))
    results = yolo_model(img_np, imgsz=640, conf=CONF_THRESHOLD, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        crop = img_np[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # CNN classification
        crop_resized = cv2.resize(crop, (256, 256))
        img_arr = img_to_array(crop_resized) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)
        pred_jenis, pred_kesegaran, pred_kematangan = cnn_model.predict(img_arr, verbose=0)
        jenis_label = fruit_idx2label[np.argmax(pred_jenis[0])]
        kesegaran_label = freshness_idx2label[np.argmax(pred_kesegaran[0])]
        kematangan_label = ripeness_idx2label[np.argmax(pred_kematangan[0])]
        label = f"{jenis_label} | {kesegaran_label} | {kematangan_label}"

        # Gambar bounding box merah
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 4)

        # Background merah untuk teks
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        cv2.rectangle(img_np, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (255, 0, 0), -1)
        cv2.putText(img_np, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

    # --- Pastikan tampil proporsional dan tidak scroll ---
    h, w, _ = img_np.shape
    aspect_ratio = w / h
    target_aspect = 16 / 9  # ukuran desktop standar

    if aspect_ratio < target_aspect:
        # Tambahkan padding kiri-kanan hitam
        new_w = int(h * target_aspect)
        pad = (new_w - w) // 2
        img_np = cv2.copyMakeBorder(img_np, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    elif aspect_ratio > target_aspect:
        # Tambahkan padding atas-bawah hitam (jarang tapi tetap aman)
        new_h = int(w / target_aspect)
        pad = (new_h - h) // 2
        img_np = cv2.copyMakeBorder(img_np, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    st.image(img_np, channels="RGB", use_container_width=True)


# === Video / Webcam Detection ===
elif start_button and source_type in ["Webcam", "Upload Video"]:
    if source_type == "Webcam":
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            st.error("Tidak bisa membuka kamera.")
            st.stop()
    else:
        if uploaded_file is None:
            st.warning("Harap upload file video terlebih dahulu.")
            st.stop()
        tfile = "temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile)

    stframe = st.empty()
    tracked_objects = {}
    next_id = 0

    while True:
        if stop_button:
            st.info("Deteksi dihentikan.")
            break

        ret, frame = cap.read()
        if not ret:
            st.success("Video selesai.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        start_yolo = time.time()
        results = yolo_model(frame_rgb, imgsz=640, conf=CONF_THRESHOLD, verbose=False)[0]
        yolo_time = time.time() - start_yolo

        detections = []
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy()
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = map(int, coords)
            conf = float(box.conf[0])
            if (x2 - x1) * (y2 - y1) < 2000:
                continue
            detections.append((x1, y1, x2, y2, conf))

        updated_ids = set()

        # === IoU Matching dan Klasifikasi (dengan delay 2 detik) ===
        current_time = time.time()
        for det in detections:
            x1, y1, x2, y2, conf = det
            best_iou, best_id = 0, None
            for obj_id, info in tracked_objects.items():
                current_iou = iou(info['box'], (x1, y1, x2, y2))
                if current_iou > best_iou:
                    best_iou, best_id = current_iou, obj_id

            if best_iou > IOU_THRESHOLD:
                # Update posisi box & reset missed counter
                tracked_objects[best_id]['box'] = (
                    int(0.7 * tracked_objects[best_id]['box'][0] + 0.3 * x1),
                    int(0.7 * tracked_objects[best_id]['box'][1] + 0.3 * y1),
                    int(0.7 * tracked_objects[best_id]['box'][2] + 0.3 * x2),
                    int(0.7 * tracked_objects[best_id]['box'][3] + 0.3 * y2),
                )
                tracked_objects[best_id]['missed'] = 0
                updated_ids.add(best_id)

                # Cek apakah sudah lebih dari 2 detik sejak pertama terdeteksi
                if 'first_seen' in tracked_objects[best_id]:
                    if current_time - tracked_objects[best_id]['first_seen'] >= 2 and not tracked_objects[best_id]['classified']:
                        x1b, y1b, x2b, y2b = tracked_objects[best_id]['box']
                        crop = frame_rgb[y1b:y2b, x1b:x2b]
                        if crop.size > 0:
                            crop_resized = cv2.resize(crop, (256, 256))
                            img_arr = img_to_array(crop_resized) / 255.0
                            img_arr = np.expand_dims(img_arr, axis=0)
                            pred_jenis, pred_kesegaran, pred_kematangan = cnn_model.predict(img_arr, verbose=0)
                            tracked_objects[best_id]['pred_jenis'] = [pred_jenis[0]]
                            tracked_objects[best_id]['pred_kesegaran'] = [pred_kesegaran[0]]
                            tracked_objects[best_id]['pred_kematangan'] = [pred_kematangan[0]]
                            tracked_objects[best_id]['classified'] = True
                else:
                    tracked_objects[best_id]['first_seen'] = current_time

            else:
                # Objek baru: simpan timestamp awal & tandai belum diklasifikasi
                tracked_objects[next_id] = {
                    'box': (x1, y1, x2, y2),
                    'pred_jenis': [],
                    'pred_kesegaran': [],
                    'pred_kematangan': [],
                    'missed': 0,
                    'first_seen': current_time,
                    'classified': False
                }
                updated_ids.add(next_id)
                next_id += 1

        # === Update objek lama ===
        for obj_id in list(tracked_objects.keys()):
            if obj_id not in updated_ids:
                tracked_objects[obj_id]['missed'] += 1
                if tracked_objects[obj_id]['missed'] > MAX_MISSED:
                    del tracked_objects[obj_id]

        # === Tampilkan hasil ===
        for obj_id, info in tracked_objects.items():
            x1, y1, x2, y2 = info['box']

            if not info.get('classified', False):
                label = "Detecting..."
            else:
                avg_jenis = np.mean(info['pred_jenis'], axis=0)
                avg_kesegaran = np.mean(info['pred_kesegaran'], axis=0)
                avg_kematangan = np.mean(info['pred_kematangan'], axis=0)

                jenis_label = fruit_idx2label[np.argmax(avg_jenis)]
                kesegaran_label = freshness_idx2label[np.argmax(avg_kesegaran)]
                kematangan_label = ripeness_idx2label[np.argmax(avg_kematangan)]
                label = f"{jenis_label} | {kesegaran_label} | {kematangan_label}"

            # Batasi panjang riwayat prediksi
            if len(info['pred_jenis']) > HISTORY_SIZE:
                info['pred_jenis'].pop(0)
                info['pred_kesegaran'].pop(0)
                info['pred_kematangan'].pop(0)

            # Gambar bounding box merah
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

            # Background merah untuk teks
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 0, 255), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

        # === Tambahkan FPS info ===
        fps = 1 / yolo_time if yolo_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

        # === Padding agar proporsional & tidak perlu scroll ===
        h, w, _ = frame.shape
        aspect_ratio = w / h
        target_aspect = 16 / 9  # rasio layar umum desktop

        if aspect_ratio < target_aspect:
            # Tambahkan padding kiri-kanan hitam
            new_w = int(h * target_aspect)
            pad = (new_w - w) // 2
            frame = cv2.copyMakeBorder(frame, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        elif aspect_ratio > target_aspect:
            # Tambahkan padding atas-bawah hitam (kalau terlalu lebar)
            new_h = int(w / target_aspect)
            pad = (new_h - h) // 2
            frame = cv2.copyMakeBorder(frame, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # === Tampilkan video tetap dalam satu layar ===
        stframe.image(frame, channels="BGR", use_container_width=True, output_format="BGR")

        # === Kontrol kecepatan agar tidak slow-motion ===
        # Gunakan waktu proses YOLO dan tampilkan sesuai FPS video asli
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or np.isnan(video_fps):
            video_fps = 24  # fallback

        # Sinkronisasi ringan (jangan 1/video_fps karena YOLO sudah ambil waktu)
        frame_delay = max(1.0 / video_fps - yolo_time, 0)
        time.sleep(frame_delay)


    cap.release()