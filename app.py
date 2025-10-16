import time
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# WebRTC
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ================= Streamlit Setup =================
st.set_page_config(page_title="AgroTechnoVision", layout="wide")

st.title("AgroTechnoVision")
st.subheader("Real-Time Fruit Detection and Classification (YOLOv12 + CNN-MTL)")
st.markdown(
    """
    **Author:** Muhammad Habil Aswad  
    *Universitas Syiah Kuala*
    """,
    unsafe_allow_html=True
)

# ================= Load models (cached) =================
@st.cache_resource
def load_models():
    # paths required by user
    yolo_model = YOLO("model/YOLOv12n2.pt")
    cnn_model = load_model("model/CNN-MTL.keras")
    return yolo_model, cnn_model

yolo_model, cnn_model = load_models()

# ================= Labels & params =================
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
CLASSIFY_DELAY_SECONDS = 2.0  # tunggu 2 detik sebelum klasifikasi objek baru

# ================= Helpers =================
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    X1, Y1, X2, Y2 = box2
    inter_w = max(0, min(x2, X2) - max(x1, X1))
    inter_h = max(0, min(y2, Y2) - max(y1, Y1))
    inter = inter_w * inter_h
    area1 = max(0, (x2 - x1)) * max(0, (y2 - y1))
    area2 = max(0, (X2 - X1)) * max(0, (Y2 - Y1))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def draw_label_on_image(img, box, label):
    x1, y1, x2, y2 = box
    # bounding box merah tebal
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
    # teks besar putih di background merah
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    # pastikan tidak keluar gambar di atas
    y_text_top = max(0, y1 - text_h - 10)
    cv2.rectangle(img, (x1, y_text_top), (x1 + text_w + 10, y1), (0, 0, 255), -1)
    cv2.putText(img, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

def classify_crop(crop):
    crop_resized = cv2.resize(crop, (256, 256))
    img_arr = img_to_array(crop_resized) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    preds = cnn_model.predict(img_arr, verbose=0)
    # cnn_model returns three outputs (p_fruit, p_fresh, p_ripen)
    return preds  # list/tuple of three arrays each shape (1, N)

# ================= UI Input =================
st.write("---")
source_type = st.radio("Pilih jenis input:", ["Webcam (Browser)", "Upload Video", "Upload Gambar"], horizontal=True)

uploaded_file = None
if source_type == "Upload Video":
    uploaded_file = st.file_uploader("Upload file video", type=["mp4", "mov", "avi"])
elif source_type == "Upload Gambar":
    uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

col_start, col_stop = st.columns(2)
start_button = col_start.button("Mulai Deteksi", use_container_width=True)
stop_button = col_stop.button("Hentikan", use_container_width=True)
st.write("---")

# ================= WebRTC Video Processor =================
class RTProcessor(VideoTransformerBase):
    def __init__(self):
        # tracking state per instance (per user session)
        self.tracked = {}   # id -> info dict
        self.next_id = 0

    def transform(self, frame):
        # Called for every frame from browser (in worker thread)
        img = frame.to_ndarray(format="bgr24")  # BGR
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        results = yolo_model(frame_rgb, imgsz=640, conf=CONF_THRESHOLD, verbose=False)[0]
        yolo_time = time.time() - start_time

        detections = []
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy()
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = map(int, coords[:4])
            conf = float(box.conf[0])
            if (x2 - x1) * (y2 - y1) < 2000:
                continue
            detections.append((x1, y1, x2, y2, conf))

        updated_ids = set()
        current_time = time.time()

        # match & update
        for det in detections:
            x1, y1, x2, y2, conf = det
            best_i, best_id = 0, None
            for oid, info in self.tracked.items():
                i = iou(info['box'], (x1, y1, x2, y2))
                if i > best_i:
                    best_i, best_id = i, oid

            if best_i > IOU_THRESHOLD:
                # update smooth box
                prev = self.tracked[best_id]['box']
                new_box = (
                    int(0.7 * prev[0] + 0.3 * x1),
                    int(0.7 * prev[1] + 0.3 * y1),
                    int(0.7 * prev[2] + 0.3 * x2),
                    int(0.7 * prev[3] + 0.3 * y2),
                )
                self.tracked[best_id]['box'] = new_box
                self.tracked[best_id]['missed'] = 0
                updated_ids.add(best_id)

                # if seen long enough and not classified yet -> classify now
                if 'first_seen' in self.tracked[best_id]:
                    if (current_time - self.tracked[best_id]['first_seen'] >= CLASSIFY_DELAY_SECONDS
                            and not self.tracked[best_id].get('classified', False)):
                        bx = self.tracked[best_id]['box']
                        crop = frame_rgb[bx[1]:bx[3], bx[0]:bx[2]]
                        if crop.size > 0:
                            p_fruit, p_fresh, p_ripen = classify_crop(crop)
                            # store history arrays
                            self.tracked[best_id]['pred_jenis'] = [p_fruit[0]]
                            self.tracked[best_id]['pred_kesegaran'] = [p_fresh[0]]
                            self.tracked[best_id]['pred_kematangan'] = [p_ripen[0]]
                            self.tracked[best_id]['classified'] = True
                else:
                    self.tracked[best_id]['first_seen'] = current_time

            else:
                # new object
                self.tracked[self.next_id] = {
                    'box': (x1, y1, x2, y2),
                    'pred_jenis': [],
                    'pred_kesegaran': [],
                    'pred_kematangan': [],
                    'missed': 0,
                    'first_seen': current_time,
                    'classified': False
                }
                updated_ids.add(self.next_id)
                self.next_id += 1

        # remove lost objects
        for oid in list(self.tracked.keys()):
            if oid not in updated_ids:
                self.tracked[oid]['missed'] += 1
                if self.tracked[oid]['missed'] > MAX_MISSED:
                    del self.tracked[oid]

        # draw results
        for oid, info in self.tracked.items():
            bx = info['box']
            if not info.get('classified', False):
                label = "Detecting..."
            else:
                # average history (if multiple predictions appended later)
                avg_jenis = np.mean(info['pred_jenis'], axis=0)
                avg_kesegaran = np.mean(info['pred_kesegaran'], axis=0)
                avg_kematangan = np.mean(info['pred_kematangan'], axis=0)
                jenis_label = fruit_idx2label[int(np.argmax(avg_jenis))]
                keseg_label = freshness_idx2label[int(np.argmax(avg_kesegaran))]
                kemat_label = ripeness_idx2label[int(np.argmax(avg_kematangan))]
                label = f"{jenis_label} | {keseg_label} | {kemat_label}"

            # keep history size
            if len(info['pred_jenis']) > HISTORY_SIZE:
                info['pred_jenis'].pop(0)
                info['pred_kesegaran'].pop(0)
                info['pred_kematangan'].pop(0)

            draw_label_on_image(img, bx, label)

        # show FPS small
        fps = 1.0 / max(yolo_time, 1e-6)
        cv2.putText(img, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return img

# ================= Main logic =================
if start_button:
    if source_type == "Webcam (Browser)":
        st.info("Webcam via browser (WebRTC) aktif — berikan izin kamera pada browser.")
        # Launch WebRTC streamer (non-blocking). VideoTransformer runs in background.
        webrtc_ctx = webrtc_streamer(
            key="agro-webrtc",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=RTProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        st.markdown("Tekan `Hentikan` untuk menonaktifkan WebRTC.")
        # Stop button behavior: webrtc_streamer provides stop via ctx.stop(), but we can
        # simply advise user to stop or refresh. To programmatically stop:
        if stop_button and 'webrtc' in locals():
            try:
                webrtc_ctx.stop()
            except Exception:
                pass

    elif source_type == "Upload Gambar" and uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_np = np.array(image.convert("RGB"))
        results = yolo_model(img_np, imgsz=640, conf=CONF_THRESHOLD, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            crop = img_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            p_fruit, p_fresh, p_ripen = classify_crop(crop)
            jenis_label = fruit_idx2label[np.argmax(p_fruit[0])]
            kesegaran_label = freshness_idx2label[np.argmax(p_fresh[0])]
            kematangan_label = ripeness_idx2label[np.argmax(p_ripen[0])]
            label = f"{jenis_label} | {kesegaran_label} | {kematangan_label}"

            draw_label_on_image(img_np, (x1, y1, x2, y2), label)

        # pad to 16:9 as before
        h, w, _ = img_np.shape
        aspect_ratio = w / h
        target_aspect = 16 / 9
        if aspect_ratio < target_aspect:
            new_w = int(h * target_aspect)
            pad = (new_w - w) // 2
            img_np = cv2.copyMakeBorder(img_np, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        elif aspect_ratio > target_aspect:
            new_h = int(w / target_aspect)
            pad = (new_h - h) // 2
            img_np = cv2.copyMakeBorder(img_np, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        st.image(img_np, channels="RGB", use_container_width=True)

    elif source_type == "Upload Video" and uploaded_file is not None:
        # Save temporary file
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
            current_time = time.time()

            # match & classify after delay (same logic as RTProcessor)
            for det in detections:
                x1, y1, x2, y2, conf = det
                best_i, best_id = 0, None
                for oid, info in tracked_objects.items():
                    i = iou(info['box'], (x1, y1, x2, y2))
                    if i > best_i:
                        best_i, best_id = i, oid

                if best_i > IOU_THRESHOLD:
                    prev = tracked_objects[best_id]['box']
                    tracked_objects[best_id]['box'] = (
                        int(0.7 * prev[0] + 0.3 * x1),
                        int(0.7 * prev[1] + 0.3 * y1),
                        int(0.7 * prev[2] + 0.3 * x2),
                        int(0.7 * prev[3] + 0.3 * y2),
                    )
                    tracked_objects[best_id]['missed'] = 0
                    updated_ids.add(best_id)

                    if 'first_seen' in tracked_objects[best_id]:
                        if (current_time - tracked_objects[best_id]['first_seen'] >= CLASSIFY_DELAY_SECONDS
                                and not tracked_objects[best_id].get('classified', False)):
                            bx = tracked_objects[best_id]['box']
                            crop = frame_rgb[bx[1]:bx[3], bx[0]:bx[2]]
                            if crop.size > 0:
                                p_fruit, p_fresh, p_ripen = classify_crop(crop)
                                tracked_objects[best_id]['pred_jenis'] = [p_fruit[0]]
                                tracked_objects[best_id]['pred_kesegaran'] = [p_fresh[0]]
                                tracked_objects[best_id]['pred_kematangan'] = [p_ripen[0]]
                                tracked_objects[best_id]['classified'] = True
                    else:
                        tracked_objects[best_id]['first_seen'] = current_time

                else:
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

            # remove lost
            for oid in list(tracked_objects.keys()):
                if oid not in updated_ids:
                    tracked_objects[oid]['missed'] += 1
                    if tracked_objects[oid]['missed'] > MAX_MISSED:
                        del tracked_objects[oid]

            # draw results
            for oid, info in tracked_objects.items():
                bx = info['box']
                if not info.get('classified', False):
                    label = "Detecting..."
                else:
                    avg_jenis = np.mean(info['pred_jenis'], axis=0)
                    avg_kesegaran = np.mean(info['pred_kesegaran'], axis=0)
                    avg_kematangan = np.mean(info['pred_kematangan'], axis=0)
                    jenis_label = fruit_idx2label[int(np.argmax(avg_jenis))]
                    keseg = freshness_idx2label[int(np.argmax(avg_kesegaran))]
                    kemat = ripeness_idx2label[int(np.argmax(avg_kematangan))]
                    label = f"{jenis_label} | {keseg} | {kemat}"

                draw_label_on_image(frame, bx, label)

            # fps
            fps = 1.0 / max(yolo_time, 1e-6)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # pad to 16:9 to avoid scroll
            h, w, _ = frame.shape
            aspect_ratio = w / h
            target_aspect = 16 / 9
            if aspect_ratio < target_aspect:
                new_w = int(h * target_aspect)
                pad = (new_w - w) // 2
                frame = cv2.copyMakeBorder(frame, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0,0,0))
            elif aspect_ratio > target_aspect:
                new_h = int(w / target_aspect)
                pad = (new_h - h) // 2
                frame = cv2.copyMakeBorder(frame, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))

            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()

else:
    st.info("Pilih mode input dan tekan 'Mulai Deteksi'.")


# ================= End =================
