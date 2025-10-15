from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
from tensorflow import keras
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import base64
from PIL import Image
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Mapping labels
JENIS_BUAH = {
    0: "alpukat", 1: "durian", 2: "jeruk-siam", 3: "mangga", 4: "nanas",
    5: "nangka", 6: "pepaya", 7: "pisang", 8: "rambutan", 9: "salak"
}

KESEGARAN = {
    0: "busuk", 1: "segar", 2: "tidak-segar"
}

KEMATANGAN = {
    0: "matang", 1: "mentah", 2: "setengah-matang"
}

# Configuration - Image sizes used during training
YOLO_IMG_SIZE = 340  # YOLO trained with imgsize 340
CNN_IMG_SIZE = 256   # CNN trained with imgsize 256

# Load models
print("Loading models...")
yolo_model = YOLO('model/YOLO12n.pt')
cnn_model = keras.models.load_model('model/CNN-MTL.keras')
print("Models loaded successfully!")

def preprocess_for_cnn(image):
    """
    Preprocess cropped image for CNN classification
    
    Args:
        image: BGR image from OpenCV (cropped fruit)
    
    Returns:
        Preprocessed image array ready for CNN model
    
    Note:
        - Resize to 256x256 (same as training)
        - Normalize to [0, 1] (divide by 255.0)
        - If your CNN used different preprocessing (e.g., ImageNet normalization),
          adjust mean/std values here
    """
    # Resize to expected input size (256x256 - same as training)
    img_resized = cv2.resize(image, (CNN_IMG_SIZE, CNN_IMG_SIZE))
    
    # Convert to array and normalize to [0, 1]
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # If you used ImageNet preprocessing during training, uncomment below:
    # from tensorflow.keras.applications.imagenet_utils import preprocess_input
    # img_array = preprocess_input(img_resized)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def classify_fruit(image):
    """Classify fruit using CNN-MTL model"""
    preprocessed = preprocess_for_cnn(image)
    predictions = cnn_model.predict(preprocessed, verbose=0)
    
    # MTL model returns 3 outputs: [jenis, kesegaran, kematangan]
    jenis_pred = np.argmax(predictions[0][0])
    kesegaran_pred = np.argmax(predictions[1][0])
    kematangan_pred = np.argmax(predictions[2][0])
    
    return {
        'jenis': JENIS_BUAH[jenis_pred],
        'kesegaran': KESEGARAN[kesegaran_pred],
        'kematangan': KEMATANGAN[kematangan_pred],
        'jenis_confidence': float(np.max(predictions[0][0])),
        'kesegaran_confidence': float(np.max(predictions[1][0])),
        'kematangan_confidence': float(np.max(predictions[2][0]))
    }

def detect_and_classify(image):
    """Detect fruits using YOLO and classify each with CNN"""
    # YOLO detection with imgsize 340 (same as training)
    results = yolo_model(image, imgsz=YOLO_IMG_SIZE, verbose=False)
    
    detections = []
    annotated_image = image.copy()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Crop detected region
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size > 0:
                # Classify cropped fruit
                classification = classify_fruit(cropped)
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Prepare text
                label_text = f"{classification['jenis']}"
                detail_text1 = f"Kesegaran: {classification['kesegaran']}"
                detail_text2 = f"Kematangan: {classification['kematangan']}"
                
                # Draw background rectangles for text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                # Main label
                (w, h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
                cv2.rectangle(annotated_image, (x1, y1-h-10), (x1+w, y1), (0, 255, 0), -1)
                cv2.putText(annotated_image, label_text, (x1, y1-5), font, font_scale, (0, 0, 0), thickness)
                
                # Detail text 1
                (w1, h1), _ = cv2.getTextSize(detail_text1, font, 0.5, 1)
                cv2.rectangle(annotated_image, (x1, y1+5), (x1+w1, y1+h1+10), (255, 255, 255), -1)
                cv2.putText(annotated_image, detail_text1, (x1, y1+h1+5), font, 0.5, (0, 0, 0), 1)
                
                # Detail text 2
                (w2, h2), _ = cv2.getTextSize(detail_text2, font, 0.5, 1)
                cv2.rectangle(annotated_image, (x1, y1+h1+15), (x1+w2, y1+h1+h2+20), (255, 255, 255), -1)
                cv2.putText(annotated_image, detail_text2, (x1, y1+h1+h2+15), font, 0.5, (0, 0, 0), 1)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'classification': classification
                })
    
    return annotated_image, detections

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and process"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and decode image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Process image
        annotated_image, detections = detect_and_classify(image)
        
        # Convert to base64 for sending back
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'detections': detections,
            'total_fruits': len(detections)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route for real-time detection"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """Generate frames for video streaming"""
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Error: Could not open camera")
        return
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            # Process frame
            annotated_frame, _ = detect_and_classify(frame)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        camera.release()

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process single frame from webcam"""
    try:
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Process image
        annotated_image, detections = detect_and_classify(image)
        
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'detections': detections,
            'total_fruits': len(detections)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create upload folder if not exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
