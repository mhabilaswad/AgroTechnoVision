# ⚙️ Konfigurasi Model - PENTING!

## 📊 Image Size Configuration

Aplikasi ini sudah dikonfigurasi sesuai dengan training:

### YOLO Model
- **Training Image Size**: 340x340
- **Konfigurasi di app.py**: `YOLO_IMG_SIZE = 340`
- **Usage**: `yolo_model(image, imgsz=340)`

### CNN Multi-Task Learning Model
- **Training Image Size**: 256x256
- **Konfigurasi di app.py**: `CNN_IMG_SIZE = 256`
- **Preprocessing**: `cv2.resize(image, (256, 256))`

---

## 🔧 Preprocessing CNN

Saat ini menggunakan normalisasi sederhana:
```python
img_array = np.array(img_resized) / 255.0  # [0, 1]
```

### ⚠️ Jika CNN Anda Menggunakan Preprocessing Berbeda:

#### 1. ImageNet Preprocessing (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
Jika Anda pakai transfer learning dari VGG16/ResNet/MobileNet, uncomment di `app.py`:
```python
from tensorflow.keras.applications.imagenet_utils import preprocess_input
img_array = preprocess_input(img_resized, mode='tf')  # atau mode='caffe'
```

#### 2. Custom Normalization
Jika training pakai normalization berbeda:
```python
# Contoh: normalize to [-1, 1]
img_array = (img_array - 0.5) * 2.0

# Atau dengan mean/std custom
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
img_array = (img_array - mean) / std
```

#### 3. No Normalization
Jika training tanpa normalisasi:
```python
img_array = np.array(img_resized, dtype=np.float32)  # [0, 255]
```

---

## 🧪 Cara Verifikasi Preprocessing Benar

### Test 1: Check Input Shape
```python
print(f"CNN expects: {cnn_model.input_shape}")
# Expected: (None, 256, 256, 3)
```

### Test 2: Check Input Range
```python
print(f"Min: {img_array.min()}, Max: {img_array.max()}")
# Should match training range (e.g., [0, 1] or [-1, 1])
```

### Test 3: Test Prediction
```python
# Upload gambar buah yang jelas
# Jika hasil klasifikasi buruk, cek preprocessing!
```

---

## 📋 Checklist Training vs Inference

Pastikan **SAMA PERSIS** antara training dan inference:

### ✅ Image Size
- [x] YOLO: 340x340
- [x] CNN: 256x256

### ⚠️ Preprocessing (VERIFY!)
- [ ] Normalization method?
  - Simple [0, 1]? ✅
  - ImageNet? 
  - Custom mean/std?
- [ ] Color space?
  - RGB ✅ (OpenCV BGR → RGB jika perlu)
  - BGR
  - Grayscale

### ✅ Model Input
- [ ] Input shape: (batch, 256, 256, 3)
- [ ] Data type: float32
- [ ] Channel order: RGB or BGR?

---

## 🔄 OpenCV BGR vs RGB Issue

**PENTING!** OpenCV load image sebagai **BGR**, tapi kebanyakan model CNN expect **RGB**.

### Jika CNN Anda Trained dengan RGB (Keras/TensorFlow default):

Tambahkan konversi di `preprocess_for_cnn`:
```python
# Convert BGR to RGB
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (CNN_IMG_SIZE, CNN_IMG_SIZE))
```

### Cara Cek Training Anda Pakai RGB atau BGR:

1. **Jika pakai `ImageDataGenerator` atau `tf.keras.preprocessing.image.load_img`**: RGB ✅
2. **Jika pakai `cv2.imread` langsung**: BGR
3. **Jika pakai PIL/Pillow**: RGB ✅

---

## 🎯 Konfigurasi Lengkap di app.py

```python
# Configuration
YOLO_IMG_SIZE = 340  # YOLO training size
CNN_IMG_SIZE = 256   # CNN training size

# Color space (pilih salah satu)
USE_RGB = True  # Set True jika CNN trained dengan RGB
                # Set False jika trained dengan BGR

def preprocess_for_cnn(image):
    # Convert BGR to RGB if needed
    if USE_RGB:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    img_resized = cv2.resize(image, (CNN_IMG_SIZE, CNN_IMG_SIZE))
    
    # Normalize
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
```

---

## 🚨 Troubleshooting

### Problem: Deteksi YOLO bagus, tapi klasifikasi CNN buruk

**Kemungkinan:**
1. ❌ Image size salah (bukan 256x256)
2. ❌ Normalization salah
3. ❌ BGR vs RGB issue
4. ❌ Model incompatible dengan TensorFlow versi baru

**Solution:**
1. ✅ Pastikan `CNN_IMG_SIZE = 256`
2. ✅ Match preprocessing dengan training
3. ✅ Convert BGR to RGB
4. ✅ Test model di local dulu

### Problem: YOLO tidak detect apa-apa

**Kemungkinan:**
1. ❌ Image size salah (bukan 340)
2. ❌ Confidence threshold terlalu tinggi
3. ❌ Model path salah

**Solution:**
1. ✅ Set `imgsz=340`
2. ✅ Lower confidence: `yolo_model(image, imgsz=340, conf=0.25)`
3. ✅ Check model path

---

## 📝 Summary

**Sudah Dikonfigurasi:**
- ✅ YOLO image size: 340
- ✅ CNN image size: 256
- ✅ Basic normalization: [0, 1]

**Perlu Anda Verifikasi:**
- ⚠️ Apakah CNN training pakai RGB atau BGR?
- ⚠️ Apakah normalization [0, 1] sudah benar?
- ⚠️ Apakah pakai ImageNet preprocessing?

**Jika ada yang berbeda dengan training, edit `preprocess_for_cnn` di `app.py`!**

---

## 🎓 Best Practice

Simpan preprocessing config saat training:
```python
# Saat training, save config
config = {
    'img_size': 256,
    'normalization': 'minmax',  # [0, 1]
    'color_space': 'RGB',
    'preprocessing': 'simple'
}
np.save('preprocessing_config.npy', config)
```

Lalu load saat inference untuk konsistensi! ✨
