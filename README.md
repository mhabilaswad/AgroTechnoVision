# 🍎 Sistem Deteksi & Klasifikasi Buah

Website AI untuk mendeteksi dan mengklasifikasi buah menggunakan YOLO v12n dan CNN Multi-Task Learning.

## 📋 Fitur

- ✅ **Upload Gambar**: Upload gambar buah untuk dideteksi dan diklasifikasi
- ✅ **Kamera Real-time**: Deteksi dan klasifikasi buah secara live menggunakan kamera
- ✅ **Multi-Task Classification**: Klasifikasi jenis, kesegaran, dan kematangan buah
- ✅ **Interface Modern**: UI responsif dan user-friendly

## 🎯 Model yang Digunakan

### YOLO v12n (`model/YOLO12n.pt`)
- Mendeteksi lokasi buah dalam gambar
- Semua deteksi berlabel 0 (object detection saja)

### CNN Multi-Task Learning (`model/CNN-MTL.keras`)
Mengklasifikasi 3 aspek buah:

**1. Jenis Buah (10 kelas):**
- 0: alpukat
- 1: durian
- 2: jeruk-siam
- 3: mangga
- 4: nanas
- 5: nangka
- 6: pepaya
- 7: pisang
- 8: rambutan
- 9: salak

**2. Kesegaran (3 kelas):**
- 0: busuk
- 1: segar
- 2: tidak-segar

**3. Kematangan (3 kelas):**
- 0: matang
- 1: mentah
- 2: setengah-matang

## 📁 Struktur Proyek

```
INFERENSI/
│
├── model/
│   ├── YOLO12n.pt          # Model YOLO untuk deteksi
│   └── CNN-MTL.keras       # Model CNN untuk klasifikasi
│
├── templates/
│   └── index.html          # Frontend HTML
│
├── static/                 # File statis (jika ada)
├── uploads/                # Folder untuk gambar yang diupload
│
├── app.py                  # Backend Flask
├── requirements.txt        # Dependencies
└── README.md              # Dokumentasi ini
```

## 🚀 Cara Menjalankan di Lokal

### 1. Instalasi Dependencies

Pastikan Python 3.8+ sudah terinstal, lalu jalankan:

```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi

```bash
python app.py
```

### 3. Akses Website

Buka browser dan akses:
```
http://localhost:5000
```

atau jika ingin diakses dari device lain di jaringan yang sama:
```
http://<IP_KOMPUTER_ANDA>:5000
```

## 💻 Cara Menggunakan

### Mode Upload Gambar

1. Klik tombol **"Upload Gambar"**
2. Pilih atau drag-drop gambar buah
3. Tunggu proses deteksi dan klasifikasi
4. Lihat hasil:
   - Gambar dengan bounding box dan label
   - Detail klasifikasi setiap buah
   - Total jumlah buah terdeteksi

### Mode Kamera Real-time

1. Klik tombol **"Kamera Real-time"**
2. Klik **"Mulai Kamera"**
3. Izinkan akses kamera
4. Arahkan kamera ke buah
5. Sistem akan mendeteksi dan mengklasifikasi secara otomatis
6. Klik **"Stop Kamera"** untuk berhenti

## 🌐 Cara Deploy Online

### Option 1: Render (Recommended - FREE)

1. **Buat akun di Render**: https://render.com

2. **Buat file `render.yaml`** di root project:
```yaml
services:
  - type: web
    name: fruit-detection-app
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
```

3. **Tambahkan gunicorn ke requirements.txt**:
```bash
echo "gunicorn==21.2.0" >> requirements.txt
```

4. **Push ke GitHub**:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <URL_REPO_ANDA>
git push -u origin main
```

5. **Connect ke Render**:
   - Login ke Render
   - New → Web Service
   - Connect GitHub repository
   - Deploy!

### Option 2: Railway (FREE dengan limit)

1. **Buat akun di Railway**: https://railway.app
2. **Buat Procfile** di root project:
```
web: gunicorn app:app
```
3. **Push ke GitHub**
4. **Connect repository di Railway**
5. **Deploy otomatis**

### Option 3: Heroku

1. **Install Heroku CLI**
2. **Login**: `heroku login`
3. **Buat app**: `heroku create nama-app-anda`
4. **Tambahkan Procfile**:
```
web: gunicorn app:app
```
5. **Push ke Heroku**:
```bash
git push heroku main
```

### Option 4: PythonAnywhere (FREE)

1. **Buat akun**: https://www.pythonanywhere.com
2. **Upload files** via Web interface
3. **Install dependencies** di console
4. **Configure WSGI file**
5. **Reload web app**

### Option 5: Google Cloud Run

1. **Buat Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
```

2. **Deploy**:
```bash
gcloud run deploy fruit-detection --source . --platform managed --region asia-southeast2 --allow-unauthenticated
```

## ⚠️ Catatan Penting untuk Deployment

1. **Model Size**: Model YOLO dan CNN cukup besar, pastikan hosting yang dipilih support file size tersebut

2. **Memory**: Inference model membutuhkan memory yang cukup. Minimal 512MB RAM

3. **Camera Access**: Untuk mode kamera, pastikan website menggunakan HTTPS (required oleh browser modern)

4. **Environment Variables**: Jika perlu, set:
   - `FLASK_ENV=production`
   - `SECRET_KEY=<random-string>`

5. **Gunicorn Config**: Untuk production, gunakan:
```bash
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 app:app
```

## 🔧 Troubleshooting

### Kamera tidak muncul
- Pastikan browser mengizinkan akses kamera
- Website harus diakses via HTTPS atau localhost
- Pastikan tidak ada aplikasi lain yang menggunakan kamera

### Error saat load model
- Pastikan path model benar: `model/YOLO12n.pt` dan `model/CNN-MTL.keras`
- Cek kompatibilitas versi TensorFlow dan Ultralytics

### Memory Error
- Kurangi ukuran gambar input
- Reduce batch size atau gunakan model yang lebih kecil

### Deteksi lambat
- Gunakan GPU jika tersedia
- Optimize image preprocessing
- Reduce video frame rate untuk mode kamera

## 📦 Dependencies

- **Flask**: Web framework
- **OpenCV**: Image processing
- **TensorFlow**: Deep learning framework untuk CNN
- **Ultralytics**: YOLO implementation
- **NumPy**: Array operations
- **Pillow**: Image handling

## 📝 Lisensi

Project ini dibuat untuk tugas akhir.

## 👨‍💻 Developer

Dibuat dengan ❤️ oleh Machine Learning Engineer

---

## 🎓 Tips Deployment Production

1. **Use CDN** untuk static files
2. **Enable caching** untuk hasil prediksi
3. **Add rate limiting** untuk mencegah abuse
4. **Monitor performance** dengan tools seperti New Relic
5. **Use load balancer** jika traffic tinggi
6. **Implement logging** untuk debugging
7. **Add authentication** jika diperlukan
8. **Backup models** secara regular

Happy Detecting! 🍎🍌🍍
