# 🚀 Quick Start Guide - Sistem Deteksi Buah

## ⚡ Jalankan di Lokal (5 Menit)

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Jalankan Server

```bash
python app.py
```

### 3️⃣ Buka Browser

```
http://localhost:5000
```

Selesai! 🎉

---

## 🌐 Deploy Online (Render - Gratis)

### Step 1: Push ke GitHub

```bash
git init
git add .
git commit -m "Initial commit - Fruit Detection System"
git branch -M main
git remote add origin https://github.com/USERNAME/REPO-NAME.git
git push -u origin main
```

### Step 2: Deploy di Render

1. Buka https://render.com dan sign up (gratis)
2. Klik **"New +"** → **"Web Service"**
3. Connect GitHub repository Anda
4. Isi konfigurasi:

   - **Name**: fruit-detection-app
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free

5. Klik **"Create Web Service"**

⏰ **Tunggu 5-10 menit**, Render akan:

- Build aplikasi
- Install dependencies
- Deploy online
- Berikan URL publik (contoh: `https://fruit-detection-app.onrender.com`)

### Step 3: Akses Website Anda! 🎊

Website Anda sudah online dan bisa diakses dari mana saja!

---

## 🔥 Deploy Alternatif (Pilih Salah Satu)

### Railway (Sangat Mudah)

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Deploy
railway up
```

### Heroku

```bash
# Install Heroku CLI
# Download dari: https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create nama-app-anda

# Deploy
git push heroku main
```

### Google Cloud Run

```bash
# Install gcloud CLI
# Download dari: https://cloud.google.com/sdk/docs/install

# Login
gcloud auth login

# Deploy
gcloud run deploy fruit-detection \
  --source . \
  --platform managed \
  --region asia-southeast2 \
  --allow-unauthenticated
```

---

## 📱 Test Aplikasi

### Test Upload Gambar:

1. Klik "Upload Gambar"
2. Pilih foto buah
3. Lihat hasil deteksi!

### Test Kamera Real-time:

1. Klik "Kamera Real-time"
2. Klik "Mulai Kamera"
3. Izinkan akses kamera
4. Arahkan ke buah → Deteksi otomatis!

---

## ⚠️ Troubleshooting

### Error: "No module named 'cv2'"

```bash
pip install opencv-python
```

### Error: "Model not found"

Pastikan struktur folder:

```
INFERENSI/
  ├── model/
  │   ├── YOLO12n.pt
  │   └── CNN-MTL.keras
  └── app.py
```

### Kamera tidak muncul

- Gunakan HTTPS atau localhost
- Izinkan akses kamera di browser

### Deploy gagal (Out of Memory)

- Gunakan hosting dengan RAM lebih besar
- Atau optimize model size

---

## 💡 Tips

✅ **Untuk Development**: Gunakan `python app.py`
✅ **Untuk Production**: Gunakan `gunicorn app:app`
✅ **Free Hosting**: Render, Railway, PythonAnywhere
✅ **Paid Hosting (Better)**: AWS, GCP, Azure, DigitalOcean

---

## 📞 Butuh Bantuan?

Jika ada masalah, cek:

1. README.md untuk dokumentasi lengkap
2. Requirements.txt untuk dependencies
3. Console/Terminal untuk error messages

Happy Coding! 🚀🍎
