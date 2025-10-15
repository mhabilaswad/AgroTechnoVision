# 🚀 Panduan Deploy ke Render

## ✅ Persiapan (Sudah Selesai!)

- [x] Repository GitHub: `mhabilaswad/AgroTechnoVision`
- [x] Branch: `master`
- [x] File `render.yaml` sudah ada ✓
- [x] File `requirements.txt` sudah siap ✓
- [x] File `app.py` sudah ada ✓

## 🎯 Langkah Deploy ke Render (5 Menit)

### 1️⃣ Buka Render

Buka browser dan kunjungi:

```
https://render.com
```

### 2️⃣ Sign Up / Login

- Klik **"Get Started"** atau **"Sign Up"**
- Pilih **"Sign up with GitHub"** (RECOMMENDED)
- Authorize Render untuk akses GitHub Anda

### 3️⃣ Create New Web Service

Setelah login:

1. Klik tombol **"New +"** (di kanan atas)
2. Pilih **"Web Service"**

### 4️⃣ Connect Repository

Di halaman "Create a new Web Service":

1. **Connect a repository:**

   - Cari: `AgroTechnoVision`
   - Atau ketik: `mhabilaswad/AgroTechnoVision`
   - Klik **"Connect"**

2. Jika repository tidak muncul:
   - Klik **"Configure account"**
   - Pilih repository yang ingin diakses
   - Save → Refresh halaman

### 5️⃣ Konfigurasi Web Service

Isi form dengan detail berikut:

**Basic Settings:**

```yaml
Name: fruit-detection-app
  (atau nama lain yang Anda inginkan)

Region: Singapore (atau pilih terdekat)
  - Singapore
  - Oregon
  - Frankfurt

Branch: master
  (sesuai dengan branch GitHub Anda)

Root Directory: (kosongkan)
```

**Build & Deploy:**

```yaml
Environment: Python 3

Build Command: pip install -r requirements.txt
  (otomatis terdeteksi dari render.yaml)

Start Command: gunicorn app:app
  (otomatis terdeteksi dari render.yaml)
```

**Instance Type:**

```
☑️ Free
    - 512 MB RAM
    - Shared CPU
    - 750 hours/month
```

### 6️⃣ Advanced Settings (Optional)

Klik **"Advanced"** jika ingin setting tambahan:

**Environment Variables:**

- Tidak perlu ditambahkan (sudah di render.yaml)

**Health Check Path:**

```
Path: /
```

**Auto-Deploy:**

```
☑️ Yes (Deploy otomatis saat ada push ke GitHub)
```

### 7️⃣ Create Web Service

Klik tombol besar **"Create Web Service"** di bawah.

### 8️⃣ Tunggu Deployment

Render akan:

1. ✅ Clone repository dari GitHub
2. ✅ Install dependencies dari `requirements.txt`
3. ✅ Build aplikasi
4. ✅ Start server dengan gunicorn
5. ✅ Assign URL publik

**Proses ini memakan waktu 5-10 menit.**

Anda akan melihat log seperti ini:

```
==> Cloning from https://github.com/mhabilaswad/AgroTechnoVision...
==> Running build command 'pip install -r requirements.txt'...
    Collecting flask==3.0.0
    Collecting opencv-python==4.8.1.78
    Collecting tensorflow==2.15.0
    ...
    Successfully installed flask-3.0.0 opencv-python-4.8.1.78 ...
==> Starting service with 'gunicorn app:app'...
    [2025-10-15 10:30:00] [1] [INFO] Starting gunicorn 21.2.0
    [2025-10-15 10:30:00] [1] [INFO] Listening at: http://0.0.0.0:10000
==> Your service is live at https://fruit-detection-app.onrender.com
```

### 9️⃣ Akses Website Anda! 🎉

Setelah selesai, Render akan memberikan URL seperti:

```
https://fruit-detection-app.onrender.com
```

**Buka URL tersebut di browser!**

---

## 🎬 Testing Website

### Test 1: Homepage

```
✓ Buka: https://fruit-detection-app.onrender.com
✓ Expected: Halaman dengan 2 tombol (Upload & Kamera)
```

### Test 2: Upload Gambar

```
✓ Klik "Upload Gambar"
✓ Upload foto buah
✓ Expected: Deteksi + klasifikasi muncul
```

### Test 3: Kamera Real-time

```
✓ Klik "Kamera Real-time"
✓ Klik "Mulai Kamera"
✓ Expected: Kamera aktif + deteksi live
```

---

## ⚙️ Settings Penting di Render Dashboard

Setelah deploy, Anda bisa akses dashboard:

### Logs

```
Dashboard → Your Service → Logs
- Lihat real-time logs
- Debug errors
```

### Settings

```
Dashboard → Your Service → Settings

1. Environment Variables
   - Tambah variable jika perlu

2. Build & Deploy
   - Change build command
   - Change start command

3. Health & Alerts
   - Set up email alerts
```

### Manual Deploy

```
Dashboard → Your Service → Manual Deploy
- Klik "Clear build cache & deploy"
- Gunakan jika ada masalah
```

---

## 🔄 Update Website (Auto-Deploy)

Setiap kali Anda push ke GitHub:

```bash
# Edit file (misal: app.py)
git add .
git commit -m "Update fitur baru"
git push origin master
```

**Render akan otomatis:**

1. Detect perubahan di GitHub
2. Re-build aplikasi
3. Deploy versi baru
4. Update website (tanpa downtime)

---

## ⚠️ Catatan Penting Render Free Tier

### Limitasi:

1. **Sleep Mode**: Server akan "sleep" setelah 15 menit tidak ada traffic
   - First request akan lambat (~30 detik) karena "wake up"
   - Request berikutnya normal
2. **Hours**: 750 jam gratis per bulan
   - Cukup untuk 1 website 24/7
3. **Memory**: 512MB RAM
   - Cukup untuk model YOLO + CNN
   - Jika error "Out of Memory", perlu upgrade

### Tips:

- Gunakan untuk demo/portfolio (gratis selamanya)
- Upgrade ke $7/month untuk always-on & lebih banyak RAM
- Set health check agar server tidak sleep terlalu cepat

---

## 🆘 Troubleshooting

### ❌ Error: "Build failed"

**Solusi:**

```bash
# Pastikan requirements.txt valid
cat requirements.txt

# Pastikan tidak ada typo
# Pastikan versi library kompatibel
```

### ❌ Error: "Application failed to respond"

**Solusi:**

```python
# Pastikan app.py menjalankan di port yang benar
# Render otomatis set PORT via environment variable

# Tambahkan di app.py:
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)
```

### ❌ Error: "Out of memory"

**Solusi:**

1. Upgrade ke paid plan ($7/month dengan 2GB RAM)
2. Atau optimize model (quantization/compression)

### ❌ Kamera tidak jalan

**Solusi:**

- Render otomatis pakai HTTPS ✓
- Pastikan browser izinkan akses kamera
- Test di device berbeda

---

## 📊 Alternatif Jika Render Tidak Cocok

### Railway (Lebih mudah, $5 kredit gratis)

```bash
npm i -g @railway/cli
railway login
railway up
```

### Google Cloud Run (Powerful, $300 kredit)

```bash
gcloud run deploy --source .
```

### Heroku (Bayar $5/month)

```bash
git push heroku master
```

---

## ✅ Checklist Deployment

Sebelum deploy, pastikan:

- [x] `app.py` ada dan benar
- [x] `requirements.txt` valid
- [x] `render.yaml` sudah ada
- [x] Model files (`YOLO12n.pt`, `CNN-MTL.keras`) ada di folder `model/`
- [x] Push ke GitHub sudah berhasil
- [x] Repository publik (atau Render bisa akses private repo)

---

## 🎓 Untuk Presentasi

**URL Anda setelah deploy:**

```
https://fruit-detection-app.onrender.com
```

Share URL ini untuk:

- Portfolio
- LinkedIn
- Resume/CV
- Presentasi tugas akhir
- Demo ke dosen/penguji

---

## 🎉 Selamat!

Setelah mengikuti panduan ini, website Anda akan:

- ✅ Online 24/7
- ✅ Bisa diakses dari mana saja
- ✅ Pakai HTTPS (secure)
- ✅ Gratis selamanya (dengan limitasi)
- ✅ Auto-deploy setiap push ke GitHub

**Good luck dengan deployment! 🚀**

---

## 📞 Need Help?

Jika ada error:

1. Cek **Logs** di Render Dashboard
2. Screenshot error → Google
3. Check Render documentation: https://render.com/docs
4. Render Discord community sangat helpful

Happy Deploying! 🎊
