# Dashboard Pengujian Model Fundus Mata

Dashboard Streamlit untuk **menguji/mengevaluasi** model klasifikasi citra fundus (Katarak, Diabetes, Glaukoma, Normal).
Hanya inferensi: **tidak ada training, fine-tuning, atau augmentasi.** Semua metrik dihitung dari model asli.

> **PERINGATAN:** project ini hanya MEMBACA `E:\SKRIPSI NO FAKE FAKE` (model dan gambar contoh).
> Jangan mengubah, menimpa, atau menghapus apa pun di sana. Semua tulisan (hasil, thumbnail, file sementara)
> masuk ke folder project ini (`results\`, `workdir\`). File kredensial apa pun tidak dibaca oleh project ini.

## Struktur folder

```
Tes Skripsi\
  app_dashboard.py      Dashboard Streamlit (UI bahasa Indonesia)
  check_setup.py        Pengecekan setup (jalankan dulu)
  config.py             Path, mapping kelas, env var
  core\
    preprocess.py       Satu fungsi preprocessing terpusat (persis app.py)
    model_loader.py     Load model H5 sekali (lazy singleton), tanpa fallback TFJS
    health.py           health_check(), model_info()      (setara /health, /model/info)
    predict.py          predict_bytes(image_bytes)         (setara /predict)
    evaluate.py         evaluate_batch(), load_latest()    (setara /evaluate/batch, /results/latest)
    metrics.py          Accuracy, P/R/F1, MCC, kappa, CM, ROC/AUC, latency
    plots.py            Confusion matrix, ROC, ekspor PNG
    errors.py           Exception khusus
  tests\                Unit test + smoke test
  results\              (otomatis) latest.json, latest_predictions.csv, thumbnail prediksi salah
  workdir\              (otomatis) ekstraksi ZIP sementara, dihapus setelah evaluasi
```

Tidak ada server HTTP: logika endpoint dipanggil langsung sebagai fungsi Python dari dashboard dan `check_setup.py`.
Kalau nanti butuh API, fungsi-fungsi di `core\` bisa dibungkus FastAPI tanpa mengubah logikanya.


## Instalasi dan menjalankan (Windows)

```
cd "D:\FreeLLM\Open Code\Project\Tes Skripsi"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python check_setup.py
streamlit run app_dashboard.py
```

Buka http://localhost:8501. Jika Streamlit meminta email saat pertama kali jalan, tekan Enter.
Jika PowerShell menolak `activate`, pakai Command Prompt (cmd) atau `.venv\Scripts\activate.bat`.

Jalankan test: `python -m pytest -q`

## Environment variable (semua opsional)

| Variabel | Default | Fungsi |
|---|---|---|
| `MODEL_PATH` | `E:\SKRIPSI NO FAKE FAKE\vscode\model_fixbanget.h5` | Lokasi model H5 |
| `DATASET_PATH` | kosong | Folder/CSV/ZIP dataset uji untuk batch |
| `PREPROCESS_MODE` | `asli_app` | Mode preprocessing awal |
| `SAMPLE_DIR` | `E:\SKRIPSI NO FAKE FAKE` | Lokasi 4 gambar contoh (hanya dibaca) |

Contoh mengisi `DATASET_PATH` tanpa mengedit kode:


```
:: Command Prompt
set DATASET_PATH=D:\dataset_uji
streamlit run app_dashboard.py

# PowerShell
$env:DATASET_PATH="D:\dataset_uji"
streamlit run app_dashboard.py
```

Selama `DATASET_PATH` kosong, tab batch menampilkan "Batch evaluation menunggu DATASET_PATH diisi."
Path juga bisa diketik di tab batch, atau dataset diunggah sebagai ZIP.

## Format dataset (deteksi otomatis)

**1. Folder per kelas** (nama folder tidak peka huruf besar/kecil, urutan kelas tidak dipengaruhi nama):

```
dataset_uji\
  Katarak\*.jpg
  Diabetes\*.jpg
  Glaukoma\*.jpg
  Normal\*.jpg
```

Folder lain di level yang sama dianggap label tidak dikenal dan evaluasi dihentikan dengan pesan jelas.

**2. CSV manifest** dengan kolom `path,label`. Path relatif dihitung dari lokasi CSV, atau path absolut.
Label berupa nama kelas atau angka 0 sampai 3 (0=Katarak, 1=Diabetes, 2=Glaukoma, 3=Normal).

```
path,label
imgs/a001.jpg,Katarak
imgs/b014.jpg,Normal
```

**3. ZIP** yang berisi salah satu format di atas (boleh satu folder pembungkus).

## Cara membaca metrik

- **Accuracy**: proporsi prediksi benar dari semua data.
- **Precision / Recall / F1 macro**: rata-rata per kelas tanpa bobot. Cocok bila kelas tidak seimbang. **Weighted** diberi bobot jumlah data tiap kelas.
- **MCC** (-1..1) dan **Cohen's kappa** (-1..1): ukuran kesepakatan yang lebih tahan terhadap ketidakseimbangan kelas. 0 berarti setara tebakan.
- **Confusion matrix**: baris = label asli, kolom = prediksi. Diagonal = benar.
- **ROC/AUC OvR**: tiap kelas melawan semua kelas lain memakai probabilitas softmax. Tidak terdefinisi bila kelas itu tidak punya sampel positif atau negatif.
- **Latency**: waktu `model.predict` per gambar (tanpa preprocessing, satu gambar per panggilan seperti app.py). Satu inferensi pemanasan pada gambar pertama dibuang dan tidak dilaporkan. p95 = 95% gambar lebih cepat dari angka ini.
- Rata-rata macro/weighted hanya memakai kelas yang muncul di label asli atau prediksi (perilaku default scikit-learn). Kelas tanpa data diberi peringatan.
- Gambar yang gagal dibaca dicatat di daftar "File gagal diproses" dan tidak dihitung dalam metrik.

## Preprocessing: catatan inkonsistensi

- `PREPROCESS_MODE=asli_app` (default) = logika `app.py`: resize 224x224, `cv2.COLOR_BGR2HSV` pada array dari PIL (yang RGB), CLAHE (clipLimit 2.3, grid 8x8) pada kanal V, `HSV2RGB`, **tanpa pembagian 255**.
  Karena PIL menghasilkan RGB tetapi dipanggil dengan `BGR2HSV`, hasil akhirnya kemungkinan besar efektif BGR (kanal R dan B tertukar). Ini **tidak diperbaiki** karena model mungkin dilatih dengan perilaku itu.
  `check_setup.py` mencetak diagnostik selisih untuk memeriksa hal ini.
- Mode diagnostik (hanya untuk perbandingan, bukan hasil resmi): `rgb2hsv`, `div255`, `resize_div255`.
- `DED.html` (percobaan TFJS lama) memakai resize + /255, **tanpa CLAHE**, dan logika "dog"/"breedNames". Bukan acuan preprocessing.
- `backend.py` memakai path lama `C:\Users\hhfzsn\Downloads\...` yang sudah tidak valid. Tidak dipakai.
- Versi TFJS model (`model_fix.tfjs`) tidak dipakai. Jika H5 gagal dimuat, error ditampilkan apa adanya tanpa fallback.

## Troubleshooting

- **Model gagal dimuat**: salin pesan error lengkap dari `check_setup.py`. Model dibuat dengan Keras 3.5.0, jadi `requirements.txt` memakai TensorFlow yang membawa Keras 3.
- **Konflik versi saat pip install**: pastikan memakai venv baru ini (Python 3.10, 64-bit), bukan venv lama di folder skripsi.
- **Upload ZIP besar**: batas unggah diatur 2048 MB di `.streamlit\config.toml`. Untuk dataset sangat besar, lebih mudah memakai `DATASET_PATH` ke folder langsung.