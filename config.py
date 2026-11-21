"""Konfigurasi terpusat. Semua nilai bisa ditimpa lewat environment variable."""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BASE_DIR = Path(__file__).resolve().parent


def _env(name, default=""):
    return os.environ.get(name, default).strip().strip('"')


# --- Model (default sesuai hasil inspeksi; bisa ditimpa env MODEL_PATH) ---
MODEL_PATH = _env("MODEL_PATH", r"E:\SKRIPSI NO FAKE FAKE\vscode\model_fixbanget.h5")

# --- Dataset uji (kosong = evaluasi batch menunggu diisi) ---
DATASET_PATH = _env("DATASET_PATH", "")

# --- Preprocessing: default = persis app.py ---
PREPROCESS_MODE = _env("PREPROCESS_MODE", "asli_app") or "asli_app"

# --- Gambar contoh (hanya DIBACA, tidak pernah ditulis) ---
SAMPLE_DIR = Path(_env("SAMPLE_DIR", r"E:\SKRIPSI NO FAKE FAKE"))

# --- Urutan kelas TETAP sesuai app.py (bukan alfabetis) ---
CLASS_NAMES = {0: "Katarak", 1: "Diabetes", 2: "Glaukoma", 3: "Normal"}
LABELS = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
NAME_TO_INDEX = {name.lower(): idx for idx, name in CLASS_NAMES.items()}

INPUT_SIZE = (224, 224)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# --- Lokasi tulis (semua di dalam folder project) ---
RESULTS_DIR = BASE_DIR / "results"
WORK_DIR = BASE_DIR / "workdir"
MISCLASSIFIED_DIR = RESULTS_DIR / "misclassified"
MAX_MISCLASSIFIED_IMAGES = 500

# Nama file contoh di root folder skripsi (label diambil dari nama file)
SAMPLE_FILES = {
    "Katarak": ["fundus katarak.jpg", "fundus katarak.jpeg", "fundus katarak.png"],
    "Diabetes": ["fundus diabetes.jpeg", "fundus diabetes.jpg", "fundus diabetes.png"],
    "Glaukoma": ["fundus glaukoma.jpg", "fundus glaukoma.jpeg", "fundus glaukoma.png"],
    "Normal": ["fundus normal.jpg", "fundus normal.jpeg", "fundus normal.png"],
}


def find_sample_images():
    """Kembalikan {label: Path atau None}. Hanya memeriksa keberadaan file."""
    found = {}
    for label, candidates in SAMPLE_FILES.items():
        found[label] = None
        for name in candidates:
            p = SAMPLE_DIR / name
            if p.is_file():
                found[label] = p
                break
    return found