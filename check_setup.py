"""Cek setup sebelum memakai dashboard. Jalankan:  python check_setup.py
Hanya MEMBACA model dan gambar contoh. Tidak menulis apa pun ke folder skripsi."""
import importlib
import platform
import sys
import time
import traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OK, FAIL, WARN = "[OK]", "[GAGAL]", "[PERINGATAN]"

DEPS = [
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("PIL", "pillow"),
    ("tensorflow", "tensorflow"),
    ("keras", "keras"),
    ("sklearn", "scikit-learn"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("streamlit", "streamlit"),
]


def section(title):
    print("\n" + "=" * 72)
    print(title)

    print("=" * 72)


def main():
    # ---------------------------------------------------------- 1. versi
    section("1. VERSI PYTHON & DEPENDENCY")
    print(f"Python : {sys.version.split()[0]} ({platform.architecture()[0]})")
    print(f"Lokasi : {sys.executable}")
    missing = []
    for mod, pkg in DEPS:
        try:
            m = importlib.import_module(mod)
            print(f"{OK} {pkg:<14} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"{FAIL} {pkg:<14} BELUM TERINSTALL  ->  pip install {pkg}")
            missing.append(pkg)
        except Exception as exc:
            print(f"{FAIL} {pkg:<14} gagal diimpor: {type(exc).__name__}: {exc}")
            missing.append(pkg)
    if missing:
        print(f"\n{FAIL} Dependency bermasalah: {', '.join(missing)}")
        print("      Pastikan venv aktif lalu jalankan:  pip install -r requirements.txt")
        return 1

    import numpy as np

    import config
    from core.model_loader import get_model, io_shapes
    from core.predict import predict_bytes
    from core.preprocess import PREPROCESS_MODES, preprocess_image

    # ---------------------------------------------------------- 2. path model

    section("2. PATH MODEL & KONFIGURASI")
    mp = Path(config.MODEL_PATH)
    print(f"Path model       : {mp}")
    print(f"File ada         : {'YA' if mp.is_file() else 'TIDAK'}")
    if mp.is_file():
        print(f"Ukuran file      : {mp.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Mode preprocess  : {config.PREPROCESS_MODE} (default harus 'asli_app')")
    print(f"DATASET_PATH     : {config.DATASET_PATH or '(kosong - batch menunggu diisi)'}")
    print(f"Folder contoh    : {config.SAMPLE_DIR}")
    if not mp.is_file():
        print(f"\n{FAIL} File model tidak ditemukan. Set env MODEL_PATH atau periksa path di atas.")
        return 2

    # ---------------------------------------------------------- 3. model
    section("3. MODEL")
    try:
        t0 = time.perf_counter()
        model = get_model()
        print(f"{OK} Model dimuat dalam {time.perf_counter() - t0:.1f} detik (compile=False)")
    except Exception as exc:
        print(f"{FAIL} Model GAGAL dimuat. Error apa adanya:\n      {type(exc).__name__}: {exc}")
        print("      Tidak ada fallback ke TFJS. Paste error ini ke Claude untuk disesuaikan.")
        return 2

    in_shape, out_shape = io_shapes(model)
    total = int(model.count_params())
    trainable = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    print(f"Input shape      : {in_shape}")
    print(f"Output shape     : {out_shape}")
    print(f"Jumlah layer     : {len(model.layers)}")
    print(f"Jumlah parameter : {total:,} (trainable {trainable:,}, non-trainable {total - trainable:,})")
    last = model.layers[-1]

    units = getattr(last, "units", None)
    act = getattr(getattr(last, "activation", None), "__name__", "?")
    status = OK if units == len(config.LABELS) else FAIL
    print(f"{status} Layer output     : {last.name}, unit={units}, aktivasi={act} (diharapkan 4, softmax)")
    print("Empat layer terakhir:")
    for layer in model.layers[-4:]:
        try:
            shp = tuple(layer.output.shape)
        except Exception:
            shp = "-"
        print(f"   - {layer.name:<22} {layer.__class__.__name__:<22} {shp}")
    print("Mapping kelas (tetap, bukan alfabetis): " + ", ".join(f"{i}={n}" for i, n in config.CLASS_NAMES.items()))

    # ---------------------------------------------------------- 4. preprocess
    samples = config.find_sample_images()
    available = {k: v for k, v in samples.items() if v is not None}

    section("4. HASIL preprocess_image() UNTUK SATU GAMBAR CONTOH")
    if not available:
        print(f"{WARN} Tidak ada gambar contoh ditemukan di {config.SAMPLE_DIR}. Langkah 4-5 dilewati.")
    else:
        label0, path0 = next(iter(available.items()))
        data0 = path0.read_bytes()
        print(f"Gambar contoh    : {path0.name}")
        for mode in PREPROCESS_MODES:
            x = preprocess_image(data0, mode)
            print(f"{OK} mode={mode:<14} shape={x.shape} dtype={x.dtype} "
                  f"min={x.min():.3f} max={x.max():.3f} mean={x.mean():.3f}")
        a = preprocess_image(data0, "asli_app")
        r = preprocess_image(data0, "rgb2hsv")
        diff_rev = float(np.abs(a[..., ::-1] - r).mean())
        diff_same = float(np.abs(a - r).mean())

        print("Diagnostik BGR vs RGB (rata-rata selisih absolut piksel, skala 0..255):")
        print(f"   asli_app vs rgb2hsv, kanal sama   : {diff_same:.3f}")
        print(f"   asli_app vs rgb2hsv, kanal dibalik: {diff_rev:.3f}")
        print("   Jika selisih 'dibalik' jauh lebih kecil, hasil asli_app efektif BGR (R dan B tertukar).")

        # ------------------------------------------------------ 5. prediksi
        section("5. PREDIKSI 4 GAMBAR CONTOH (mode asli_app)")
        benar = 0
        for label, path in samples.items():
            if path is None:
                print(f"{WARN} contoh {label}: file tidak ditemukan di {config.SAMPLE_DIR}")
                continue
            try:
                res = predict_bytes(path.read_bytes(), mode="asli_app")
            except Exception:
                print(f"{FAIL} Prediksi gagal untuk {path.name}:")
                traceback.print_exc()
                return 3
            cocok = res["predicted_class"] == label
            benar += int(cocok)
            print(f"\n{path.name}  (label dari nama file: {label})")
            print(f"   prediksi : {res['predicted_class']}  confidence {res['confidence']:.4f}  "
                  f"-> {'sesuai nama file' if cocok else 'TIDAK sesuai nama file'}")
            for name in config.LABELS:
                print(f"      {name:<9} {res['probabilities'][name]:.6f}")
            print(f"   latency  : {res['latency_ms']:.1f} ms (inferensi), {res['preprocess_ms']:.1f} ms (preprocess)")
        print(f"\nRingkas: {benar} dari {len(available)} contoh sesuai nama file. "
              f"Ini sanity check, BUKAN metrik evaluasi.")

        section("6. DIAGNOSTIK: PERBANDINGAN MODE PREPROCESSING (hanya untuk dilihat)")
        print(f"{'file':<24}" + "".join(f"{m:<26}" for m in PREPROCESS_MODES))
        for label, path in available.items():

            data = path.read_bytes()
            cells = []
            for mode in PREPROCESS_MODES:
                res = predict_bytes(data, mode=mode)
                cells.append(f"{res['predicted_class']} {res['confidence']:.2f}")
            print(f"{path.name:<24}" + "".join(f"{c:<26}" for c in cells))
        print("Hanya mode 'asli_app' yang sesuai app.py. Mode lain murni untuk perbandingan.")

    print(f"\n{OK} Selesai. Jika semua bagian di atas OK, lanjut: streamlit run app_dashboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())