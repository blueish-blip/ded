"""Logika setara /evaluate/batch dan /results/latest.

Format dataset (terdeteksi otomatis):
  1) Folder: <dataset>/Katarak|Diabetes|Glaukoma|Normal/*.jpg  (nama folder tidak peka huruf besar/kecil)
  2) CSV manifest dengan kolom: path,label  (path relatif terhadap lokasi CSV, atau absolut)
  3) File .zip berisi salah satu format di atas
"""
import io
import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

import config
from core.errors import DatasetError, DatasetNotConfigured, InvalidImageError, UnknownLabelError
from core.metrics import compute_metrics
from core.predict import predict_bytes
from core.preprocess import PREPROCESS_MODES

RESULT_JSON = config.RESULTS_DIR / "latest.json"
RESULT_CSV = config.RESULTS_DIR / "latest_predictions.csv"


# ------------------------------------------------------------------ deteksi dataset
def _hidden(p):
    return p.name.startswith((".", "_"))


def _label_to_index(raw):
    s = str(raw).strip()
    if s.lower() in config.NAME_TO_INDEX:
        return config.NAME_TO_INDEX[s.lower()]
    if s.isdigit() and int(s) in config.CLASS_NAMES:
        return int(s)
    raise UnknownLabelError(s)


def _subdirs(path):
    return [d for d in path.iterdir() if d.is_dir() and not _hidden(d)]


def _find_folder_root(path, depth=2):
    dirs = _subdirs(path)
    if any(d.name.lower() in config.NAME_TO_INDEX for d in dirs):
        return path
    if depth > 0 and len(dirs) == 1:  # ZIP sering punya 1 folder pembungkus
        return _find_folder_root(dirs[0], depth - 1)
    return None


def _samples_from_folder(root):
    dirs = _subdirs(root)
    unknown = [d.name for d in dirs if d.name.lower() not in config.NAME_TO_INDEX]
    if unknown:
        raise UnknownLabelError(
            f"Folder kelas tidak dikenal: {unknown[:10]}. Nama yang valid: {config.LABELS}."
        )
    samples = []

    for d in sorted(dirs, key=lambda x: x.name.lower()):
        idx = config.NAME_TO_INDEX[d.name.lower()]
        files = sorted(
            p for p in d.rglob("*")
            if p.is_file() and p.suffix.lower() in config.IMAGE_EXTS and not p.name.startswith("._")
        )
        for p in files:
            samples.append({"path": p, "label": idx, "rel": p.relative_to(root).as_posix()})
    return samples


def _samples_from_manifest(csv_path):
    try:
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    except (pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as exc:
        raise DatasetError(f"CSV manifest tidak bisa dibaca: {exc}") from exc
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"path", "label"}.issubset(df.columns):
        raise DatasetError(f"CSV manifest harus punya kolom 'path' dan 'label'. Kolom ditemukan: {list(df.columns)}")
    samples, unknown = [], set()
    for raw_path, raw_label in zip(df["path"], df["label"]):
        raw_path = str(raw_path).strip()
        if not raw_path:
            continue
        try:
            idx = _label_to_index(raw_label)
        except UnknownLabelError:
            unknown.add(str(raw_label))
            continue
        p = Path(raw_path)
        if not p.is_absolute():
            p = csv_path.parent / p

        samples.append({"path": p, "label": idx, "rel": raw_path.replace("\\", "/")})
    if unknown:
        raise UnknownLabelError(
            f"Label tidak dikenal di manifest: {sorted(unknown)[:10]}. "
            f"Yang valid: {config.LABELS} (atau angka 0-3)."
        )
    return samples


def discover_samples(path):
    """Kembalikan (samples, format, warnings). Deteksi folder vs CSV otomatis."""
    path = Path(path)
    warnings = []
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise DatasetError(f"File dataset harus .csv atau .zip, bukan '{path.suffix}'.")
        samples, fmt = _samples_from_manifest(path), "manifest"
    elif path.is_dir():
        root = _find_folder_root(path)
        if root is not None:
            samples, fmt = _samples_from_folder(root), "folder"
        else:
            csvs = sorted(path.glob("*.csv"))
            if len(csvs) == 1:
                samples, fmt = _samples_from_manifest(csvs[0]), "manifest"
            elif len(csvs) > 1:
                raise DatasetError("Ada lebih dari satu CSV di folder; arahkan DATASET_PATH ke file CSV yang dimaksud.")
            else:
                raise DatasetError(
                    f"Struktur dataset tidak dikenali. Butuh subfolder {config.LABELS} atau satu CSV manifest."
                )
    else:

        raise DatasetError(f"Path dataset tidak ditemukan: {path}")

    if not samples:
        raise DatasetError("Dataset kosong: tidak ada gambar yang ditemukan.")
    counts = {name: 0 for name in config.LABELS}
    for s in samples:
        counts[config.CLASS_NAMES[s["label"]]] += 1
    missing = [name for name, c in counts.items() if c == 0]
    if missing:
        warnings.append(f"Kelas tanpa data: {', '.join(missing)}. Metrik kelas tersebut tidak bermakna.")
    return samples, fmt, warnings


def extract_zip_safe(zip_path, dest_dir):
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                target = (dest / member.filename).resolve()
                if target != dest and dest not in target.parents:
                    raise DatasetError("ZIP tidak aman (path keluar dari folder tujuan).")
            zf.extractall(dest)
    except zipfile.BadZipFile as exc:
        raise DatasetError("File ZIP rusak atau bukan ZIP yang valid.") from exc
    return dest


# ------------------------------------------------------------------ simpan / muat hasil
def save_latest(result):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RESULT_JSON.with_suffix(".json.tmp")

    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, RESULT_JSON)
    pd.DataFrame(result["rows"]).to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")


def load_latest():
    """Setara GET /results/latest. None bila belum ada evaluasi."""
    if not RESULT_JSON.is_file():
        return None
    return json.loads(RESULT_JSON.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ evaluasi
def _save_thumb(data, index):
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((512, 512))
        name = f"{index:05d}.jpg"
        img.save(config.MISCLASSIFIED_DIR / name, quality=85)
        return name
    except Exception:
        return None


def _run(samples, fmt, mode, progress_cb, warnings, dataset_info):
    shutil.rmtree(config.MISCLASSIFIED_DIR, ignore_errors=True)
    config.MISCLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)

    rows, failed, probs, wrong_count = [], [], [], 0
    warmed = False
    total = len(samples)
    for i, s in enumerate(samples):

        try:
            data = Path(s["path"]).read_bytes()
        except OSError as exc:
            failed.append({"file": s["rel"], "reason": f"Tidak bisa membaca file: {exc}"})
            if progress_cb:
                progress_cb(i + 1, total, s["rel"], len(rows), len(failed))
            continue
        try:
            res = predict_bytes(data, mode=mode, warmup=not warmed)
            warmed = True
        except InvalidImageError as exc:
            failed.append({"file": s["rel"], "reason": str(exc)})
            if progress_cb:
                progress_cb(i + 1, total, s["rel"], len(rows), len(failed))
            continue

        correct = res["predicted_index"] == s["label"]
        row = {
            "file": s["rel"],
            "path": str(s["path"]),
            "true_label": config.CLASS_NAMES[s["label"]],
            "true_index": s["label"],
            "pred_label": res["predicted_class"],
            "pred_index": res["predicted_index"],
            "confidence": round(res["confidence"], 6),
            "correct": bool(correct),
            "latency_ms": round(res["latency_ms"], 3),
        }
        for name, p in zip(config.LABELS, res["probs_vector"]):
            row[f"prob_{name}"] = round(p, 6)
        row["thumb"] = None
        if not correct:

            wrong_count += 1
            if wrong_count <= config.MAX_MISCLASSIFIED_IMAGES:
                row["thumb"] = _save_thumb(data, wrong_count)
        rows.append(row)
        probs.append(res["probs_vector"])
        if progress_cb:
            progress_cb(i + 1, total, s["rel"], len(rows), len(failed))

    if not rows:
        reasons = "; ".join(f"{f['file']}: {f['reason']}" for f in failed[:3])
        raise DatasetError(f"Semua file gagal diproses. Contoh: {reasons}")

    metrics = compute_metrics(
        [r["true_index"] for r in rows],
        [r["pred_index"] for r in rows],
        probs,
        [r["latency_ms"] for r in rows],
    )
    warnings = list(warnings)
    if metrics["classes_without_support"]:
        warnings.append(
            "Tidak ada sampel berlabel untuk: " + ", ".join(metrics["classes_without_support"])
            + ". Rata-rata macro/weighted hanya memakai kelas yang muncul di label asli atau prediksi."
        )
    if wrong_count > config.MAX_MISCLASSIFIED_IMAGES:
        warnings.append(
            f"Galeri prediksi salah dibatasi {config.MAX_MISCLASSIFIED_IMAGES} gambar "
            f"dari {wrong_count} (tabel prediksi tetap lengkap)."
        )
    if mode != "asli_app":
        warnings.append(f"Mode preprocessing '{mode}' BUKAN preprocessing asli app.py (diagnostik).")


    support = {name: 0 for name in config.LABELS}
    for r in rows:
        support[r["true_label"]] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset_info | {"format": fmt},
        "preprocess_mode": mode,
        "model_path": str(config.MODEL_PATH),
        "labels": config.LABELS,
        "counts": {
            "found": total,
            "evaluated": len(rows),
            "failed": len(failed),
            "support": support,
            "misclassified": wrong_count,
        },
        "metrics": metrics,
        "rows": rows,
        "failed": failed,
        "warnings": warnings,
    }


def evaluate_batch(dataset_path=None, mode=None, progress_cb=None, source_name=None):
    """Evaluasi batch. progress_cb(done, total, nama_file, n_berhasil, n_gagal)."""
    raw = dataset_path if dataset_path not in (None, "") else config.DATASET_PATH
    if not str(raw).strip():
        raise DatasetNotConfigured("Batch evaluation menunggu DATASET_PATH diisi.")
    path = Path(str(raw).strip().strip('"')).expanduser()
    if not path.exists():
        raise DatasetError(f"Path dataset tidak ditemukan: {path}")


    mode = mode or config.PREPROCESS_MODE
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"Mode preprocessing '{mode}' tidak dikenal. Pilihan: {PREPROCESS_MODES}")

    work_dir = None
    try:
        if path.is_file() and path.suffix.lower() == ".zip":
            work_dir = config.WORK_DIR / uuid.uuid4().hex
            scan_target = extract_zip_safe(path, work_dir)
        else:
            scan_target = path
        samples, fmt, warnings = discover_samples(scan_target)
        info = {"path": str(path), "source_name": source_name or path.name}
        result = _run(samples, fmt, mode, progress_cb, warnings, info)
    finally:
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)

    save_latest(result)
    return result