"""Logika setara endpoint /predict."""
import time

import numpy as np

import config
from core.errors import ModelLoadError
from core.model_loader import get_model
from core.preprocess import preprocess_image


def predict_bytes(image_bytes, mode=None, warmup=False):
    """Prediksi satu gambar dari bytes.

    warmup=True menjalankan satu inferensi tambahan (tidak diukur, tidak dilaporkan)
    pada gambar yang SAMA agar latency pertama tidak bias oleh inisialisasi TensorFlow.
    """
    mode = mode or config.PREPROCESS_MODE
    t0 = time.perf_counter()
    x = preprocess_image(image_bytes, mode)
    preprocess_ms = (time.perf_counter() - t0) * 1000.0

    model = get_model()
    if warmup:
        model.predict(x, verbose=0)
    t1 = time.perf_counter()
    out = model.predict(x, verbose=0)
    latency_ms = (time.perf_counter() - t1) * 1000.0

    probs = np.asarray(out, dtype=np.float64)[0]
    if probs.shape != (len(config.LABELS),):
        raise ModelLoadError(f"Output model berbentuk {probs.shape}, diharapkan ({len(config.LABELS)},).")

    if not np.isfinite(probs).all():
        raise RuntimeError("Output model mengandung NaN/Inf.")

    pred_idx = int(np.argmax(probs))  # sama dengan np.argmax(model.predict(...), axis=1)[0] di app.py
    ranking = sorted(
        ({"class": config.CLASS_NAMES[i], "probability": float(p)} for i, p in enumerate(probs)),
        key=lambda d: d["probability"],
        reverse=True,
    )
    return {
        "predicted_index": pred_idx,
        "predicted_class": config.CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {config.CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
        "probs_vector": [float(p) for p in probs],
        "ranking": ranking,
        "preprocess_mode": mode,
        "latency_ms": latency_ms,
        "preprocess_ms": preprocess_ms,
    }