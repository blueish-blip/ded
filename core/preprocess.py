"""Preprocessing terpusat.

Mode default "asli_app" = salinan fungsi preprocess_image() dari app.py:
    resize (224,224) -> cvtColor(BGR2HSV) -> CLAHE(clipLimit=2.3, grid 8x8) pada kanal V
    -> cvtColor(HSV2RGB) -> expand_dims, TANPA pembagian 255.

Perbedaan teknis satu-satunya: di app.py objek PIL langsung dikirim ke cv2.cvtColor,
di sini diubah dulu ke np.array (isinya sama). Hasil di-cast ke float32
(nilai uint8 tidak berubah; model.predict memang meng-cast ke float32).

Rentang nilai keluaran:
  asli_app, rgb2hsv      : float32, 0..255 (tanpa normalisasi)
  div255, resize_div255  : float32, 0..1
"""
import io

import cv2
import numpy as np
from PIL import Image

import config
from core.errors import InvalidImageError

PREPROCESS_MODES = ("asli_app", "rgb2hsv", "div255", "resize_div255")

MODE_DESCRIPTIONS = {
    "asli_app": "Persis app.py: resize 224, BGR2HSV, CLAHE (2.3, 8x8) pada V, HSV2RGB, tanpa /255. "
                "Dipakai untuk hasil resmi.",
    "rgb2hsv": "Diagnostik: seperti asli_app tetapi memakai RGB2HSV (bukan BGR2HSV).",
    "div255": "Diagnostik: asli_app lalu dibagi 255.",
    "resize_div255": "Diagnostik: hanya resize lalu dibagi 255, tanpa CLAHE (gaya percobaan TFJS lama).",
}


EXPECTED_SHAPE = (1, config.INPUT_SIZE[0], config.INPUT_SIZE[1], 3)


def decode_image(image_bytes):
    """Bytes -> PIL RGB. Melempar InvalidImageError jika kosong/bukan gambar/corrupt."""
    if not isinstance(image_bytes, (bytes, bytearray)) or len(image_bytes) == 0:
        raise InvalidImageError("File kosong atau bukan data biner gambar.")
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(
            f"File bukan gambar yang valid atau corrupt ({type(exc).__name__}: {exc})."
        ) from exc


def validate_input_array(arr):
    if arr.shape != EXPECTED_SHAPE:
        raise ValueError(f"Shape input {arr.shape}, diharapkan {EXPECTED_SHAPE}.")
    if arr.dtype != np.float32:
        raise ValueError(f"Dtype input {arr.dtype}, diharapkan float32.")
    if not np.isfinite(arr).all():
        raise ValueError("Input mengandung NaN/Inf.")


def preprocess_image(image_bytes, mode=None):
    """Satu-satunya fungsi preprocessing. Keluaran: float32 shape (1,224,224,3)."""
    mode = mode or config.PREPROCESS_MODE
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"Mode preprocessing '{mode}' tidak dikenal. Pilihan: {PREPROCESS_MODES}")

    img = decode_image(image_bytes)

    img = img.resize(config.INPUT_SIZE)          # sama dengan app.py: img.resize((224, 224))
    arr = np.array(img, dtype=np.uint8)           # array RGB (H, W, 3)

    if mode == "resize_div255":
        final = arr
    else:
        to_hsv = cv2.COLOR_RGB2HSV if mode == "rgb2hsv" else cv2.COLOR_BGR2HSV
        im = cv2.cvtColor(arr, to_hsv)
        h, s, v = cv2.split(im)
        eq = cv2.createCLAHE(clipLimit=2.3, tileGridSize=(8, 8))
        v_eq = eq.apply(v)
        im_eq = cv2.merge([h, s, v_eq])
        final = cv2.cvtColor(im_eq, cv2.COLOR_HSV2RGB)

    out = final.astype(np.float32)
    if mode in ("div255", "resize_div255"):
        out = out / 255.0
    out = np.expand_dims(out, axis=0)
    validate_input_array(out)
    return out