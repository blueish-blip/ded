import io

import cv2
import numpy as np
import pytest
from PIL import Image

import config
from core.errors import InvalidImageError
from core.preprocess import PREPROCESS_MODES, preprocess_image


def _png_bytes(w=300, h=260, seed=0, mode="RGB"):
    rng = np.random.default_rng(seed)
    shape = (h, w, 3) if mode == "RGB" else (h, w)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).convert(mode).save(buf, format="PNG")
    return buf.getvalue()


def _reference_app_py(image_bytes):
    """Salinan logika app.py (dengan np.array agar eksplisit)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    im = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(im)
    eq = cv2.createCLAHE(clipLimit=2.3, tileGridSize=(8, 8))
    v_eq = eq.apply(v)
    im_eq = cv2.merge([h, s, v_eq])
    im_final = cv2.cvtColor(im_eq, cv2.COLOR_HSV2RGB)
    return np.expand_dims(im_final, axis=0)



@pytest.mark.parametrize("mode", PREPROCESS_MODES)
def test_shape_dtype(mode):
    x = preprocess_image(_png_bytes(), mode)
    assert x.shape == (1, 224, 224, 3)
    assert x.dtype == np.float32
    assert np.isfinite(x).all()


@pytest.mark.parametrize("mode", PREPROCESS_MODES)
def test_deterministic(mode):
    data = _png_bytes(seed=3)
    assert np.array_equal(preprocess_image(data, mode), preprocess_image(data, mode))


def test_asli_app_matches_reference():
    data = _png_bytes(seed=5)
    ref = _reference_app_py(data).astype(np.float32)
    assert np.array_equal(preprocess_image(data, "asli_app"), ref)


def test_value_ranges():
    data = _png_bytes(seed=1)
    a = preprocess_image(data, "asli_app")
    d = preprocess_image(data, "div255")
    assert a.max() > 1.0 and a.min() >= 0.0 and a.max() <= 255.0   # tanpa /255
    assert d.min() >= 0.0 and d.max() <= 1.0


@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_various_input_modes(mode):

    x = preprocess_image(_png_bytes(mode=mode), "asli_app")
    assert x.shape == (1, 224, 224, 3)


def test_rgba_input():
    buf = io.BytesIO()
    Image.new("RGBA", (50, 40), (10, 20, 30, 128)).save(buf, format="PNG")
    assert preprocess_image(buf.getvalue(), "asli_app").shape == (1, 224, 224, 3)


def test_invalid_inputs():
    with pytest.raises(InvalidImageError):
        preprocess_image(b"")
    with pytest.raises(InvalidImageError):
        preprocess_image(b"ini bukan gambar")
    with pytest.raises(InvalidImageError):
        preprocess_image(_png_bytes()[:40])  # corrupt/terpotong


def test_unknown_mode():
    with pytest.raises(ValueError):
        preprocess_image(_png_bytes(), "ngawur")


def test_class_mapping_fixed():
    assert config.CLASS_NAMES == {0: "Katarak", 1: "Diabetes", 2: "Glaukoma", 3: "Normal"}