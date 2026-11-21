"""Memuat model H5 SEKALI (lazy singleton). Tidak ada load saat import modul.
Tidak ada fallback ke TFJS: kalau H5 gagal, errornya dilaporkan apa adanya."""
import threading
from pathlib import Path

import config
from core.errors import ModelLoadError

_lock = threading.Lock()
_model = None


def io_shapes(model):
    """(input_shape, output_shape) sebagai list; None untuk dimensi batch."""
    return list(model.inputs[0].shape), list(model.outputs[0].shape)


def get_model(force_reload=False):
    global _model
    if _model is not None and not force_reload:
        return _model
    with _lock:
        if _model is not None and not force_reload:
            return _model
        path = Path(config.MODEL_PATH)
        if not path.is_file():
            raise ModelLoadError(f"File model tidak ditemukan: {path}")
        try:
            import tensorflow as tf
        except Exception as exc:  # ImportError, DLL load failed, dll.
            raise ModelLoadError(
                f"TensorFlow tidak bisa diimpor ({type(exc).__name__}: {exc}). "

                f"Jalankan: pip install -r requirements.txt"
            ) from exc
        try:
            model = tf.keras.models.load_model(str(path), compile=False)
        except Exception as exc:
            raise ModelLoadError(
                f"Gagal memuat model H5 '{path}': {type(exc).__name__}: {exc}"
            ) from exc

        in_shape, out_shape = io_shapes(model)
        expected_in = [None, config.INPUT_SIZE[0], config.INPUT_SIZE[1], 3]
        if in_shape != expected_in:
            raise ModelLoadError(f"Input shape model {in_shape}, diharapkan {expected_in}.")
        if out_shape[-1] != len(config.LABELS):
            raise ModelLoadError(
                f"Output model {out_shape[-1]} unit, diharapkan {len(config.LABELS)} kelas."
            )
        _model = model
        return _model