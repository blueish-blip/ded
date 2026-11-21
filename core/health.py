"""Logika setara endpoint /health dan /model/info (dipanggil langsung, bukan HTTP)."""
import platform
from pathlib import Path

import numpy as np

import config
from core.errors import ModelLoadError
from core.model_loader import get_model, io_shapes


def _versions():
    out = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import cv2
        out["opencv"] = cv2.__version__
    except Exception as exc:
        out["opencv"] = f"tidak tersedia ({type(exc).__name__}: {exc})"
    try:
        import tensorflow as tf
        out["tensorflow"] = tf.__version__
    except Exception as exc:
        out["tensorflow"] = f"tidak tersedia ({type(exc).__name__}: {exc})"
    try:
        import keras
        out["keras"] = keras.__version__
    except Exception as exc:
        out["keras"] = f"tidak tersedia ({type(exc).__name__}: {exc})"
    return out


def health_check():

    info = {
        "status": "error",
        **_versions(),
        "model_path": str(config.MODEL_PATH),
        "model_exists": Path(config.MODEL_PATH).is_file(),
        "model_loaded": False,
        "input_shape": None,
        "output_shape": None,
        "class_names": config.CLASS_NAMES,
        "preprocess_mode_default": config.PREPROCESS_MODE,
        "dataset_path_configured": bool(config.DATASET_PATH),
        "error": None,
    }
    try:
        model = get_model()
        in_shape, out_shape = io_shapes(model)
        info.update(status="ok", model_loaded=True, input_shape=in_shape, output_shape=out_shape)
    except ModelLoadError as exc:
        info["error"] = str(exc)
    return info


def _layer_shape(layer):
    try:
        return str(tuple(layer.output.shape))
    except Exception:
        return "-"


def model_info():
    model = get_model()
    in_shape, out_shape = io_shapes(model)

    total = int(model.count_params())
    trainable = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    lines = []

    def _collect(line, *args, **kwargs):
        lines.append(str(line))

    model.summary(print_fn=_collect)
    layers = [
        {
            "no": i,
            "nama": layer.name,
            "tipe": layer.__class__.__name__,
            "output_shape": _layer_shape(layer),
            "parameter": int(layer.count_params()),
            "trainable": bool(layer.trainable),
        }
        for i, layer in enumerate(model.layers)
    ]
    return {
        "input_shape": in_shape,
        "output_shape": out_shape,
        "num_layers": len(model.layers),
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": total - trainable,
        "classes": config.CLASS_NAMES,
        "layers": layers,
        "summary": "\n".join(lines),
    }