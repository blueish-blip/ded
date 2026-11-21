import pytest

import config
from core.health import health_check
from core.predict import predict_bytes


def test_health_ok():
    info = health_check()
    assert info["status"] == "ok", info.get("error")
    assert info["model_loaded"] is True
    assert info["input_shape"][1:] == [224, 224, 3]
    assert info["output_shape"][-1] == 4


def test_predict_one_sample():
    samples = {k: v for k, v in config.find_sample_images().items() if v is not None}
    if not samples:
        pytest.skip(f"Tidak ada gambar contoh di {config.SAMPLE_DIR}")
    _, path = next(iter(samples.items()))
    res = predict_bytes(path.read_bytes())
    assert res["predicted_class"] in config.LABELS
    assert len(res["probabilities"]) == 4
    assert sum(res["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert res["latency_ms"] > 0