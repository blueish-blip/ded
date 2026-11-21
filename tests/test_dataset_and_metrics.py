import numpy as np
import pytest
from PIL import Image

import config
from core.errors import DatasetError, UnknownLabelError
from core.evaluate import discover_samples
from core.metrics import compute_metrics


def _img(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (10, 20, 30)).save(path)


def test_folder_case_insensitive(tmp_path):
    _img(tmp_path / "katarak" / "a.jpg")
    _img(tmp_path / "NORMAL" / "b.png")
    samples, fmt, warns = discover_samples(tmp_path)
    assert fmt == "folder"
    assert {s["label"] for s in samples} == {0, 3}
    assert any("tanpa data" in w for w in warns)


def test_folder_unknown_class(tmp_path):
    _img(tmp_path / "Katarak" / "a.jpg")
    _img(tmp_path / "Anjing" / "b.jpg")
    with pytest.raises(UnknownLabelError):
        discover_samples(tmp_path)


def test_empty_dataset(tmp_path):

    (tmp_path / "Katarak").mkdir()
    with pytest.raises(DatasetError):
        discover_samples(tmp_path)


def test_manifest(tmp_path):
    _img(tmp_path / "imgs" / "a.jpg")
    _img(tmp_path / "imgs" / "b.jpg")
    csv = tmp_path / "manifest.csv"
    csv.write_text("path,label\nimgs/a.jpg,Glaukoma\nimgs/b.jpg,1\n", encoding="utf-8")
    samples, fmt, _ = discover_samples(csv)
    assert fmt == "manifest"
    assert [s["label"] for s in samples] == [2, 1]


def test_manifest_unknown_label(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text("path,label\nx.jpg,Kucing\n", encoding="utf-8")
    with pytest.raises(UnknownLabelError):
        discover_samples(csv)


def test_metrics_known_values():
    y_true = [0, 0, 1, 1, 2, 2, 3, 3]
    y_pred = [0, 1, 1, 1, 2, 3, 3, 3]
    y_prob = np.eye(4)[y_pred] * 0.9 + 0.025
    m = compute_metrics(y_true, y_pred, y_prob, [10.0] * 8)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["confusion_matrix"] == [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 1, 1], [0, 0, 0, 2]]
    assert [c["support"] for c in m["per_class"]] == [2, 2, 2, 2]
    assert all(m["roc"][n] is not None for n in config.LABELS)
    assert m["latency_ms"]["p95"] == pytest.approx(10.0)