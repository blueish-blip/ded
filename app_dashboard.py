"""Dashboard Streamlit untuk MENGUJI model klasifikasi fundus (inferensi saja, tanpa training)."""
import json
import uuid

import pandas as pd
import streamlit as st

import config
from core import plots
from core.errors import DatasetError, DatasetNotConfigured, InvalidImageError, ModelLoadError
from core.evaluate import evaluate_batch, load_latest
from core.health import health_check, model_info
from core.model_loader import get_model
from core.predict import predict_bytes
from core.preprocess import MODE_DESCRIPTIONS, PREPROCESS_MODES

st.set_page_config(page_title="Dashboard Uji Model Fundus", layout="wide")

UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Pengaturan")
default_idx = PREPROCESS_MODES.index(config.PREPROCESS_MODE) if config.PREPROCESS_MODE in PREPROCESS_MODES else 0
if config.PREPROCESS_MODE not in PREPROCESS_MODES:
    st.sidebar.error(f"PREPROCESS_MODE '{config.PREPROCESS_MODE}' tidak dikenal; memakai 'asli_app'.")
mode = st.sidebar.selectbox("Mode preprocessing", PREPROCESS_MODES, index=default_idx)
st.sidebar.caption(MODE_DESCRIPTIONS[mode])
if mode != "asli_app":
    st.sidebar.warning("Mode diagnostik, BUKAN preprocessing asli app.py. Jangan dipakai untuk melaporkan hasil resmi.")
st.sidebar.divider()
st.sidebar.caption(f"Model: {config.MODEL_PATH}")

st.sidebar.caption("Mode aplikasi: hanya inferensi (tanpa training, fine-tuning, atau augmentasi).")

st.title("Dashboard Pengujian Model Klasifikasi Fundus Mata")
st.caption("Kelas: Katarak, Diabetes, Glaukoma, Normal. Semua angka berasal dari inferensi model asli.")
if mode != "asli_app":
    st.warning(f"Mode preprocessing aktif: **{mode}** (diagnostik). Hasil tidak setara dengan app.py.")


# ---------------------------------------------------------------- model (dimuat sekali saat startup)
@st.cache_resource(show_spinner="Memuat model H5 (sekali saat startup)...")
def _load_model_once():
    return get_model()


try:
    _load_model_once()
except ModelLoadError as exc:
    st.error("Model gagal dimuat. Tidak ada fallback ke TFJS. Error apa adanya:")
    st.code(str(exc))
    with st.expander("Info lingkungan"):
        st.json(health_check())
    st.stop()


# ---------------------------------------------------------------- util tampilan
def _pct(v):
    return "-" if v is None else f"{v:.2%}"


def _num(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"



def _ms(v):
    return "-" if v is None else f"{v:.1f} ms"


def _prob_bars(ranking):
    for rank, item in enumerate(ranking, 1):
        p = min(max(item["probability"], 0.0), 1.0)
        st.progress(p, text=f"{rank}. {item['class']} - {item['probability']:.2%}")


def _render_prediction(res):
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediksi Top-1", res["predicted_class"])
    c2.metric("Confidence", f"{res['confidence']:.2%}")
    c3.metric("Waktu inferensi", _ms(res["latency_ms"]))
    st.markdown("**Probabilitas per kelas (urut dari tertinggi)**")
    _prob_bars(res["ranking"])
    st.caption(f"Mode preprocessing: {res['preprocess_mode']} | waktu preprocessing: {_ms(res['preprocess_ms'])}")


@st.dialog("Detail prediksi salah", width="large")
def _show_detail(row):
    thumb = config.MISCLASSIFIED_DIR / row["thumb"] if row.get("thumb") else None
    left, right = st.columns([1, 1])
    with left:
        if thumb is not None and thumb.is_file():
            st.image(str(thumb), use_container_width=True)
        else:
            st.info("Pratinjau tidak tersedia.")
    with right:
        st.markdown(f"**File:** `{row['file']}`")

        st.markdown(f"**Label asli:** {row['true_label']}")
        st.markdown(f"**Prediksi:** {row['pred_label']} ({row['confidence']:.2%})")
        st.markdown(f"**Latency:** {_ms(row['latency_ms'])}")
        st.caption(f"Path: {row['path']}")
        ranking = sorted(
            ({"class": n, "probability": row[f"prob_{n}"]} for n in config.LABELS),
            key=lambda d: d["probability"],
            reverse=True,
        )
        _prob_bars(ranking)


def _render_results(res):
    m, counts = res["metrics"], res["counts"]
    ds = res["dataset"]
    st.caption(
        f"Dibuat (UTC): {res['generated_at']} | Dataset: {ds['source_name']} ({ds['format']}) | "
        f"Mode preprocessing: {res['preprocess_mode']}"
    )
    for w in res.get("warnings", []):
        st.warning(w)

    st.subheader("Ringkasan metrik")
    r1 = st.columns(4)
    r1[0].metric("Accuracy", _pct(m["accuracy"]))
    r1[1].metric("Total data dievaluasi", counts["evaluated"])
    r1[2].metric("Berhasil / gagal", f"{counts['evaluated']} / {counts['failed']}")
    r1[3].metric("AUC macro (OvR)", _num(m["auc_macro"]))
    r2 = st.columns(4)
    r2[0].metric("Precision (macro)", _pct(m["macro"]["precision"]))
    r2[1].metric("Recall (macro)", _pct(m["macro"]["recall"]))
    r2[2].metric("F1 (macro)", _pct(m["macro"]["f1"]))

    r2[3].metric("MCC", _num(m["mcc"]))
    r3 = st.columns(4)
    r3[0].metric("Precision (weighted)", _pct(m["weighted"]["precision"]))
    r3[1].metric("Recall (weighted)", _pct(m["weighted"]["recall"]))
    r3[2].metric("F1 (weighted)", _pct(m["weighted"]["f1"]))
    r3[3].metric("Cohen's kappa", _num(m["kappa"]))
    r4 = st.columns(4)
    lat = m["latency_ms"]
    r4[0].metric("Latency rata-rata", _ms(lat["mean"]))
    r4[1].metric("Latency p95", _ms(lat["p95"]))
    r4[2].metric("Latency median", _ms(lat["median"]))
    r4[3].metric("Latency min / max", f"{_ms(lat['min'])} / {_ms(lat['max'])}")

    st.subheader("Confusion matrix")
    normalize = st.toggle("Normalisasi per baris (tampilkan persen)", value=False)
    fig = plots.confusion_matrix_figure(m["confusion_matrix"], res["labels"], normalize=normalize)
    cm_png = plots.fig_to_png_bytes(fig)
    st.pyplot(fig, use_container_width=False)
    plots.close(fig)
    st.download_button("Unduh confusion matrix (PNG)", cm_png, "confusion_matrix.png", "image/png")

    st.subheader("Classification report per kelas")
    report_df = pd.DataFrame(m["per_class"]).rename(
        columns={"class": "Kelas", "precision": "Precision", "recall": "Recall", "f1": "F1", "support": "Support"}
    )
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    st.subheader("ROC / AUC one-vs-rest")
    roc_fig = plots.roc_figure(m["roc"], res["labels"])
    if roc_fig is None:
        st.info("ROC tidak bisa dihitung: tiap kelas butuh sampel positif dan negatif.")
    else:

        st.pyplot(roc_fig, use_container_width=False)
        plots.close(roc_fig)
    auc_rows = [{"Kelas": n, "AUC": (m["roc"][n] or {}).get("auc")} for n in res["labels"]]
    st.dataframe(pd.DataFrame(auc_rows), hide_index=True)

    st.subheader("Galeri prediksi salah")
    wrong = [r for r in res["rows"] if not r["correct"]]
    if not wrong:
        st.success("Tidak ada prediksi salah pada data yang dievaluasi.")
    else:
        pick = st.selectbox("Filter label asli", ["Semua"] + res["labels"])
        shown = [r for r in wrong if pick == "Semua" or r["true_label"] == pick]
        per_page = 12
        pages = max(1, -(-len(shown) // per_page))
        page = st.number_input("Halaman", min_value=1, max_value=pages, value=1, step=1)
        st.caption(f"{len(shown)} prediksi salah | halaman {page} dari {pages}")
        chunk = shown[(page - 1) * per_page: page * per_page]
        for start in range(0, len(chunk), 4):
            cols = st.columns(4)
            for col, row in zip(cols, chunk[start:start + 4]):
                with col:
                    thumb = config.MISCLASSIFIED_DIR / row["thumb"] if row.get("thumb") else None
                    if thumb is not None and thumb.is_file():
                        st.image(str(thumb), use_container_width=True)
                    else:
                        st.caption("(tanpa pratinjau)")
                    st.caption(
                        f"Asli: **{row['true_label']}** | Prediksi: **{row['pred_label']}** "
                        f"({row['confidence']:.1%})\n\n`{row['file']}`"
                    )
                    if st.button("Detail", key=f"detail_{row['file']}_{row['pred_index']}_{start}_{page}"):
                        _show_detail(row)


    st.subheader("Tabel prediksi lengkap & ekspor")
    df = pd.DataFrame(res["rows"]).drop(columns=["thumb"], errors="ignore")
    st.dataframe(df, use_container_width=True, hide_index=True)
    e1, e2, e3 = st.columns(3)
    e1.download_button("Unduh prediksi (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                       "prediksi_batch.csv", "text/csv")
    e2.download_button("Unduh classification report (CSV)", report_df.to_csv(index=False).encode("utf-8-sig"),
                       "classification_report.csv", "text/csv")
    e3.download_button("Unduh hasil lengkap (JSON)",
                       json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8"),
                       "hasil_evaluasi.json", "application/json")
    if res["failed"]:
        with st.expander(f"File gagal diproses ({len(res['failed'])})"):
            st.dataframe(pd.DataFrame(res["failed"]), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- tab
tab_single, tab_batch, tab_info = st.tabs(["Prediksi satu gambar", "Evaluasi batch", "Info model"])

# ======== TAB 1: single image
with tab_single:
    st.subheader("Prediksi satu gambar")
    source = st.radio("Sumber gambar", ["Unggah file", "Gambar contoh (folder skripsi, hanya dibaca)"], horizontal=True)

    data, name, expected = None, None, None
    if source == "Unggah file":
        up = st.file_uploader("Seret & lepas atau pilih gambar fundus", type=UPLOAD_TYPES, key="single_upload")
        if up is not None:
            data, name = up.getvalue(), up.name
    else:
        available = {k: v for k, v in config.find_sample_images().items() if v is not None}

        if not available:
            st.warning(f"Tidak ada gambar contoh ditemukan di {config.SAMPLE_DIR}.")
        else:
            choice = st.selectbox("Pilih gambar contoh", list(available.keys()),
                                  format_func=lambda k: f"{available[k].name}")
            data, name, expected = available[choice].read_bytes(), available[choice].name, choice

    if data is not None:
        left, right = st.columns([1, 1])
        with left:
            st.image(data, caption=name, use_container_width=True)
            go = st.button("Prediksi", type="primary")
        file_key = (name, len(data), mode)
        if go:
            try:
                st.session_state["single"] = {"key": file_key, "res": predict_bytes(data, mode=mode)}
            except InvalidImageError as exc:
                st.session_state.pop("single", None)
                st.error(str(exc))
        saved = st.session_state.get("single")
        with right:
            if saved and saved["key"] == file_key:
                _render_prediction(saved["res"])
                if expected:
                    ok = saved["res"]["predicted_class"] == expected
                    st.caption(f"Label contoh dari nama file: {expected} -> "
                               f"{'sesuai' if ok else 'tidak sesuai'} (hanya sanity check, bukan metrik).")
            else:
                st.info("Tekan **Prediksi** untuk menjalankan model.")

# ======== TAB 2: batch
with tab_batch:

    st.subheader("Evaluasi batch")
    if not config.DATASET_PATH:
        st.info(
            "Batch evaluation menunggu DATASET_PATH diisi. Isi lewat environment variable "
            "(lihat README) lalu jalankan ulang Streamlit, atau ketik path / unggah ZIP di bawah."
        )
    st.caption("Format: folder berisi subfolder Katarak/Diabetes/Glaukoma/Normal, CSV manifest "
               "(kolom path,label), atau ZIP berisi salah satunya.")

    src = st.radio("Sumber dataset", ["Path folder / CSV", "Unggah ZIP"], horizontal=True)
    path_in, zip_up = "", None
    if src == "Path folder / CSV":
        path_in = st.text_input("Path dataset", value=config.DATASET_PATH, placeholder=r"D:\dataset_uji")
    else:
        zip_up = st.file_uploader("Unggah ZIP dataset", type=["zip"], key="batch_zip")

    c_run, c_load = st.columns([1, 1])
    run = c_run.button("Mulai evaluasi", type="primary")
    if c_load.button("Muat hasil evaluasi terakhir"):
        latest = load_latest()
        if latest is None:
            st.info("Belum ada hasil evaluasi tersimpan.")
        else:
            st.session_state["batch_result"] = latest

    if run:
        if src == "Path folder / CSV" and not path_in.strip():
            st.info("Batch evaluation menunggu DATASET_PATH diisi.")
        elif src == "Unggah ZIP" and zip_up is None:
            st.info("Pilih file ZIP terlebih dahulu.")
        else:
            bar = st.progress(0.0, text="Menyiapkan...")

            status = st.empty()

            def _cb(done, total, fname, ok, fail):
                bar.progress(done / total, text=f"{done}/{total} - {fname}")
                status.caption(f"Berhasil: {ok} | Gagal: {fail}")

            tmp_zip = None
            try:
                if zip_up is not None and src == "Unggah ZIP":
                    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
                    tmp_zip = config.WORK_DIR / f"upload_{uuid.uuid4().hex}.zip"
                    tmp_zip.write_bytes(zip_up.getbuffer())
                    target, label = tmp_zip, zip_up.name
                else:
                    target, label = path_in, None
                st.session_state["batch_result"] = evaluate_batch(
                    target, mode=mode, progress_cb=_cb, source_name=label
                )
                bar.progress(1.0, text="Selesai")
            except DatasetNotConfigured as exc:
                st.info(str(exc))
            except DatasetError as exc:
                st.error(str(exc))
            except Exception as exc:  # error tak terduga ditampilkan apa adanya
                st.error("Evaluasi berhenti karena error:")
                st.exception(exc)
            finally:
                if tmp_zip is not None:
                    tmp_zip.unlink(missing_ok=True)

    if "batch_result" in st.session_state:
        st.divider()

        _render_results(st.session_state["batch_result"])

# ======== TAB 3: info model
with tab_info:
    st.subheader("Status & info model")
    st.json(health_check())
    info = model_info()
    a, b, c, d = st.columns(4)
    a.metric("Jumlah layer", info["num_layers"])
    b.metric("Total parameter", f"{info['total_params']:,}")
    c.metric("Trainable", f"{info['trainable_params']:,}")
    d.metric("Non-trainable", f"{info['non_trainable_params']:,}")
    st.write(f"Input: `{info['input_shape']}` | Output: `{info['output_shape']}`")
    st.write("Kelas: " + ", ".join(f"{i} = {n}" for i, n in info["classes"].items()))
    st.dataframe(pd.DataFrame(info["layers"]), use_container_width=True, hide_index=True)
    with st.expander("Ringkasan arsitektur (model.summary)"):
        st.code(info["summary"])