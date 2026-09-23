"""
Aplikasi Prediksi Penyakit Daun Tomat
=====================================
Menggunakan model Dense Layer Classifier (Neural Network) yang dilatih
di atas fitur (embedding) 2048-dimensi hasil ekstraksi ResNet50
(ImageNet, global average pooling). TIDAK menggunakan StandardScaler --
model dilatih langsung di atas fitur mentah ResNet50.

Cara pakai (lokal):
    streamlit run app.py

Untuk deploy online, model TIDAK disertakan di repo GitHub (ukurannya
besar). App ini mengunduh model dari Hugging Face saat pertama kali
dijalankan. Isi MODEL_URL dan CLASS_NAMES_URL di bawah dengan link
file kamu di Hugging Face.
"""

import pickle
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Ganti dengan URL "resolve/main" ke masing-masing file kamu di Hugging Face
MODEL_URL = "https://huggingface.co/datasets/rinaldi2211/tomato-disease-model/blob/main/dense_classifier_model.keras"
CLASS_NAMES_URL = "https://huggingface.co/datasets/rinaldi2211/tomato-disease-model/blob/main/dense_class_names.pkl"

MODEL_PATH = Path(__file__).parent / "dense_classifier_model.keras"
CLASS_NAMES_PATH = Path(__file__).parent / "dense_class_names.pkl"

IMG_SIZE = (224, 224)


def _download(url: str, dest: Path, min_size_bytes: int = 1024):
    """Unduh satu file secara streaming, validasi ukurannya, hapus kalau gagal."""
    if dest.exists() and dest.stat().st_size >= min_size_bytes:
        return
    if not url or url.startswith("GANTI_DENGAN"):
        st.error(f"URL belum diisi di app.py untuk file: {dest.name}")
        st.stop()
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
        if tmp_path.stat().st_size < min_size_bytes:
            tmp_path.unlink(missing_ok=True)
            st.error(f"Unduhan {dest.name} tidak lengkap. Coba Rerun atau reboot app.")
            st.stop()
        tmp_path.rename(dest)
    except requests.RequestException as e:
        st.error(f"Gagal mengunduh {dest.name}: {e}")
        st.stop()


def ensure_artifacts_downloaded():
    with st.spinner("Mengunduh model (hanya sekali di awal)..."):
        _download(MODEL_URL, MODEL_PATH, min_size_bytes=1024 * 1024)  # model biasanya >1MB
        _download(CLASS_NAMES_URL, CLASS_NAMES_PATH, min_size_bytes=16)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "rb") as f:
        class_names = pickle.load(f)
    return model, class_names


@st.cache_resource(show_spinner=False)
def load_feature_extractor():
    return ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))


def extract_features(image: Image.Image, extractor) -> np.ndarray:
    """HARUS identik dengan preprocessing saat ekstraksi features.npy (load_img default = NEAREST)."""
    image = image.convert("RGB").resize(IMG_SIZE, resample=Image.NEAREST)
    arr = img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return extractor.predict(arr, verbose=0)


def predict(image: Image.Image, model, class_names, extractor):
    features = extract_features(image, extractor)  # TIDAK di-scaling, sesuai training

    proba = model.predict(features, verbose=0)[0]  # softmax output, shape (num_classes,)
    pred_idx = int(np.argmax(proba))

    pred_label = class_names[pred_idx]
    prob_dict = {class_names[i]: float(proba[i]) for i in range(len(class_names))}
    return pred_label, prob_dict


def main():
    st.set_page_config(page_title="Prediksi Penyakit Daun Tomat", page_icon="🍅", layout="centered")
    st.title("🍅 Prediksi Penyakit Daun Tomat")
    st.caption("Model: ResNet-50 (ekstraksi fitur) + Dense Layer Classifier")

    ensure_artifacts_downloaded()
    model, class_names = load_artifacts()

    uploaded_file = st.file_uploader("Unggah gambar daun tomat (jpg/png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        with st.spinner("Memuat model ekstraksi fitur (ResNet50)..."):
            extractor = load_feature_extractor()

        with st.spinner("Memproses prediksi..."):
            pred_label, prob_dict = predict(image, model, class_names, extractor)

        confidence = prob_dict[pred_label]
        st.subheader("Hasil Prediksi")
        if pred_label == "Tomato___healthy":
            st.success(f"✅ **{pred_label}** (keyakinan {confidence:.1%})")
        else:
            st.warning(f"⚠️ **{pred_label}** (keyakinan {confidence:.1%})")

        st.write("**Rincian probabilitas per kelas:**")
        for label, p in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
            st.write(label)
            st.progress(min(max(p, 0.0), 1.0), text=f"{p:.1%}")


if __name__ == "__main__":
    main()
