"""
Aplikasi Prediksi Penyakit Daun Tomat
=====================================
Menggunakan model Random Forest yang dilatih di atas fitur (embedding)
2048-dimensi hasil ekstraksi ResNet50 (ImageNet, global average pooling).

Cara pakai (lokal):
    streamlit run app.py

Untuk deploy online (Streamlit Community Cloud dsb.), file model TIDAK
disertakan di repo (ukurannya >100MB, melebihi batas GitHub). Sebagai
gantinya, app ini akan mengunduh model dari Hugging Face saat pertama kali
dijalankan. Isi MODEL_URL di bawah dengan link file model kamu di Hugging Face
(lihat README.md untuk cara upload-nya).
"""

import pickle
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image

# TensorFlow / Keras hanya di-import saat dibutuhkan (feature extractor)
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Ganti dengan URL "resolve/main" ke file model kamu di Hugging Face,
# contoh: "https://huggingface.co/datasets/USERNAME/REPO/resolve/main/tomato_disease_rf_model.pkl"
MODEL_URL = "https://huggingface.co/datasets/rinaldi2211/tomato-disease-model/resolve/main/tomato_disease_rf_model.pkl?download=true"

MODEL_PATH = Path(__file__).parent / "tomato_disease_rf_model.pkl"
IMG_SIZE = (224, 224)  # ukuran input standar ResNet50


def ensure_model_downloaded():
    """Unduh file model dari Hugging Face jika belum ada di disk lokal.

    Mengunduh secara streaming (chunk demi chunk) dan memvalidasi ukuran file
    setelah selesai, supaya kalau koneksi putus di tengah jalan, file yang
    tidak lengkap dihapus dan bisa dicoba ulang (bukan malah dipakai dan
    menyebabkan UnpicklingError saat pickle.load).
    """
    MIN_EXPECTED_SIZE_BYTES = 100 * 1024 * 1024  # model asli ~186MB

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= MIN_EXPECTED_SIZE_BYTES:
        return

    if not MODEL_URL or MODEL_URL.startswith("GANTI_DENGAN"):
        st.error(
            "MODEL_URL belum diisi di app.py. Upload model ke Hugging Face "
            "dulu, lalu tempel link 'resolve/main'-nya ke variabel MODEL_URL."
        )
        st.stop()

    with st.spinner("Mengunduh model (hanya sekali di awal, ~186MB)..."):
        try:
            tmp_path = MODEL_PATH.with_suffix(".tmp")
            with requests.get(MODEL_URL, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)

            downloaded_size = tmp_path.stat().st_size
            if downloaded_size < MIN_EXPECTED_SIZE_BYTES:
                tmp_path.unlink(missing_ok=True)
                st.error(
                    f"Unduhan model tidak lengkap (hanya {downloaded_size / 1e6:.1f}MB, "
                    f"seharusnya ~186MB). Coba klik tombol 'Rerun' di kanan atas, "
                    "atau reboot app dari 'Manage app'."
                )
                st.stop()

            tmp_path.rename(MODEL_PATH)
        except requests.RequestException as e:
            st.error(f"Gagal mengunduh model dari Hugging Face: {e}")
            st.stop()

# Label deskripsi singkat untuk tiap kelas (opsional, mempercantik tampilan)
LABEL_INFO = {
    "Tomato___Bacterial_spot": "Bercak Bakteri",
    "Tomato___Early_blight": "Bercak Kering (Early Blight)",
    "Tomato___Late_blight": "Busuk Daun (Late Blight)",
    "Tomato___Leaf_Mold": "Jamur Daun (Leaf Mold)",
    "Tomato___Septoria_leaf_spot": "Bercak Septoria",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tungau Laba-laba",
    "Tomato___Target_Spot": "Bercak Target (Target Spot)",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Virus Keriting Kuning Daun (TYLCV)",
    "Tomato___Tomato_mosaic_virus": "Virus Mosaik Tomat",
    "Tomato___healthy": "Sehat",
}


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model RF + scaler + label encoder + class names dari pickle."""
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data


@st.cache_resource(show_spinner=False)
def load_feature_extractor():
    """Load ResNet50 pretrained (tanpa top layer) sebagai feature extractor."""
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",  # -> output vektor 2048 dimensi
        input_shape=(224, 224, 3),
    )
    return base_model


def extract_features(image: Image.Image, extractor) -> np.ndarray:
    """Ubah gambar PIL menjadi vektor fitur 2048-dim via ResNet50."""
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    features = extractor.predict(arr, verbose=0)
    return features  # shape (1, 2048)


def predict(image: Image.Image, artifacts, extractor):
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    class_names = artifacts["class_names"]

    features = extract_features(image, extractor)
    features_scaled = scaler.transform(features)

    pred_idx = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]

    pred_label = class_names[pred_idx]
    prob_dict = {class_names[i]: float(proba[i]) for i in range(len(class_names))}
    return pred_label, prob_dict


def main():
    st.set_page_config(
        page_title="Prediksi Penyakit Daun Tomat",
        page_icon="🍅",
        layout="centered",
    )

    st.title("🍅 Prediksi Penyakit Daun Tomat")
    st.caption(
        "Unggah foto daun tomat, model Random Forest (berbasis fitur ResNet50) "
        "akan memprediksi jenis penyakitnya."
    )

    ensure_model_downloaded()
    artifacts = load_artifacts()

    with st.expander("ℹ️ Info model"):
        st.write(f"**Akurasi saat training:** {artifacts.get('accuracy', 0):.2%}")
        st.write(f"**Jumlah kelas:** {len(artifacts['class_names'])}")
        st.write("**Daftar kelas:**")
        st.write(", ".join(artifacts["class_names"]))

    uploaded_file = st.file_uploader(
        "Unggah gambar daun tomat (jpg/png)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        with st.spinner("Memuat model ekstraksi fitur (ResNet50)..."):
            extractor = load_feature_extractor()

        with st.spinner("Memproses prediksi..."):
            pred_label, prob_dict = predict(image, artifacts, extractor)

        friendly_label = LABEL_INFO.get(pred_label, pred_label)
        confidence = prob_dict[pred_label]

        st.subheader("Hasil Prediksi")
        if pred_label == "Tomato___healthy":
            st.success(f"✅ **{friendly_label}** (keyakinan {confidence:.1%})")
        else:
            st.warning(f"⚠️ **{friendly_label}** (keyakinan {confidence:.1%})")

        st.write("**Rincian probabilitas per kelas:**")
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        for label, p in sorted_probs:
            st.write(f"{LABEL_INFO.get(label, label)}")
            st.progress(min(max(p, 0.0), 1.0), text=f"{p:.1%}")

        st.caption(
            "Catatan: hasil prediksi bersifat prediktif, bukan diagnosis pasti. "
            "Untuk kasus penting, konsultasikan dengan ahli pertanian/penyuluh."
        )


if __name__ == "__main__":
    main()
