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


MODEL_URL = "https://huggingface.co/datasets/rinaldi2211/tomato-disease-model/resolve/main/dense_classifier_model.keras"
CLASS_NAMES_URL = "https://huggingface.co/datasets/rinaldi2211/tomato-disease-model/resolve/main/dense_class_names.pkl"

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


def auto_crop_leaf(image: Image.Image, padding_ratio: float = 0.15) -> tuple[Image.Image, bool]:
    """Deteksi area daun (warna hijau) secara sederhana dan crop ke bounding box-nya.

    Ini membantu mengurangi pengaruh latar belakang yang kompleks (tanah, daun lain,
    langit, dsb.) pada citra dunia nyata, dengan mengisolasi area utama daun sebelum
    diekstrak fiturnya. Ini adalah langkah TAMBAHAN yang tidak ada saat ekstraksi
    features.npy asli, jadi sifatnya eksperimental -- efeknya bisa membantu atau
    tidak berpengaruh, tergantung kondisi gambar.

    Return: (gambar_hasil_crop, apakah_crop_dilakukan)
    """
    hsv = image.convert("HSV")
    h_arr = np.array(hsv.getchannel("H"), dtype=np.int16)
    s_arr = np.array(hsv.getchannel("S"), dtype=np.int16)
    v_arr = np.array(hsv.getchannel("V"), dtype=np.int16)

    # Rentang hue "hijau daun" dalam skala PIL (0-255, setara ~50-170 derajat dari 360)
    green_mask = (h_arr >= 35) & (h_arr <= 120) & (s_arr >= 40) & (v_arr >= 30)

    # Kalau area hijau yang terdeteksi terlalu sedikit (<1% piksel), jangan crop --
    # kemungkinan gambar sudah close-up/latar polos, atau daun sedang sakit parah
    # (banyak bercak coklat) sehingga warna hijau tersisa sedikit.
    if green_mask.sum() < 0.01 * green_mask.size:
        return image, False

    rows = np.any(green_mask, axis=1)
    cols = np.any(green_mask, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    h_img, w_img = green_mask.shape
    pad_h = int((bottom - top) * padding_ratio)
    pad_w = int((right - left) * padding_ratio)
    top = max(0, top - pad_h)
    bottom = min(h_img - 1, bottom + pad_h)
    left = max(0, left - pad_w)
    right = min(w_img - 1, right + pad_w)

    cropped = image.crop((left, top, right + 1, bottom + 1))
    return cropped, True


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

    st.caption("💡 Untuk hasil terbaik: foto close-up satu daun, latar belakang polos, pencahayaan cukup.")

    use_auto_crop = st.checkbox(
        "Aktifkan deteksi otomatis area daun (eksperimental)",
        value=True,
        help="Mencoba memotong latar belakang secara otomatis berdasarkan warna hijau daun, "
             "untuk membantu foto dengan latar belakang kompleks (kebun, tanah, dsb).",
    )

    uploaded_file = st.file_uploader("Unggah gambar daun tomat (jpg/png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        image_to_predict = image
        was_cropped = False
        if use_auto_crop:
            image_to_predict, was_cropped = auto_crop_leaf(image)

        col1, col2 = st.columns(2) if was_cropped else (st.container(), None)
        with col1:
            st.image(image, caption="Gambar asli", use_column_width=True)
        if was_cropped:
            with col2:
                st.image(image_to_predict, caption="Setelah auto-crop", use_column_width=True)
        elif use_auto_crop:
            st.caption("ℹ️ Area daun tidak terdeteksi jelas, prediksi memakai gambar asli (tanpa crop).")

        with st.spinner("Memuat model ekstraksi fitur (ResNet50)..."):
            extractor = load_feature_extractor()

        with st.spinner("Memproses prediksi..."):
            pred_label, prob_dict = predict(image_to_predict, model, class_names, extractor)

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

