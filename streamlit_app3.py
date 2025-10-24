import streamlit as st
import os, zipfile, shutil, gdown
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import torch

# ---------------------------------------------------------------------
# 1️⃣ Streamlit setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="Fake News Detection (BERT + CNN Ensemble)", page_icon="🧠")
st.title("📰 Fake News Detection – BERT + CNN Hybrid")
st.markdown("""
This app uses an **ensemble** of a Transformer (**BERT**, 80%) and a **CNN model** (20%)  
to classify whether a news article is **Real** or **Fake**.
---
""")

# ---------------------------------------------------------------------
# 2️⃣ Helper functions
# ---------------------------------------------------------------------
def download_and_unzip(file_id, zip_name, extract_dir):
    """Download and extract a ZIP file from Google Drive."""
    url = f"https://drive.google.com/uc?id={file_id}"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    st.info(f"📦 Downloading {zip_name} …")
    output = gdown.download(url, zip_name, quiet=False)

    if not output or not os.path.exists(zip_name) or os.path.getsize(zip_name) == 0:
        raise FileNotFoundError(f"❌ Download failed or empty file: {zip_name}")

    st.info(f"📂 Extracting {zip_name} …")
    with zipfile.ZipFile(zip_name, "r") as zf:
        zf.extractall(extract_dir)
    st.success(f"✅ Extracted {zip_name} → {extract_dir}")
    return extract_dir


def find_model_folder(base_dir: str) -> str:
    """Finds a Hugging Face model directory containing config.json."""
    for root, _, files in os.walk(base_dir):
        if "config.json" in files:
            return root
    raise FileNotFoundError(f"config.json not found in {base_dir}")

# ---------------------------------------------------------------------
# 3️⃣ Load models (cached)
# ---------------------------------------------------------------------
@st.cache_resource
def load_models():
    # 🔹 Replace with your actual Google Drive file IDs
    # https://drive.google.com/file/d/1Cs9qaSdQnPP6G7EGBs-IQAN0P_axBY0C/view?usp=sharing
    bert_file_id = "1Cs9qaSdQnPP6G7EGBs-IQAN0P_axBY0C"
    # https://drive.google.com/file/d/1KaIr9zdOqolNVeV0Lc58dnsAOklIDYLa/view?usp=sharing
    cnn_file_id = "1KaIr9zdOqolNVeV0Lc58dnsAOklIDYLa"

    # --- Download + extract ---
    bert_dir = download_and_unzip(bert_file_id, "bert_model.zip", "bert_model")
    cnn_dir = download_and_unzip(cnn_file_id, "fake_news_cnn_model.zip", "fake_news_cnn_model")

    # --- Load BERT ---
    st.info("🧠 Loading BERT model + tokenizer …")
    bert_dir_actual = find_model_folder(bert_dir)
    tokenizer = AutoTokenizer.from_pretrained(bert_dir_actual)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir_actual)
    bert_model.eval()

    # --- Load CNN (.h5 only) ---
    st.info("🔁 Loading CNN model (.h5 file only) …")

    # Recursively search for cnn_model.h5
    cnn_model_path = None
    for root, _, files in os.walk(cnn_dir):
        for f in files:
            if f.lower() == "cnn_model.h5":
                cnn_model_path = os.path.join(root, f)
                break
        if cnn_model_path:
        break

    if not cnn_model_path or not os.path.exists(cnn_model_path):
        raise FileNotFoundError("cnn_model.h5 not found inside fake_news_cnn_model.zip")

    st.write(f"📁 Detected CNN model file: {cnn_model_path}")
    cnn_model = tf.keras.models.load_model(cnn_model_path)

    # Label encoder (for display)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(["Fake", "Real"])

    st.success("✅ Models loaded successfully.")
    return tokenizer, bert_model, cnn_model, label_encoder


try:
    tokenizer, bert_model, cnn_model, label_encoder = load_models()
except Exception as e:
    st.error(f"⚠️ Model loading failed: {e}")
    st.stop()

# ---------------------------------------------------------------------
# 4️⃣ Prediction functions
# ---------------------------------------------------------------------
def predict_with_bert(text: str):
    """Predict using BERT Transformer"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy().squeeze()
    return probs  # [P(fake), P(real)]


def simple_text_preprocess(text: str, max_len=200):
    """Fallback tokenizer logic for CNN (if no tokenizer.json available)."""
    text = text.lower()
    tokens = text.split()[:max_len]  # simple whitespace split
    vocab = {w: i+1 for i, w in enumerate(sorted(set(tokens)))}
    seq = [vocab.get(w, 0) for w in tokens]
    padded = pad_sequences([seq], maxlen=max_len, padding='post', truncating='post')
    return padded


def predict_with_cnn(text: str):
    """Predict using CNN model with fallback tokenizer."""
    padded = simple_text_preprocess(text)
    p_real = float(cnn_model.predict(padded, verbose=0).flatten()[0])
    return np.array([1 - p_real, p_real])  # [P(fake), P(real)]


def ensemble_predict(text, w_bert=0.8, w_cnn=0.2):
    """Weighted ensemble between BERT and CNN predictions"""
    p_bert = predict_with_bert(text)
    p_cnn = predict_with_cnn(text)
    ensemble = w_bert * p_bert + w_cnn * p_cnn
    label = int(np.argmax(ensemble))
    conf = float(ensemble[label])
    return label, conf, p_bert, p_cnn

# ---------------------------------------------------------------------
# 5️⃣ Streamlit UI
# ---------------------------------------------------------------------
st.subheader("🗞️ Enter News Text")
user_input = st.text_area(
    "Paste a news headline or article below:",
    height=180,
    placeholder="e.g., Government launches new renewable-energy plan ..."
)

if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text for analysis.")
    else:
        with st.spinner("Analyzing with ensemble model …"):
            label, conf, p_bert, p_cnn = ensemble_predict(user_input)

        st.markdown("---")
        if label == 1:
            st.success(f"✅ This appears to be **Real News** ({conf*100:.2f}% confidence)")
        else:
            st.error(f"🚨 This appears to be **Fake News** ({conf*100:.2f}% confidence)")

        st.markdown("---")
        st.subheader("📊 Model Contributions")
        st.write(f"**BERT Prediction:** {'Real' if np.argmax(p_bert)==1 else 'Fake'} ({p_bert[1]*100:.2f}% real)")
        st.write(f"**CNN Prediction:** {'Real' if np.argmax(p_cnn)==1 else 'Fake'} ({p_cnn[1]*100:.2f}% real)")
        st.progress(int(conf * 100))

st.markdown("---")
st.caption("🧠 Developed by Dushantha (SherinDe) · Powered by Streamlit + TensorFlow + Hugging Face")
