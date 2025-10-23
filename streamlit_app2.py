import streamlit as st
import os, zipfile, shutil, gdown, json
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import torch

# ---------------------------------------------------------------------
# 1️⃣ Streamlit page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="Fake News Detection (BERT + LSTM Ensemble)", page_icon="🧠")
st.title("📰 Fake News Detection – BERT + LSTM Hybrid")
st.markdown("""
This app uses an **ensemble** of a Transformer (**BERT**, 80 %) and an **LSTM** (20 %)  
to classify whether a news article is **Real** or **Fake**.
---
""")

# ---------------------------------------------------------------------
# 2️⃣ Utilities: download + extract Google Drive ZIPs
# ---------------------------------------------------------------------
def download_and_unzip(file_id, zip_name, extract_dir):
    url = f"https://drive.google.com/uc?id={file_id}"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    st.info(f"📦 Downloading {zip_name} …")
    output = gdown.download(url, zip_name, quiet=False)

    if output is None or not os.path.exists(zip_name) or os.path.getsize(zip_name) == 0:
        raise FileNotFoundError(f"❌ Download failed or empty file: {zip_name}")

    st.info(f"📂 Extracting {zip_name} …")
    with zipfile.ZipFile(zip_name, "r") as zf:
        zf.extractall(extract_dir)

    if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) == 0:
        raise FileNotFoundError(f"❌ Extraction failed: {extract_dir}")

    st.success(f"✅ Extracted {zip_name} → {extract_dir}")
    return extract_dir


def find_model_folder(base_dir: str) -> str:
    """Locate the directory containing config.json for Hugging Face models."""
    for root, _, files in os.walk(base_dir):
        if "config.json" in files:
            return root
    raise FileNotFoundError(f"config.json not found under {base_dir}")

# ---------------------------------------------------------------------
# 3️⃣ Load models (cached)
# ---------------------------------------------------------------------
@st.cache_resource
def load_models():
    # 👉 Replace with your actual Google Drive file IDs
    # https://drive.google.com/file/d/1Cs9qaSdQnPP6G7EGBs-IQAN0P_axBY0C/view?usp=sharing
    bert_file_id = "1Cs9qaSdQnPP6G7EGBs-IQAN0P_axBY0C"
    # https://drive.google.com/file/d/1GDvAdAnP5zX7Cat7J0AFiKyj0kykIdqj/view?usp=sharing
    lstm_file_id = "1GDvAdAnP5zX7Cat7J0AFiKyj0kykIdqj"

    bert_dir = download_and_unzip(bert_file_id, "bert_model.zip", "bert_model")
    lstm_dir = download_and_unzip(lstm_file_id, "lstm_model.zip", "lstm_model_package")

    # --- Load BERT ---
    st.info("🧠 Loading BERT model + tokenizer …")
    bert_dir_actual = find_model_folder(bert_dir)
    st.write(f"📁 Detected BERT directory: {bert_dir_actual}")
    tokenizer = AutoTokenizer.from_pretrained(bert_dir_actual)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir_actual)
    bert_model.eval()

    # --- Load LSTM (.keras + tokenizer) ---
    st.info("🔁 Loading LSTM model (.keras + tokenizer) …")
    keras_files = [os.path.join(root, f)
                   for root, _, files in os.walk(lstm_dir)
                   for f in files if f.endswith(".keras")]
    if not keras_files:
        raise FileNotFoundError("No .keras file found inside LSTM directory!")
    lstm_model_path = keras_files[0]
    st.write(f"📁 Detected LSTM model file: {lstm_model_path}")
    lstm_model = tf.keras.models.load_model(lstm_model_path)

   # Load tokenizer
    tok_path = os.path.join(lstm_dir, "tokenizer_lstm.json")
    if not os.path.exists(tok_path):
        for root, _, files in os.walk(lstm_dir):
            if "tokenizer_lstm.json" in files:
                tok_path = os.path.join(root, "tokenizer_lstm.json")
            break
    if not os.path.exists(tok_path):
        raise FileNotFoundError("tokenizer_lstm.json not found in LSTM ZIP")

    # ✅ Correct way: read as string, not dict
    with open(tok_path, "r") as f:
        tokenizer_json = f.read()
    tokenizer_lstm = tokenizer_from_json(tokenizer_json)


    # Label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(["Fake", "Real"])

    return tokenizer, bert_model, lstm_model, tokenizer_lstm, label_encoder


try:
    tokenizer, bert_model, lstm_model, tokenizer_lstm, label_encoder = load_models()
    st.success("✅ All models loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Model loading failed: {e}")
    st.stop()

# ---------------------------------------------------------------------
# 4️⃣ Prediction functions
# ---------------------------------------------------------------------
def predict_with_bert(text: str):
    """Predict using BERT Transformer"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():  # disable gradients
        logits = bert_model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy().squeeze()
    return probs  # [P(fake), P(real)]


def predict_with_lstm(text: str):
    """Predict using LSTM model and tokenizer"""
    seq = tokenizer_lstm.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
    probs = lstm_model.predict(padded, verbose=0).flatten()
    if probs.size == 1:
        probs = np.array([1 - probs[0], probs[0]])
    return probs  # [P(fake), P(real)]


def ensemble_predict(text, w_bert=0.8, w_lstm=0.2):
    """Weighted ensemble between BERT and LSTM predictions"""
    p_bert = predict_with_bert(text)
    p_lstm = predict_with_lstm(text)
    ensemble = w_bert * p_bert + w_lstm * p_lstm
    label = int(np.argmax(ensemble))
    conf = float(ensemble[label])
    return label, conf, p_bert, p_lstm

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
            label, conf, p_bert, p_lstm = ensemble_predict(user_input)

        st.markdown("---")
        if label == 1:
            st.success(f"✅ This appears to be **Real News** ({conf*100:.2f}% confidence)")
        else:
            st.error(f"🚨 This appears to be **Fake News** ({conf*100:.2f}% confidence)")

        st.markdown("---")
        st.subheader("📊 Model Contributions")
        st.write(f"**BERT Prediction:** {'Real' if np.argmax(p_bert)==1 else 'Fake'} ({p_bert[1]*100:.2f}% real)")
        st.write(f"**LSTM Prediction:** {'Real' if np.argmax(p_lstm)==1 else 'Fake'} ({p_lstm[1]*100:.2f}% real)")
        st.progress(int(conf * 100))

st.markdown("---")
st.caption("🧠 Developed by Dushantha (SherinDe) · Powered by Streamlit + TensorFlow + Hugging Face")

