import streamlit as st
import os, zipfile, shutil, gdown
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------
# 1️⃣ Streamlit page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="Fake News Detection (BERT)", page_icon="🧠")
st.title("📰 Fake News Detection – BERT Model")
st.markdown("""
This app uses a fine-tuned **BERT Transformer** model  
to classify whether a news article is **Real** or **Fake**.
---
""")

# ---------------------------------------------------------------------
# 2️⃣ Helper functions
# ---------------------------------------------------------------------
def download_and_unzip(file_id, zip_name, extract_dir):
    """Download and extract model ZIP from Google Drive."""
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


def find_bert_folder(base_dir: str):
    """
    Recursively find the folder that contains both:
    - config.json
    - a *.bin model file (like pytorch_model.bin)
    """
    for root, _, files in os.walk(base_dir):
        has_config = "config.json" in files
        has_bin = any(f.endswith(".bin") for f in files)
        if has_config and has_bin:
            return root
    raise FileNotFoundError("❌ Could not locate config.json or .bin file in extracted model folder structure.")

# ---------------------------------------------------------------------
# 3️⃣ Load BERT model + tokenizer (cached)
# ---------------------------------------------------------------------
@st.cache_resource
def load_bert_model():
    # 🔹 Replace this with your actual Google Drive file ID
	# https://drive.google.com/file/d/1Cs9qaSdQnPP6G7EGBs-IQAN0P_axBY0C/view?usp=sharing
    bert_file_id = "1Cs9qaSdQnPP6G7EGBs-IQAN0P_axBY0C"

    # Download and extract
    bert_dir = download_and_unzip(bert_file_id, "bert_model.zip", "bert_model")

    # Automatically find actual model folder
    bert_model_dir = find_bert_folder(bert_dir)
    st.write(f"📁 Detected BERT directory: {bert_model_dir}")

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(bert_model_dir)
    model.eval()

    st.success("✅ BERT model and tokenizer loaded successfully!")
    return tokenizer, model


try:
    tokenizer, bert_model = load_bert_model()
except Exception as e:
    st.error(f"⚠️ Model loading failed: {e}")
    st.stop()

# ---------------------------------------------------------------------
# 4️⃣ Prediction function
# ---------------------------------------------------------------------
def predict_with_bert(text: str):
    """Run prediction using the BERT model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy().squeeze()
    return probs  # [P(fake), P(real)]

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
        with st.spinner("Analyzing using BERT model …"):
            probs = predict_with_bert(user_input)
            label = int(np.argmax(probs))
            confidence = float(probs[label])

        st.markdown("---")
        if label == 1:
            st.success(f"✅ This appears to be **Real News** ({confidence*100:.2f}% confidence)")
        else:
            st.error(f"🚨 This appears to be **Fake News** ({confidence*100:.2f}% confidence)")

        st.markdown("---")
        st.subheader("📊 Model Confidence")
        st.write(f"**Fake:** {probs[0]*100:.2f}%")
        st.write(f"**Real:** {probs[1]*100:.2f}%")
        st.progress(int(confidence * 100))

st.markdown("---")
st.caption("🧠 Developed by Dushantha (SherinDe) · Powered by Streamlit + Hugging Face Transformers")
