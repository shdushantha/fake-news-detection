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
This app uses a fine-tuned **BERT Transformer (SafeTensors format)** model  
to classify whether a news article is **Real** or **Fake**.
---
""")

# ---------------------------------------------------------------------
# 2️⃣ Download + extract Google Drive ZIP
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


def find_model_dir(base_dir):
    """
    Recursively find the folder containing both config.json and model weights (.bin or .safetensors).
    Works with any nested folder structure.
    """
    for root, _, files in os.walk(base_dir):
        if "config.json" in files and any(f.endswith((".bin", ".safetensors")) for f in files):
            return root
    raise FileNotFoundError("❌ Could not locate config.json or model.safetensors file in the extracted model.")

# ---------------------------------------------------------------------
# 3️⃣ Load BERT model and tokenizer
# ---------------------------------------------------------------------
@st.cache_resource
def load_bert_model():
    # 🔹 Replace this with your actual Google Drive file ID
    # https://drive.google.com/file/d/1Cs9qaSdQnPP6G7EGBs-IQAN0P_axBY0C/view?usp=sharing
    bert_file_id = "1Cs9qaSdQnPP6G7EGBs-IQAN0P_axBY0C"

    bert_dir = download_and_unzip(bert_file_id, "bert_model.zip", "bert_model")
    model_dir = find_model_dir(bert_dir)
    st.write(f"📁 Detected BERT model directory: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    st.success("✅ BERT model and tokenizer loaded successfully!")
    return tokenizer, model


try:
    tokenizer, bert_model = load_bert_model()
except Exception as e:
    st.error(f"⚠️ Model loading failed: {e}")
    st.stop()

# ---------------------------------------------------------------------
# 4️⃣ Prediction
# ---------------------------------------------------------------------
def predict_with_bert(text: str):
    """Predict Fake vs Real using the BERT model."""
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

        # -----------------------------------------------------------------
        # 🥧 Add pie chart visualization
        # -----------------------------------------------------------------
        st.subheader("🎯 Probability Distribution")
        labels = ["Fake", "Real"]
        colors = ["#FF6F61", "#4CAF50"]
        explode = (0.05, 0.05)  # separate slices slightly
        fig, ax = plt.subplots()
        ax.pie(
            probs,
            labels=labels,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            explode=explode,
            shadow=True,
            textprops={"fontsize": 12, "weight": "bold"}
        )
        ax.axis("equal")  # Equal aspect ratio ensures pie is circular
        st.pyplot(fig)

st.markdown("---")
st.caption("🧠 Developed by Dushantha (SherinDe) · Powered by Streamlit + Hugging Face Transformers")

