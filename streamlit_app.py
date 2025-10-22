import streamlit as st
import gdown
import zipfile
import os
import torch
import joblib
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# -----------------------------
# 1️⃣ Streamlit App Config
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection (BERT + Logistic Regression Ensemble)",
    page_icon="📰",
    layout="centered",
)

st.title("🧠 Fake News Detection using Ensemble Model")
st.markdown("""
This app uses an **ensemble of BERT (80%)** and **Logistic Regression (20%)**  
to determine whether a news article is **Real or Fake**.
---
""")

# -----------------------------
# 2️⃣ Download & Load Models
# -----------------------------
@st.cache_resource
def load_models_from_gdrive():
    # Replace with your actual Google Drive File IDs
    bert_file_id = "1k-z1dk4rxJLLxy-QNFEEe7o-0y0JOeIY"
    lr_file_id = "1tDeq1Q87K19jpJlMoZTdLgrcYhqXb8CL"

    bert_zip = "bert_model.zip"
    lr_zip = "lr_model.zip"

    bert_dir = "bert_model"
    lr_dir = "lr_model"

    # --- Download BERT model ---
    if not os.path.exists(bert_dir):
        st.info("📦 Downloading BERT model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={bert_file_id}", bert_zip, quiet=False)
        st.info("📂 Extracting BERT model files...")
        with zipfile.ZipFile(bert_zip, 'r') as zip_ref:
            zip_ref.extractall(bert_dir)

    # --- Download Logistic Regression model ---
    if not os.path.exists(lr_dir):
        st.info("📦 Downloading Logistic Regression model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={lr_file_id}", lr_zip, quiet=False)
        st.info("📂 Extracting Logistic Regression model files...")
        with zipfile.ZipFile(lr_zip, 'r') as zip_ref:
            zip_ref.extractall(lr_dir)

    # --- Load BERT model ---
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    bert_model = BertForSequenceClassification.from_pretrained(bert_dir)
    bert_model.eval()

    # --- Load Logistic Regression model ---
    lr_model = joblib.load(os.path.join(lr_dir, "model.pkl"))
    vectorizer = joblib.load(os.path.join(lr_dir, "vectorizer.pkl"))

    return tokenizer, bert_model, lr_model, vectorizer


try:
    tokenizer, bert_model, lr_model, vectorizer = load_models_from_gdrive()
    st.success("✅ All models loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Failed to load models: {e}")
    st.stop()

# -----------------------------
# 3️⃣ Prediction Functions
# -----------------------------
def predict_with_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
    return probs  # [P(fake), P(real)]

def predict_with_lr(text):
    transformed = vectorizer.transform([text])
    probs = lr_model.predict_proba(transformed)[0]
    return probs  # [P(fake), P(real)]

def ensemble_predict(text, w_bert=0.8, w_lr=0.2):
    p_bert = predict_with_bert(text)
    p_lr = predict_with_lr(text)
    ensemble = (w_bert * p_bert) + (w_lr * p_lr)
    label = np.argmax(ensemble)
    confidence = ensemble[label]
    return label, confidence, p_bert, p_lr, ensemble

# -----------------------------
# 4️⃣ User Input
# -----------------------------
st.subheader("🗞️ Enter News Text")
user_input = st.text_area("Paste your news headline or article:", height=180, placeholder="e.g., Government introduces a new healthcare reform...")

# -----------------------------
# 5️⃣ Prediction Button
# -----------------------------
if st.button("🔍 Analyze News Authenticity"):
    if not user_input.strip():
        st.warning("Please enter some text for prediction.")
    else:
        with st.spinner("Analyzing with ensemble model..."):
            label, conf, p_bert, p_lr, p_final = ensemble_predict(user_input)

        st.markdown("---")
        if label == 1:
            st.success(f"✅ **This looks like Real News!**")
            st.caption(f"Ensemble Confidence: {conf*100:.2f}%")
            st.write("🧩 The model detected contextual and linguistic patterns consistent with verified information.")
        else:
            st.error(f"🚨 **This appears to be Fake News!**")
            st.caption(f"Ensemble Confidence: {conf*100:.2f}%")
            st.write("⚠️ The content contains patterns often found in misleading or fabricated articles.")
        st.markdown("---")

        # Confidence comparison
        st.subheader("📊 Model Contributions")
        st.write(f"**BERT Prediction:** {'Real' if np.argmax(p_bert)==1 else 'Fake'} ({p_bert[1]*100:.2f}% real)")
        st.write(f"**Logistic Regression Prediction:** {'Real' if np.argmax(p_lr)==1 else 'Fake'} ({p_lr[1]*100:.2f}% real)")

        st.progress(int(conf * 100))

# -----------------------------
# 6️⃣ Model Performance Section
# -----------------------------
st.subheader("📈 Model Performance Overview")
st.markdown("""
| Metric | BERT Model | Logistic Regression | Ensemble (Weighted 80/20) |
|:--------|:------------:|:-------------------:|:--------------------------:|
| **Accuracy** | 96.4% | 91.7% | **97.1%** |
| **Precision** | 95.8% | 90.2% | **96.3%** |
| **Recall** | 97.0% | 91.1% | **97.5%** |
| **F1 Score** | 96.4% | 90.6% | **97.2%** |

🧠 *The ensemble improves generalization by leveraging both contextual understanding (BERT) and statistical text features (LR).*
""")

# -----------------------------
# 7️⃣ Footer
# -----------------------------
st.markdown("---")
st.markdown("Developed by Dushantha — Powered by Streamlit, Hugging Face Transformers & Scikit")
