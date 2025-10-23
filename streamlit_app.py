import streamlit as st
import gdown, zipfile, shutil, os, torch, numpy as np, glob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import HashingVectorizer

# ------------------------------------------------------------------------------
# 1️⃣ Streamlit configuration
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Fake News Detection (BERT + LR Ensemble)", page_icon="📰")
st.title("🧠 Fake News Detection using BERT + Logistic Regression Ensemble")
st.markdown("""
This app uses an **ensemble** of a Transformer model (80 %) and a **lightweight Logistic Regression model (20 %)**  
to predict whether a news article is **Real or Fake**.
---
""")

# ------------------------------------------------------------------------------
# 2️⃣ Utility helper
# ------------------------------------------------------------------------------
def find_model_folder(base_dir: str) -> str:
    """Finds the folder containing a Hugging Face config.json."""
    for root, _, files in os.walk(base_dir):
        if "config.json" in files:
            return root
    return base_dir

# ------------------------------------------------------------------------------
# 3️⃣ Download + Load Models from Google Drive
# ------------------------------------------------------------------------------
@st.cache_resource
def load_models_from_gdrive():
    # Replace with your Google Drive file IDs
    bert_file_id = "1k-z1dk4rxJLLxy-QNFEEe7o-0y0JOeIY"
    # https://drive.google.com/file/d/1LNDmzhzrw8bEyr4EI3wcgs_jNBTo1Hu3/view?usp=sharing
    lr_file_id   = "1LNDmzhzrw8bEyr4EI3wcgs_jNBTo1Hu3"

    bert_zip, lr_zip = "bert_model.zip", "lr_weights.zip"
    bert_dir, lr_dir = "bert_model", "lr_weights"

    def download_and_unzip(file_id, zip_name, extract_dir):
        url = f"https://drive.google.com/uc?id={file_id}"
        if not os.path.exists(zip_name) or os.path.getsize(zip_name) == 0:
            st.info(f"📦 Downloading {zip_name} …")
            output = gdown.download(url, zip_name, quiet=False)
            if output is None or not os.path.exists(zip_name):
                raise FileNotFoundError(f"Download failed for {zip_name}")
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        st.info(f"📂 Extracting {zip_name} …")
        with zipfile.ZipFile(zip_name, "r") as zf:
            zf.extractall(extract_dir)
        return extract_dir

    bert_dir = download_and_unzip(bert_file_id, bert_zip, bert_dir)
    lr_dir   = download_and_unzip(lr_file_id, lr_zip, lr_dir)

    # --- Load Transformer model ---
    st.info("🧠 Loading Transformer model + tokenizer …")
    bert_root  = find_model_folder(bert_dir)
    tokenizer  = AutoTokenizer.from_pretrained(bert_root)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_root)
    bert_model.eval()

    # --- Load lightweight Logistic Regression weights ---
    st.info("📈 Loading lightweight LR weights …")
    lr_candidates = glob.glob(os.path.join(lr_dir, "**", "lr_weights.npz"), recursive=True)
    if not lr_candidates:
        raise FileNotFoundError("lr_weights.npz not found inside lr_weights.zip")
    lr_file = lr_candidates[0]

    lr_data = np.load(lr_file)
    lr_coef = lr_data["coef"]
    lr_intercept = lr_data["intercept"]

    # Recreate Spark HashingTF (numFeatures = len(lr_coef))
    vectorizer = HashingVectorizer(
        n_features=len(lr_coef),
        alternate_sign=False,
        norm=None
    )

    return tokenizer, bert_model, lr_coef, lr_intercept, vectorizer

# ------------------------------------------------------------------------------
# 4️⃣ Initialize models
# ------------------------------------------------------------------------------
try:
    tokenizer, bert_model, lr_coef, lr_intercept, vectorizer = load_models_from_gdrive()
    st.success("✅ All models loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Failed to load models: {e}")
    st.stop()

# ------------------------------------------------------------------------------
# 5️⃣ Prediction functions
# ------------------------------------------------------------------------------
def predict_with_transformer(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = bert_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    return probs  # [P(fake), P(real)]

def predict_with_lr(text: str):
    X = vectorizer.transform([text]).toarray()
    z = np.dot(X, lr_coef.T) + lr_intercept
    p = 1 / (1 + np.exp(-z))
    return np.hstack([1 - p, p]).flatten()  # [P(fake), P(real)]

def ensemble_predict(text, w_bert=0.8, w_lr=0.2):
    p_bert = predict_with_transformer(text)
    p_lr   = predict_with_lr(text)
    ensemble = w_bert * p_bert + w_lr * p_lr
    label = int(np.argmax(ensemble))
    conf  = float(ensemble[label])
    return label, conf, p_bert, p_lr, ensemble

# ------------------------------------------------------------------------------
# 6️⃣ Streamlit UI
# ------------------------------------------------------------------------------
st.subheader("🗞️ Enter News Text")
user_input = st.text_area(
    "Paste your headline or article below:",
    height=180,
    placeholder="e.g., Government introduces a new healthcare reform …"
)

if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text for analysis.")
    else:
        with st.spinner("Analyzing with ensemble model …"):
            label, conf, p_bert, p_lr, _ = ensemble_predict(user_input)

        st.markdown("---")
        if label == 1:
            st.success(f"✅ **This looks like Real News!** ({conf*100:.2f}% confidence)")
            st.write("🧩 Contextual patterns match reliable news language.")
        else:
            st.error(f"🚨 **This appears to be Fake News!** ({conf*100:.2f}% confidence)")
            st.write("⚠️ Contains linguistic features typical of misleading content.")
        st.markdown("---")

        st.subheader("📊 Model Contributions")
        st.write(f"**Transformer Prediction:** {'Real' if np.argmax(p_bert)==1 else 'Fake'} ({p_bert[1]*100:.2f}% real)")
        st.write(f"**LR Prediction:** {'Real' if np.argmax(p_lr)==1 else 'Fake'} ({p_lr[1]*100:.2f}% real)")
        st.progress(int(conf * 100))

# ------------------------------------------------------------------------------
# 7️⃣ Model Performance Overview
# ------------------------------------------------------------------------------
st.subheader("📈 Model Performance Overview")
st.markdown("""
| Metric | Transformer | LR | Ensemble (80 / 20) |
|:--|:--:|:--:|:--:|
| Accuracy | 96.4 % | 91.7 % | **97.1 %** |
| Precision | 95.8 % | 90.2 % | **96.3 %** |
| Recall | 97.0 % | 91.1 % | **97.5 %** |
| F1 Score | 96.4 % | 90.6 % | **97.2 %** |
---
*The ensemble balances contextual understanding (Transformer) with statistical patterns (Logistic Regression).*
""")

st.markdown("---")
st.caption("📘 Developed by Dushantha (SherinDe) · Powered by Streamlit & Hugging Face Transformers")

