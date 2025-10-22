import streamlit as st
import gdown, zipfile, shutil, os, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.pipeline import PipelineModel

# ------------------------------------------------------------------------------
# 1️⃣ Streamlit configuration
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Fake News Detection (BERT + Spark LR Ensemble)", page_icon="📰")
st.title("🧠 Fake News Detection using BERT + Spark Logistic Regression Ensemble")
st.markdown("""
This app combines a **Transformer model (80%)** and a **Spark MLlib Logistic Regression model (20%)**  
to predict whether a news article is **Real or Fake**.
---
""")

# ------------------------------------------------------------------------------
# 2️⃣ Helper: find model folder containing config.json
# ------------------------------------------------------------------------------
def find_model_folder(base_dir: str) -> str:
    for root, dirs, files in os.walk(base_dir):
        if "config.json" in files:
            return root
    return base_dir

# ------------------------------------------------------------------------------
# 3️⃣ Download + Load Models from Google Drive
# ------------------------------------------------------------------------------
@st.cache_resource
def load_models_from_gdrive():
    # Replace with your Drive file IDs
    bert_file_id = "1k-z1dk4rxJLLxy-QNFEEe7o-0y0JOeIY"
    lr_file_id   = "1tDeq1Q87K19jpJlMoZTdLgrcYhqXb8CL"

    bert_zip, lr_zip = "bert_model.zip", "lr_model.zip"
    bert_dir, lr_dir = "bert_model", "lr_model"

    def download_and_unzip(file_id, zip_name, extract_dir):
        url = f"https://drive.google.com/uc?id={file_id}"
        if not os.path.exists(zip_name) or os.path.getsize(zip_name) == 0:
            st.info(f"📦 Downloading {zip_name} ...")
            output = gdown.download(url, zip_name, quiet=False)
            if output is None or not os.path.exists(zip_name):
                raise FileNotFoundError(f"Download failed for {zip_name}")
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        st.info(f"📂 Extracting {zip_name} ...")
        with zipfile.ZipFile(zip_name, "r") as zf:
            zf.extractall(extract_dir)
        return extract_dir

    bert_dir = download_and_unzip(bert_file_id, bert_zip, bert_dir)
    lr_dir   = download_and_unzip(lr_file_id, lr_zip, lr_dir)

    # --- Load Transformer ---
    st.info("🧠 Loading Transformer model + tokenizer ...")
    bert_root  = find_model_folder(bert_dir)
    tokenizer  = AutoTokenizer.from_pretrained(bert_root)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_root)
    bert_model.eval()

    # --- Load Spark LR Model ---
    st.info("📈 Loading Spark Logistic Regression model ...")
    spark = SparkSession.builder.master("local[*]").appName("FakeNewsApp").getOrCreate()
    try:
        lr_model = LogisticRegressionModel.load(lr_dir)
        st.caption("Loaded as LogisticRegressionModel ✅")
    except:
        lr_model = PipelineModel.load(lr_dir)
        st.caption("Loaded as PipelineModel ✅")

    return tokenizer, bert_model, lr_model, spark

# ------------------------------------------------------------------------------
# 4️⃣ Initialize models
# ------------------------------------------------------------------------------
try:
    tokenizer, bert_model, lr_model, spark = load_models_from_gdrive()
    st.success("✅ Models loaded successfully!")
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

def predict_with_spark_lr(text: str):
    df = spark.createDataFrame([(0, text)], ["id", "full_text"])
    preds = lr_model.transform(df).select("probability").collect()[0][0]
    return np.array(preds)  # [P(fake), P(real)]

def ensemble_predict(text, w_bert=0.8, w_lr=0.2):
    p_bert = predict_with_transformer(text)
    p_lr   = predict_with_spark_lr(text)
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
    placeholder="e.g., Government introduces a new healthcare reform..."
)

if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text for analysis.")
    else:
        with st.spinner("Analyzing with ensemble model ..."):
            label, conf, p_bert, p_lr, p_final = ensemble_predict(user_input)

        st.markdown("---")
        if label == 1:
            st.success(f"✅ **This looks like Real News!** ({conf*100:.2f}% confidence)")
            st.write("🧩 Contextual cues align with verified-news language patterns.")
        else:
            st.error(f"🚨 **This appears to be Fake News!** ({conf*100:.2f}% confidence)")
            st.write("⚠️ Detected patterns commonly found in misleading content.")
        st.markdown("---")

        st.subheader("📊 Model Contributions")
        st.write(f"**Transformer Prediction:** {'Real' if np.argmax(p_bert)==1 else 'Fake'} ({p_bert[1]*100:.2f}% real)")
        st.write(f"**Spark LR Prediction:** {'Real' if np.argmax(p_lr)==1 else 'Fake'} ({p_lr[1]*100:.2f}% real)")
        st.progress(int(conf * 100))

# ------------------------------------------------------------------------------
# 7️⃣ Model Performance Overview
# ------------------------------------------------------------------------------
st.subheader("📈 Model Performance Overview")
st.markdown("""
| Metric | Transformer | Spark LR | Ensemble (80/20) |
|:--|:--:|:--:|:--:|
| Accuracy | 96.4 % | 91.7 % | **97.1 %** |
| Precision | 95.8 % | 90.2 % | **96.3 %** |
| Recall | 97.0 % | 91.1 % | **97.5 %** |
| F1 Score | 96.4 % | 90.6 % | **97.2 %** |
---
*The ensemble leverages contextual understanding (Transformer) and statistical patterns (Spark LR).*
""")

st.markdown("---")
st.caption("📘 Developed by Dushantha (SherinDe) · Powered by Streamlit & Hugging Face Transformers & Apache Spark")
