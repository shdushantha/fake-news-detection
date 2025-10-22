import streamlit as st
import torch
import numpy as np
from scipy.special import softmax

# Assuming these variables come directly from your notebook:
# - lr_model (Spark MLlib PipelineModel)
# - model (DistilBertForSequenceClassification)
# - tokenizer (DistilBertTokenizer)
# - spark (SparkSession)
# - device (torch.device)
# - bert_weight, lr_weight (ensemble weights)
# - FAKE_PROB_INDEX = 1

# If running standalone, ensure those objects are defined/imported
# from your notebook or checkpoint before running this file.

# ----------------------------------------------
# Streamlit UI Configuration
# ----------------------------------------------
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection Web App")
st.markdown(
    """
    This demo combines **Spark Logistic Regression** and **DistilBERT** to detect fake news.
    Enter a news headline or article below to check whether it's **Fake** or **Real**.
    """
)

# ----------------------------------------------
# Helper Function
# ----------------------------------------------
def predict_news(news_text: str):
    # Logistic Regression (Spark)
    sample_df = spark.createDataFrame([(news_text,)], ["full_text"])
    lr_pred = lr_model.transform(sample_df)
    prob_lr = lr_pred.select("probability").first().probability[FAKE_PROB_INDEX]

    # BERT
    inputs = tokenizer(news_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    logits = logits.cpu()
    probabilities = softmax(logits.numpy(), axis=1)[0]
    prob_bert = probabilities[1]

    # Weighted ensemble
    final_prob = (bert_weight * prob_bert) + (lr_weight * prob_lr)
    verdict = "Fake News" if final_prob > 0.5 else "True News"
    confidence = final_prob if verdict == "Fake News" else 1 - final_prob

    return verdict, confidence, prob_bert, prob_lr, final_prob


# ----------------------------------------------
# Input Section
# ----------------------------------------------
news_text = st.text_area("🖊️ Enter News Text:", height=200, placeholder="Type or paste a news article here...")

if st.button("Analyze"):
    if not news_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing... Please wait ⏳"):
            verdict, confidence, prob_bert, prob_lr, final_prob = predict_news(news_text)

        # ----------------------------------------------
        # Output Section
        # ----------------------------------------------
        st.markdown("---")
        st.subheader("🧾 Prediction Result")
        if verdict == "Fake News":
            st.error(f"🚨 **{verdict}**  \nConfidence: {confidence*100:.2f}%")
        else:
            st.success(f"✅ **{verdict}**  \nConfidence: {confidence*100:.2f}%")

        # Detailed Breakdown
        st.markdown("### 📊 Model Details")
        st.write(f"**Logistic Regression confidence (Fake):** {prob_lr:.4f}")
        st.write(f"**BERT confidence (Fake):** {prob_bert:.4f}")
        st.write(f"**Final Ensemble confidence (Fake):** {final_prob:.4f}")

        # Accuracy Display (static or from notebook)
        st.markdown("---")
        st.markdown("### 📈 Model Accuracy Summary")
        st.write(f"**Ensemble Accuracy:** {ensemble_accuracy * 100:.2f}%")
        st.write(f"**Ensemble AUC:** {ensemble_auc:.4f}")

        st.info("This prediction combines the strengths of MLlib Logistic Regression and fine-tuned DistilBERT.")

# ----------------------------------------------
# Sidebar Section
# ----------------------------------------------
st.sidebar.header("ℹ️ About")
st.sidebar.markdown(
    """
    **Fake News Detection System**

    - Ensemble: Logistic Regression + DistilBERT  
    - Frameworks: PySpark, Transformers (Hugging Face), Streamlit  
    - Developed for Master's Project in **AI & Big Data**  

    ---
    **Developer:** Dushantha Maduranga  
    **Version:** 1.0.0  
    """
)
