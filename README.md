# Fake News Detection with PySpark and Transformers

This project implements a sophisticated fake news detection system using a combination of big data processing with PySpark and advanced NLP with a fine-tuned DistilBERT model from Hugging Face Transformers. The models are combined into a weighted ensemble to achieve high accuracy.

## 📝 Description

The goal of this project is to accurately classify news articles as either "True" or "Fake". The notebook processes a large dataset of news articles, engineers features, and trains a BERT (DistilBERT) model:

An advanced **DistilBERT** model, a lighter and faster version of BERT, fine-tuned for the specific task of news classification.

The entire workflow, from data ingestion to a real-time prediction function, is demonstrated.

## 🛠️ Core Technologies

-   **Big Data Processing**: Apache Spark (PySpark)
-   **Machine Learning**: Spark MLlib, Scikit-learn
-   **Deep Learning / NLP**: PyTorch, Hugging Face Transformers (DistilBERT)
-   **Data Manipulation**: Pandas, NumPy
-   **Environment**: Google Colab with GPU acceleration

## ⚙️ Installation

To run this project, first clone the repository and then install the required dependencies.

```bash
git clone [https://github.com/shdushantha/fake-news-detection.git](https://github.com/shdushantha/fake-news-detection.git)
cd your-repository
```

## 🚀 Usage
1. Download the Dataset: This project requires the Fake.csv and True.csv files. You can find a version of this dataset on Kaggle.
2. Update File Paths: In the notebook, update the paths to Fake.csv and True.csv to point to their location in your environment. The notebook is configured to use Google Drive, but you can modify it for local paths.
3. Run the Notebook: Open and run the FakeNewsDetection_BigData.ipynb notebook in a Jupyter environment (like Jupyter Lab or Google Colab). A GPU is highly recommended for fine-tuning the BERT model.

## 📊 Project Workflow

The notebook follows these key steps:

1. Setup: Initializes a Spark Session to handle big data processing.
2. Data Loading & Preparation:
    Loads Fake.csv and True.csv into Spark DataFrames.
    Assigns binary labels (1 for fake, 0 for true).
    Merges the two datasets and combines the title and text columns into a single full_text feature.
   
3. Text Preprocessing: Cleans the text by converting it to lowercase and removing non-alphabetic characters.
4. Train-Test Split: Splits the dataset into an 80% training set and a 20% testing set.
5. Model : Fine-Tuning DistilBERT (Hugging Face):
    The training and testing data are converted to Pandas DataFrames.
    A pre-trained distilbert-base-uncased model and its tokenizer are loaded.
    The model is fine-tuned on the training dataset for 1 epoch.

6. Model Saving: Fine-tuned BERT model is saved to google drive for later use.
   
## ✨ Results

The ensemble model demonstrates outstanding performance, significantly outperforming the baseline and achieving near-perfect classification scores.
Example Usage:

```bash
Model	AUC Score	Final Accuracy
Final Ensemble Model	1.0000	99.89%
```

## 🤖 Inference Function
The notebook includes a function predict_news(news_text) that loads the saved models and classifies any given news text in real-time.

Example Usage:
```bash
# Example 1: A potentially fake news headline
fake_news_sample = "BREAKING: Scientists Discover Unicorns Living in a Hidden Valley in the Andes, Government Trying to Cover it Up."
predict_news(fake_news_sample)

# Example 2: A plausible, true-sounding news headline
true_news_sample = "The Federal Reserve announced today it would hold interest rates steady, citing moderate economic growth and stable inflation figures."
predict_news(true_news_sample)
```
