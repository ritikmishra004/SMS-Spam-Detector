# 📩 SMS Spam Detection System

This project is a machine learning based SMS spam classifier 
that predicts whether a message is Spam or Not Spam (Ham).

The model uses TF-IDF vectorization and supervised learning algorithms 
to classify text messages.

## 🧠 Problem Statement

Spam messages cause financial loss and security risks.
The goal of this project is to build a text classification model 
that can automatically detect spam messages.

## 📊 Dataset

- Dataset: SMS Spam Collection Dataset
- Total Messages: 5572
- Ham: ~87%
- Spam: ~13%

## 🔎 Exploratory Data Analysis

- Spam messages are generally longer.
- Strong correlation between number of words and characters.
- Common spam words: free, win, claim, prize, urgent.
- Dataset is imbalanced.

## ⚙️ Model Training

- Text Preprocessing (Lowercase, Tokenization, Stopwords Removal, Stemming)
- TF-IDF Vectorization
- Model Used: Multinomial Naive Bayes

## 📈 Model Performance

- Accuracy: 97%
- Precision (Spam): 100%

## 🛠 Tech Stack

- Python
- Scikit-learn
- NLTK
- Streamlit
- Pandas

## 💻 How to Run Locally

cd sms-spam-detection
pip install -r requirements.txt
streamlit run frontend/app.py

## 👨‍💻 Author

Ritik Mishra
Aspiring Data Science Engineer