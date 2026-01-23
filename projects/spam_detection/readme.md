# 📧 Spam Email Detection using TensorFlow & NLP

A deep learning project that classifies emails as **Spam** or **Ham (Normal)** using Natural Language Processing (NLP) and Long Short-Term Memory (LSTM) networks.

## 🚀 Project Overview
This project uses a Sequential Neural Network to analyze the text content of emails and predict the likelihood of them being unsolicited spam. By using an LSTM layer, the model understands the contextual relationship between words rather than just looking for keywords.

## 🛠️ Tech Stack
* **Language:** Python
* **Editor:** VS Code
* **Deep Learning:** TensorFlow / Keras
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, WordCloud
* **NLP:** NLTK (Natural Language Toolkit)

## 📊 Model Performance
The model achieves high accuracy by processing text through the following pipeline:
1.  **Text Cleaning:** Removal of punctuation and stop words.
2.  **Tokenization:** Converting text into sequences of integers.
3.  **Padding:** Ensuring all input sequences have a uniform length.
4.  **Deep Learning:** * `Embedding Layer`: For word vector representation.
    * `LSTM Layer`: To capture sequential patterns in text.
    * `Dense Layers`: For final classification via Sigmoid activation.

**Final Test Accuracy:** ~93.8%

## 📂 Project Structure
```text
├── spam_detection.ipynb   # Main Jupyter Notebook
├── Emails.csv             # Dataset file
├── spam_model.h5          # Saved trained model
└── README.md              # Project documentation
