# 🧠 BBC News Text Classification using LSTM

This project is a Natural Language Processing (NLP) application that classifies news articles from the BBC dataset into one of five categories: **business**, **entertainment**, **politics**, **sport**, or **tech** using a **Long Short Term Memory (LSTM)** built with TensorFlow/Keras.

---

## 📂 Dataset

- **Source**: [BBC Text Classification Dataset](https://www.kaggle.com/datasets/cpmarket/bbc-news)
- **Structure**:
  - `category`: The label (e.g., sport, business, etc.)
  - `text`: The article content

---

## 🧰 Tools & Libraries

- Python 🐍
- TensorFlow / Keras
- NLTK (for stopword removal)
- Scikit-learn (Label Encoding, Train/Test Split, Evaluation)
- Seaborn & Matplotlib (Visualization)

---

## 🧼 Data Preprocessing

✅ The following steps were performed:
- Lowercasing all text
- Stopword removal using NLTK
- Tokenization using Keras Tokenizer
- Sequence padding (post-padding)
- Label encoding of target classes

---

## 🧠 Model Architecture

| Layer         | Description                                      |
|---------------|--------------------------------------------------|
| Embedding     | Converts word indices into dense vectors         |
| Dropout       | Reduces overfitting                              |
| Bidirectional | Learns sequential word dependencies              |
| Dense (ReLU)  | Fully connected hidden layer                     |
| Dense (Softmax)| Final output layer with 5-class probabilities   |




<img width="653" height="682" alt="لقطة شاشة 2025-07-13 074618" src="https://github.com/user-attachments/assets/bab3afb0-d957-4d50-8288-7fcb6484be7b" />

```python
Sequential([
    (Embedding(vocab_size, embedding_dim))
    (Bidirectional(LSTM(64, return_sequences=True)))
    (GlobalMaxPool1D())
    (Dense(64, activation='relu'))
    (Dropout(0.5))
    (Dense(5, activation='softmax'))
])


Loss Function: sparse_categorical_crossentropy

Optimizer: Adam

Metrics: accuracy

Epochs: 20 (can be tuned)

Batch Size: 32

Train/Test Split: 80/20

📈 Evaluation
Accuracy on test set: 🟢 93 %

Classification Report:

precision    recall  f1-score   support

     business       0.92      0.99      0.95        74
entertainment       0.98      0.90      0.94        50
     politics       0.86      0.90      0.88        41
        sport       1.00      0.96      0.98        57
         tech       0.92      0.90      0.91        52

     accuracy                           0.94       274
    macro avg       0.94      0.93      0.93       274
 weighted avg       0.94      0.94      0.94       274

Precision, Recall, and F1-score per class

Confusion Matrix:

Visualized with Seaborn heatmap
```

<img width="679" height="566" alt="لقطة شاشة 2025-07-13 074601" src="https://github.com/user-attachments/assets/1b9621ba-47ea-4ad6-b4ab-b0cb057f1493" />

