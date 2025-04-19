# 📚 NLP Analysis of *Pride and Prejudice* by Jane Austen

This project applies Natural Language Processing (NLP) techniques to the classic novel *Pride and Prejudice* to extract meaningful insights from its early chapters. The goal is to explore how computational methods can be used to analyze literature in a modern, data-driven way.

---

## ✨ Project Highlights

- **Named Entity Recognition (NER)** to identify characters, locations, dates, and organizations.
- **Sentiment Analysis** of the first five chapters to evaluate tone and subjectivity.
- **Topic Modeling** using LDA to extract main themes.
- **TF-IDF Visualization** with PCA and t-SNE for topic separation.

---

## 🧠 Techniques Used

- **NER**: Spacy
- **Sentiment Analysis**: TextBlob
- **Topic Modeling**: LDA from Gensim
- **TF-IDF**: Scikit-learn
- **Visualization**: Matplotlib & Seaborn (PCA, t-SNE)

---

---

## 📊 Results Summary

### 🔹 Named Entity Recognition:
Key entities like **Elizabeth**, **Bingley**, **Mr. Bennet**, **London**, and **North England** were correctly identified and categorized.

### 🔹 Sentiment Analysis:
- Chapters show a generally **positive** tone with moderate subjectivity.
- Example polarity range: `0.08` to `0.31`.

### 🔹 Topic Modeling (Top 3 Topics):
- **Topic 1**: mr, pride, miss, lucas, bingley  
- **Topic 2**: mr, bennet, visit, bingley, dear  
- **Topic 3**: bingley, darcy, good, miss, people  

### 🔹 TF-IDF Visualization:
- **PCA** and **t-SNE** confirm the semantic separation of topics in a 2D space.

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/PrideAndPrejudice-NLP.git
   cd PrideAndPrejudice-NLP
   


