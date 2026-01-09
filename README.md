# Sentiment Analyzer (MVP Project)

## Project Overview
This project is a Sentiment Analysis web application developed as part of an MVP assessment.
It predicts whether a given movie review expresses a **positive** or **negative** sentiment.

---

## 🧠 Approach
- Text preprocessing using NLTK
- Feature extraction using TF-IDF
- Classification using Logistic Regression
- Model deployment using Flask

---

## Tech Stack
- Python
- Pandas
- Scikit-learn
- NLTK
- Flask
- HTML (Jinja2 templates)

---

##  Project Structure
```
Sentiment_Analyzer/
│── app.py
│── train_model.py
│── requirements.txt
│── README.md
│── model/
│ ├── sentiment_model.pkl
│ └── vectorizer.pkl
│── templates/
│ └── index.html
│── data/
│ └── imdb_sample.csv 
```
---
## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Sahanasoudri/sentiment-analyzer-mvp.git
cd Sentiment_Analyzer
```
### 2️⃣ Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the application
```bash
python app.py
```
Open your browser and go to:
http://127.0.0.1:5000/


## 📊 Model Performance
- Accuracy achieved: **~83%**
- Evaluation metrics used:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report

---

## ✅ Features
- User can enter a movie review
- Application predicts sentiment (Positive / Negative)
- Clean and simple web interface
- Trained model reused without retraining

---

## 📎 Notes
- This project was developed independently as part of an MVP assessment.
- Dataset size was limited to **1000 samples** for faster experimentation.

---

## 👩‍💻 Author
Sahana


