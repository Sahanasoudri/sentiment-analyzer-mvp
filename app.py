from flask import Flask, render_template, request
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)

model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    review_text = ""

    if request.method == "POST":
        review_text = request.form["review"]
        cleaned = clean_text(review_text)
        vectorized = vectorizer.transform([cleaned])
        result = model.predict(vectorized)[0]
        prediction = result

    return render_template(
        "index.html",
        prediction=prediction,
        review=review_text
    )

if __name__ == "__main__":
    app.run(debug=True)
