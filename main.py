import re
import nltk
import string
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

svm_model = joblib.load('svm_linear_sentiment_model_pipeline.pkl')
vectorizer = joblib.load('svm_vectorizer.pkl')

class TextInput(BaseModel):
    text: str

def preprocess_text(text):
     
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\b(?:title|text)\b', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    processed_text = ' '.join(tokens)
    return processed_text


def predict_disease(text):
    processed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([processed_text])
    prediction = svm_model.predict(text_vectorized)[0]
    return prediction

@app.post("/predict_disease")
async def predict_disease_endpoint(text_input: TextInput):
    predicted_class = predict_disease(text_input.text)
    return {"predicted_disease_class": predicted_class}
