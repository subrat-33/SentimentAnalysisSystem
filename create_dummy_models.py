import os
import pickle
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def create_dummy_tfidf():
    print("Creating dummy TF-IDF models...")
    texts = ["I love this", "This is bad", "Happy day", "Sad moment"]
    labels = [1, 0, 1, 0]
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    
    model = LogisticRegression()
    model.fit(X, labels)
    
    os.makedirs("models", exist_ok=True)
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("models/logistic_regression.pkl", "wb") as f:
        pickle.dump(model, f)
    print("TF-IDF models saved.")

def create_dummy_bert():
    print("Downloading/Saving base BERT model as dummy...")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    save_path = "models/bert_sentiment"
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print("BERT model saved.")

if __name__ == "__main__":
    create_dummy_tfidf()
    # BERT might take a while to download, so I'll leave it as an option or just do it
    try:
        create_dummy_bert()
    except Exception as e:
        print(f"Skipping BERT dummy creation (requires internet/time): {e}")
