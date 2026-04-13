import os
import pickle
import re
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower().strip()
    return text

def train_on_imdb():
    print("Step 1: Downloading IMDB Dataset (this might take a minute)...")
    dataset = load_dataset("imdb")
    
    # We'll use 10,000 samples for training to keep it fast but accurate
    # and 2,000 for testing
    train_data = dataset['train'].shuffle(seed=42).select(range(10000))
    test_data = dataset['test'].shuffle(seed=42).select(range(2000))
    
    print("Step 2: Cleaning text data...")
    X_train = [clean_text(x) for x in train_data['text']]
    y_train = train_data['label']
    
    X_test = [clean_text(x) for x in test_data['text']]
    y_test = test_data['label']
    
    print(f"Step 3: Training TF-IDF + Logistic Regression on {len(X_train)} samples...")
    # Using a pipeline to keep vectorizer and model linked
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    model = LogisticRegression(max_iter=1000)
    
    X_train_vec = vectorizer.fit_transform(X_train)
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Training Complete! Test Accuracy: {acc:.2%}")
    
    print("Step 4: Saving models to the 'models' directory...")
    os.makedirs("models", exist_ok=True)
    
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        
    with open("models/logistic_regression.pkl", "wb") as f:
        pickle.dump(model, f)
        
    print("🚀 Done! Your TF-IDF model is now trained on real-world movie reviews.")
    print("Please restart your FastAPI server (main.py) to load the new models.")

if __name__ == "__main__":
    train_on_imdb()
