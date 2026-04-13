import os
import pickle
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import re

# Optimized Text Preprocessing
def clean_text(text: str) -> str:
    text = re.sub(r'<.*?>', '', text) # Remove HTML
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\s+', ' ', text).strip() # Multi-space to single
    return text

STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "with", "and", "or", "but", "if", "then", "this", "that"}

app = FastAPI(title="Sentiment Analysis API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from contextlib import asynccontextmanager

# Global models dictionary for cleaner access
models = {
    "tfidf_vec": None,
    "lr": None,
    "bert_tok": None,
    "bert_mod": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load TF-IDF + LR
    try:
        with open("models/tfidf_vectorizer.pkl", "rb") as f:
            models["tfidf_vec"] = pickle.load(f)
        with open("models/logistic_regression.pkl", "rb") as f:
            models["lr"] = pickle.load(f)
        print("✅ TF-IDF models loaded successfully")
    except Exception as e:
        print(f"⚠️ Error loading TF-IDF models: {e}")

    # Load BERT
    try:
        model_path = "models/bert_sentiment"
        models["bert_tok"] = AutoTokenizer.from_pretrained(model_path)
        models["bert_mod"] = AutoModelForSequenceClassification.from_pretrained(model_path)
        models["bert_mod"].eval()
        print("✅ BERT model loaded successfully")
    except Exception as e:
        print(f"⚠️ Error loading BERT model: {e}")
    
    yield
    # Cleanup logic (if any) would go here

app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

def get_tfidf_top_words(text, vectorizer, model, top_n=5):
    # Vectorize the text
    X = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Get feature importance for this specific text
    row = X.getrow(0)
    indices = row.indices
    data = row.data
    
    # Logistic Regression coefficients for Positive class
    coef = model.coef_[0]
    
    contributions = []
    for idx, val in zip(indices, data):
        word = feature_names[idx]
        score = val * coef[idx]
        contributions.append({"word": word, "score": float(score)})
    
    # Sort by absolute score (importance)
    contributions.sort(key=lambda x: abs(x["score"]), reverse=True)
    return contributions[:top_n]

def get_bert_top_words(text, tokenizer, model, top_n=5):
    # Base prediction
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        base_probs = F.softmax(outputs.logits, dim=1)
        base_label_idx = torch.argmax(base_probs, dim=1).item()
        base_conf = base_probs[0][base_label_idx].item()

    words = [w for w in text.split() if w.lower() not in STOP_WORDS]
    if not words:
        words = text.split()[:10] # Fallback to first 10 if all are stop words
    
    if not words: return []

    importance = []
    # Leave One Out (LOO) importance - Optimized
    for i in range(min(len(words), 15)): # Limit to top 15 words to keep it fast
        temp_words = words[:i] + words[i+1:]
        temp_text = " ".join(temp_words)
        
        inputs_temp = tokenizer(temp_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs_temp = model(**inputs_temp)
            probs_temp = F.softmax(outputs_temp.logits, dim=1)
            # Change in probability of the predicted label
            conf_change = base_conf - probs_temp[0][base_label_idx].item()
            
            # Sign consistency: 
            # If predicted class is Positive (1), importance helps it be Positive.
            # If predicted class is Negative (0), importance helps it be Negative.
            # We want: Positive Score = Positive Sentiment, Negative Score = Negative Sentiment.
            final_score = conf_change if base_label_idx == 1 else -conf_change
            importance.append({"word": words[i], "score": float(final_score)})
    
    importance.sort(key=lambda x: abs(x["score"]), reverse=True)
    return importance[:top_n]

@app.post("/predict/tfidf")
async def predict_tfidf(request: PredictRequest):
    cleaned_text = clean_text(request.text)
    if not cleaned_text:
        return {"label": "Neutral", "confidence": 0, "top_words": []}
        
    if models["tfidf_vec"] is None or models["lr"] is None:
        raise HTTPException(status_code=500, detail="TF-IDF models not loaded")
    
    X = models["tfidf_vec"].transform([cleaned_text])
    prob = models["lr"].predict_proba(X)[0]
    label_idx = np.argmax(prob)
    confidence = float(prob[label_idx])
    
    # Custom threshold for Neutrality
    if confidence < 0.6:
        label = "Neutral"
    else:
        label = "Positive" if label_idx == 1 else "Negative"
    
    return {
        "label": label,
        "confidence": confidence,
        "top_words": get_tfidf_top_words(cleaned_text, models["tfidf_vec"], models["lr"])
    }

@app.post("/predict/bert")
async def predict_bert(request: PredictRequest):
    cleaned_text = clean_text(request.text)
    if not cleaned_text:
        return {"label": "Neutral", "confidence": 0, "top_words": []}

    if models["bert_tok"] is None or models["bert_mod"] is None:
        raise HTTPException(status_code=500, detail="BERT model not loaded")
    
    inputs = models["bert_tok"](cleaned_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = models["bert_mod"](**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        label_idx = torch.argmax(probs, dim=1).item()
        confidence = float(probs[0][label_idx])
    
    if confidence < 0.6:
        label = "Neutral"
    else:
        label = "Positive" if label_idx == 1 else "Negative"
        
    top_words = get_bert_top_words(cleaned_text, models["bert_tok"], models["bert_mod"])
    
    return {
        "label": label,
        "confidence": confidence,
        "top_words": top_words
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
