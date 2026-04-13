# Technical Documentation: Sentiment Analysis Web App

This document provides a comprehensive overview of the technical architecture, algorithms, and implementation logic used in the Sentiment Analysis Web Application.

---

## 1. Technology Stack

### Backend
- **FastAPI**: A high-performance Python web framework used to serve the analysis endpoints.
- **Uvicorn**: An ASGI web server implementation for Python, used to run the FastAPI application.
- **Scikit-learn**: Used for the traditional Machine Learning pipeline (TF-IDF + Logistic Regression).
- **HuggingFace Transformers**: Used to load and inference the BERT (Bidirectional Encoder Representations from Transformers) model.
- **PyTorch**: The underlying deep learning framework for BERT inference.
- **Datasets (HuggingFace)**: Used in the training script to fetch real-world IMDB movie reviews.

### Frontend
- **HTML5**: Semantic structure of the application.
- **Vanilla CSS3**: Implements a **Dynamic Theme System** (Light/Dark mode) using CSS Custom Properties (Variables) and data-attributes.
- **Vanilla JavaScript (ES6+)**: Handles asynchronous API calls (using `fetch`), DOM manipulation, and persistence of theme preferences in LocalStorage.

---

## 2. Machine Learning Algorithms & Models

The application employs two distinct approaches to sentiment analysis to provide comparative insights.

### Approach A: TF-IDF + Logistic Regression
- **Vectorization (TF-IDF)**: Converts text into numerical vectors, emphasizing unique words and de-emphasizing common stop words.
- **Preprocessing Pipeline**: Every input is stripped of HTML tags, URLs, and extra whitespace before analysis to ensure accuracy.
- **Neutrality Guard**: If the model confidence is **< 60%**, the result is labeled as **Neutral**. This prevents the model from forcing a guess on non-opinionated text (like "Hello").

### Approach B: BERT (Transformers) - Optimized
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`. A transformer-based model that uses self-attention to understand context bidirectionally.
- **Optimized Word Importance**: Since running BERT is computationally expensive, the importance calculation now:
  1. Filters out common stop-words (*the, is, an*).
  2. Limits analysis to the 15 most meaningful words in the sentence.
  3. Uses a perturbation-based "Leave-One-Out" strategy to measure confidence drops.

---

## 3. UI/UX Design System

### Theme Management
The application features a **Dual-Theme system**:
- **Light Mode**: A clean, minimalist "SaaS-style" aesthetic with high readability.
- **Dark Mode**: A professional, deep-navy theme optimized for low-light environments.
- **Persistence**: Your choice is saved in `localStorage`, so the app remembers your preference even after a refresh.

### Visualizations
- **Horizontal Bar Charts**: Dynamically generated bars that scale proportionally based on the most influential word in the set.
- **Mood Emojis**: Dynamic badges that change based on sentiment (😊 Positive, 😠 Negative, 😐 Neutral).
- **Robust Error Handling**: Each model card has its own error state. If one model fails to respond, the other will still display its results, ensuring a "graceful failure" experience.

---

## 4. API & Directory Structure

### Endpoint Logic
1. `POST /predict/tfidf`: Returns sentiment analysis via the Scikit-learn pipeline.
2. `POST /predict/bert`: Returns high-accuracy sentiment via the Transformer model.

### Directory Overview
```
SentimentAnalysis/
├── main.py                    # Backend server & Inference logic
├── train_tfidf_real.py        # Standalone script to train on IMDB data
├── index.html                 # Frontend structure & Theme toggle
├── style.css                  # Design system & CSS Variables
├── script.js                  # Async API logic & Dynamic UI
├── models/                    # Saved .pkl models and BERT checkpoints
└── README.md                  # Project overview and setup
```
