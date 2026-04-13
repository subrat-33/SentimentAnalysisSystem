# Sentiment Insight | Dual-Model Sentiment Analysis

A professional-grade sentiment analysis web application featuring a split-model architecture (Traditional ML vs. Transformer Deep Learning). This tool provides real-time analysis, word importance visualization, and a modern, adaptive UI.

## ✨ Features
- **Dual-Approach Analysis**: Compare results from **TF-IDF + Logistic Regression** (speed) and **Fine-tuned BERT** (accuracy) simultaneously.
- **Dynamic Theme System**: Seamless toggle between professional **Light Mode** and sleek **Dark Mode**.
- **Neutral Sentiment Mapping**: Advanced threshold logic (60% confidence) to correctly identify neutral text like greetings and facts.
- **Explainable AI (XAI)**: Visualize exactly which words pushed the sentiment in a specific direction using dynamic bar charts.
- **Real-World Training**: Includes a script to train your localized model on the **IMDB Movie Reviews** dataset (10k samples).

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the Models (Choose One)
*   **The Pro Way (Recommended)**: Train the TF-IDF model on 10,000 real movie reviews:
    ```bash
    python train_tfidf_real.py
    ```
*   **The Quick Way**: Create tiny placeholder models for immediate UI testing:
    ```bash
    python create_dummy_models.py
    ```

### 3. Launch the Backend
```bash
python main.py
```
The API serves at `http://localhost:8000`.

### 4. Open the App
Open `index.html` in any modern web browser.

## 📁 Project Structure
- `main.py`: FastAPI backend handling inference and text cleaning.
- `train_tfidf_real.py`: Training script for commercial-grade accuracy.
- `index.html` / `style.css` / `script.js`: Clean, responsive frontend with theme management.
- `models/`: Storage for pickled models and BERT checkpoints.
- `IMPLEMENTATION_DETAILS.md`: Deep-dive technical documentation.

## 🛠️ Technology Stack
- **FastAPI / Uvicorn** (Backend)
- **Scikit-learn** (Classical ML)
- **HuggingFace Transformers & PyTorch** (Deep Learning)
- **Vanilla CSS3 & JS (ES6+)** (Frontend)
