from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for analyzing sentiment in Spanish Amazon reviews",
    version="1.0.0"
)

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

try:
    model = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.joblib"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    model_config = joblib.load(os.path.join(MODELS_DIR, "model_config.joblib"))
    preprocess_config = joblib.load(os.path.join(MODELS_DIR, "preprocess_config.joblib"))
except FileNotFoundError as e:
    raise RuntimeError(f"Failed to load model files: {e}")


class TextInput(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float


@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(input_data: TextInput):
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    text_vectorized = vectorizer.transform([input_data.text])
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]
    confidence = float(max(probabilities))
    
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    sentiment = sentiment_map.get(prediction, "unknown")
    
    return SentimentResponse(
        text=input_data.text,
        sentiment=sentiment,
        confidence=confidence
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)