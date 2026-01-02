"""
SentimentAPI - API de Análisis de Sentimientos en Español
=========================================================

API REST para clasificación de sentimientos en reseñas de Amazon en español.
Clasificaciones: Positivo, Neutro, Negativo

Sistema Homologado:
- Sentimiento: Positivo / Neutro / Negativo
- Estrellas: 1-5 ⭐
- Confidence Score: 0.0-1.0

Ejecutar con: uvicorn main:app --reload --port 8000
Documentación: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import re
import os
from typing import Optional, Dict
from datetime import datetime
import logging
import nltk
from nltk.stem.snowball import SnowballStemmer

# Descargar recursos de NLTK (solo la primera vez)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURACIÓN DE LA API
# =============================================================================

app = FastAPI(
    title="SentimentAPI - Amazon ES",
    description="API de Análisis de Sentimientos para reseñas de Amazon en español (Sistema Homologado)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS para permitir peticiones desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# RUTAS DE ARCHIVOS DEL MODELO
# =============================================================================

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.joblib")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.joblib")

# Variables globales para el modelo
vectorizer = None
model = None
model_config = None

# Estadísticas de uso
stats = {
    "total_requests": 0,
    "positive_count": 0,
    "negative_count": 0,
    "neutral_count": 0,
    "avg_confidence": 0.0,
    "start_time": datetime.now().isoformat()
}

# =============================================================================
# FUNCIONES DE HOMOLOGACIÓN
# =============================================================================

def confidence_to_stars(confidence: float) -> int:
    """Convierte confidence score (0-1) a estrellas (1-5)."""
    if confidence >= 0.80:
        return 5
    elif confidence >= 0.60:
        return 4
    elif confidence >= 0.40:
        return 3
    elif confidence >= 0.20:
        return 2
    else:
        return 1

def confidence_to_sentiment(confidence: float) -> str:
    """Convierte confidence score a sentimiento."""
    if confidence >= 0.60:
        return "Positivo"
    elif confidence >= 0.40:
        return "Neutro"
    else:
        return "Negativo"

# =============================================================================
# FUNCIÓN DE PREPROCESAMIENTO (IDÉNTICA AL NOTEBOOK)
# =============================================================================

# Inicializar stemmer (mismo que el notebook)
stemmer = SnowballStemmer('spanish')

def preprocess_text(text: str) -> str:
    """
    Preprocesa texto en español para análisis de sentimientos.
    IMPORTANTE: Esta función debe ser IDÉNTICA a la del notebook.
    """
    if not text:
        return ""
    
    # Convertir a minúsculas
    text = str(text).lower()
    
    # Eliminar URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Eliminar menciones y hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Eliminar caracteres especiales (mantener letras españolas)
    text = re.sub(r'[^\w\sáéíóúüñ¡¿]', '', text)
    
    # Eliminar números
    text = re.sub(r'\d+', '', text)
    
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Stopwords en español (mismas que el notebook)
    stop_words = {
        'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las',
        'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'es', 'lo', 'como',
        'más', 'pero', 'sus', 'le', 'ya', 'o', 'fue', 'este', 'ha', 'si', 'porque',
        'esta', 'son', 'entre', 'está', 'cuando', 'muy', 'sin', 'sobre', 'ser',
        'tiene', 'también', 'me', 'hasta', 'hay', 'donde', 'han', 'quien',
        'amazon', 'producto', 'compra', 'artículo', 'envío'
    }
    
    # Tokenizar y filtrar
    tokens = text.split()
    
    # Aplicar stemming (CRÍTICO: igual que el notebook)
    tokens = [
        stemmer.stem(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 2
    ]
    
    return ' '.join(tokens)

# =============================================================================
# MODELOS PYDANTIC (VALIDACIÓN)
# =============================================================================

class TextInput(BaseModel):
    """Modelo de entrada para el análisis de sentimiento."""
    text: str = Field(..., min_length=1, max_length=5000, 
                      description="Texto a analizar (reseña, comentario)")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('El texto no puede estar vacío')
        return v.strip()

class SentimentResponse(BaseModel):
    """Modelo de respuesta del análisis de sentimiento con homologación."""
    prevision: str = Field(..., description="Sentimiento predicho: Positivo, Neutro o Negativo")
    probabilidad: float = Field(..., description="Probabilidad de la clase predicha (0-1)")
    confidence_score: float = Field(..., description="Score de confianza normalizado (0-1)")
    estrellas: int = Field(..., description="Equivalente en estrellas (1-5)")
    probabilidades: Dict[str, float] = Field(..., description="Probabilidades por clase")

class HealthResponse(BaseModel):
    """Modelo de respuesta del health check."""
    status: str
    model_loaded: bool
    version: str

class StatsResponse(BaseModel):
    """Modelo de respuesta de estadísticas."""
    total_requests: int
    positive_count: int
    negative_count: int
    neutral_count: int
    positive_percentage: float
    avg_confidence: float
    start_time: str

# =============================================================================
# FUNCIONES DE CARGA DEL MODELO
# =============================================================================

def load_model():
    """Carga el modelo y vectorizador al iniciar la API."""
    global vectorizer, model, model_config
    
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        
        if os.path.exists(CONFIG_PATH):
            model_config = joblib.load(CONFIG_PATH)
        
        logger.info("✅ Modelo cargado exitosamente")
        logger.info(f"   Clases: {model.classes_.tolist()}")
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")
        raise

# =============================================================================
# EVENTOS DE CICLO DE VIDA
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Evento ejecutado al iniciar la API."""
    logger.info("🚀 Iniciando SentimentAPI v2.0...")
    load_model()
    logger.info("✅ API lista para recibir peticiones")

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "nombre": "SentimentAPI - Amazon ES",
        "version": "2.0.0",
        "descripcion": "API de Análisis de Sentimientos para reseñas de Amazon en español",
        "sistema_homologado": {
            "sentimiento": ["Positivo", "Neutro", "Negativo"],
            "estrellas": "1-5",
            "confidence_score": "0.0-1.0"
        },
        "endpoints": {
            "POST /sentiment": "Analizar sentimiento de un texto",
            "GET /health": "Estado de la API",
            "GET /stats": "Estadísticas de uso"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Verifica el estado de la API y del modelo."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="2.0.0"
    )

@app.get("/stats", response_model=StatsResponse, tags=["General"])
async def get_stats():
    """Obtiene estadísticas de uso de la API."""
    total = stats["total_requests"]
    pos_pct = (stats["positive_count"] / total * 100) if total > 0 else 0
    
    return StatsResponse(
        total_requests=total,
        positive_count=stats["positive_count"],
        negative_count=stats["negative_count"],
        neutral_count=stats["neutral_count"],
        positive_percentage=round(pos_pct, 2),
        avg_confidence=round(stats["avg_confidence"], 4),
        start_time=stats["start_time"]
    )

@app.post("/sentiment", response_model=SentimentResponse, tags=["Predicción"])
async def analyze_sentiment(input_data: TextInput):
    """
    Analiza el sentimiento de un texto en español.
    
    Sistema Homologado - Devuelve:
    - **prevision**: Sentimiento (Positivo, Neutro, Negativo)
    - **probabilidad**: Probabilidad de la clase predicha
    - **confidence_score**: Score normalizado (0-1)
    - **estrellas**: Equivalente en estrellas (1-5)
    - **probabilidades**: Desglose por clase
    """
    global stats
    
    # Verificar que el modelo está cargado
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=500, 
            detail="Modelo no cargado. Reinicie la API."
        )
    
    # Preprocesar el texto
    processed_text = preprocess_text(input_data.text)
    
    # Verificar que el texto procesado no esté vacío
    if not processed_text.strip():
        raise HTTPException(
            status_code=400,
            detail="El texto no contiene palabras significativas después del preprocesamiento."
        )
    
    # Vectorizar y predecir
    text_vectorized = vectorizer.transform([processed_text])
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]
    
    # Obtener todas las probabilidades por clase
    proba_dict = {cls: round(float(prob), 4) for cls, prob in zip(model.classes_, probabilities)}
    
    # Calcular confidence score ponderado
    prob_positivo = proba_dict.get('Positivo', 0)
    prob_neutro = proba_dict.get('Neutro', 0)
    prob_negativo = proba_dict.get('Negativo', 0)
    
    # Fórmula: Confidence Score = (P_positivo * 1.0) + (P_neutro * 0.5) + (P_negativo * 0.0)
    confidence_score = (prob_positivo * 1.0) + (prob_neutro * 0.5) + (prob_negativo * 0.0)
    confidence_score = round(confidence_score, 4)
    
    # Convertir a estrellas
    estrellas = confidence_to_stars(confidence_score)
    
    # Obtener probabilidad de la clase predicha
    class_idx = list(model.classes_).index(prediction)
    probability = probabilities[class_idx]
    
    # Actualizar estadísticas
    stats["total_requests"] += 1
    n = stats["total_requests"]
    stats["avg_confidence"] = stats["avg_confidence"] + (confidence_score - stats["avg_confidence"]) / n
    
    if prediction == "Positivo":
        stats["positive_count"] += 1
    elif prediction == "Negativo":
        stats["negative_count"] += 1
    else:
        stats["neutral_count"] += 1
    
    # Log de la predicción
    logger.info(f"Predicción: {prediction} | ⭐{estrellas} | CS:{confidence_score:.2%} - '{input_data.text[:50]}...'")
    
    return SentimentResponse(
        prevision=prediction,
        probabilidad=round(float(probability), 4),
        confidence_score=confidence_score,
        estrellas=estrellas,
        probabilidades=proba_dict
    )

# =============================================================================
# EJECUCIÓN DIRECTA
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
