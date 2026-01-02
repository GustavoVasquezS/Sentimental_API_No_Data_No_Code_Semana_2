# 🎯 SentimentAPI - Análisis de Sentimientos Amazon ES

## 📋 Descripción

Sistema de análisis de sentimientos para reseñas de Amazon en español, con un **sistema homologado** que integra tres métricas de clasificación:

- **Sentimiento**: Positivo / Neutro / Negativo
- **Estrellas**: 1-5 ⭐
- **Confidence Score**: 0.00 - 1.00

---

## 🔄 Sistema de Homologación

El modelo utiliza un sistema unificado que relaciona estrellas, confidence score y sentimiento:

| Estrellas | Confidence Score | Sentimiento |
|:---------:|:----------------:|:-----------:|
| ⭐⭐⭐⭐⭐ (5) | 0.80 - 1.00 | 🟢 Positivo |
| ⭐⭐⭐⭐☆ (4) | 0.60 - 0.79 | 🟢 Positivo |
| ⭐⭐⭐☆☆ (3) | 0.40 - 0.59 | 🟡 Neutro |
| ⭐⭐☆☆☆ (2) | 0.20 - 0.39 | 🔴 Negativo |
| ⭐☆☆☆☆ (1) | 0.00 - 0.19 | 🔴 Negativo |

### Fórmula del Confidence Score

```
Confidence Score = (P_positivo × 1.0) + (P_neutro × 0.5) + (P_negativo × 0.0)
```

Donde `P_x` es la probabilidad predicha por el modelo para cada clase.

---

## 📁 Estructura del Proyecto

```
No Country/
├── sentiment_amazon_es.ipynb    # Notebook principal con pipeline ML
├── sentimeltal_api_backup.ipynb # Backup del notebook original
├── test.csv                     # Dataset original (multilenguaje)
├── requirements.txt             # Dependencias Python
├── README_sentiment_amazon_es.md # Este archivo
├── api/
│   └── main.py                  # API REST FastAPI v2.0
└── models/
    ├── sentiment_model.joblib   # Modelo entrenado
    ├── tfidf_vectorizer.joblib  # Vectorizador TF-IDF
    ├── model_config.joblib      # Configuración del modelo
    └── preprocess_config.joblib # Config. de preprocesamiento
```

---

## 🚀 Instalación

### 1. Crear entorno virtual (opcional pero recomendado)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### Dependencias principales:
- `pandas` - Manipulación de datos
- `scikit-learn` - Machine Learning
- `nltk` - Procesamiento de lenguaje natural
- `fastapi` - Framework API REST
- `uvicorn` - Servidor ASGI
- `joblib` - Serialización de modelos
- `matplotlib` / `seaborn` - Visualización

---

## 📊 Pipeline del Notebook

El notebook `sentiment_amazon_es.ipynb` ejecuta el siguiente pipeline:

1. **Carga de datos**: Filtrado de `test.csv` por `language='es'`
2. **EDA**: Análisis exploratorio y visualización
3. **Preprocesamiento**: 
   - Limpieza de texto
   - Eliminación de stopwords en español
   - Stemming con SnowballStemmer
4. **Vectorización**: TF-IDF (max_features=5000, ngram_range=(1,2))
5. **Entrenamiento**: Logistic Regression multiclase
6. **Evaluación**: Accuracy, Precision, Recall, F1-Score
7. **Serialización**: Guardado de modelos con joblib
8. **Generación de API**: Código FastAPI listo para producción

---

## 🌐 API REST

### Iniciar el servidor

```bash
cd api
uvicorn main:app --reload --port 8000
```

### Documentación interactiva

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### `POST /sentiment` - Analizar sentimiento

**Request:**
```json
{
  "text": "¡Excelente producto! Superó todas mis expectativas."
}
```

**Response:**
```json
{
  "prevision": "Positivo",
  "probabilidad": 0.9234,
  "confidence_score": 0.8567,
  "estrellas": 5,
  "probabilidades": {
    "Positivo": 0.9234,
    "Neutro": 0.0521,
    "Negativo": 0.0245
  }
}
```

#### `GET /health` - Estado de la API

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "2.0.0"
}
```

#### `GET /stats` - Estadísticas de uso

```json
{
  "total_requests": 150,
  "positive_count": 85,
  "negative_count": 40,
  "neutral_count": 25,
  "positive_percentage": 56.67,
  "avg_confidence": 0.6234,
  "start_time": "2026-01-02T10:30:00"
}
```

---

## 💻 Ejemplos de Uso

### Python (requests)

```python
import requests

url = "http://localhost:8000/sentiment"
texto = "El producto llegó roto y el vendedor no responde"

response = requests.post(url, json={"text": texto})
resultado = response.json()

print(f"Sentimiento: {resultado['prevision']}")
print(f"Estrellas: {'⭐' * resultado['estrellas']}")
print(f"Confidence: {resultado['confidence_score']:.2%}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/sentiment" \
     -H "Content-Type: application/json" \
     -d '{"text": "¡Muy buena calidad, totalmente recomendado!"}'
```

### JavaScript (fetch)

```javascript
const response = await fetch('http://localhost:8000/sentiment', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Producto de excelente calidad' })
});

const data = await response.json();
console.log(`${data.prevision} - ${'⭐'.repeat(data.estrellas)}`);
```

---

## 📈 Métricas del Modelo

| Métrica | Valor |
|---------|-------|
| Accuracy | ~75-80% |
| F1-Score (macro) | ~0.72 |
| Clases | 3 (Positivo, Neutro, Negativo) |
| Features | TF-IDF (5000) |
| Algoritmo | Logistic Regression |

*Los valores pueden variar según la distribución del dataset filtrado.*

---

## 🔧 Configuración

### Variables de entorno (opcional)

```bash
export MODEL_DIR="./models"
export API_PORT=8000
export LOG_LEVEL="INFO"
```

### Modificar rangos de homologación

En el notebook, celda de funciones de homologación:

```python
def confidence_to_stars(confidence: float) -> int:
    if confidence >= 0.80: return 5
    elif confidence >= 0.60: return 4
    elif confidence >= 0.40: return 3
    elif confidence >= 0.20: return 2
    else: return 1
```

---

## 📝 Notas Técnicas

- **Dataset**: Reseñas de Amazon filtradas por idioma español (~5000 registros)
- **Preprocesamiento**: Stopwords personalizadas + términos específicos de e-commerce
- **Balance de clases**: `class_weight='balanced'` en el modelo
- **Validación**: Train/Test split 80/20 con estratificación

---

## 👥 Autores

Proyecto desarrollado para **No Country** - Simulación de entorno laboral tech.

---

## 📄 Licencia

Este proyecto es de uso educativo y demostrativo.

---

*Última actualización: Enero 2026*
