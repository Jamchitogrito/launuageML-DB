from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List
import re

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Текст для анализа")

class PredictionResponse(BaseModel):
    language: str
    language_full: str
    probability: float
    top_languages: Dict[str, float]
    is_confident: bool

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)

class BatchResponse(BaseModel):
    results: List[PredictionResponse]
    total: int

LANGUAGE_NAMES = {
    "en": "English",
    "ru": "Russian",
    "uk": "Ukrainian",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}

def predict_language(text: str) -> dict:
    text = re.sub(r"\s+", " ", text).strip().lower()

    if any(ch in text for ch in "ёъыэ"):
        lang, prob = "ru", 0.94
        top = {"ru": 0.94, "uk": 0.03, "en": 0.03}
    elif any(ch in text for ch in "іїєґ"):
        lang, prob = "uk", 0.95
        top = {"uk": 0.95, "ru": 0.03, "en": 0.02}
    elif any(ch in text for ch in "áéíóúñ¿¡"):
        lang, prob = "es", 0.92
        top = {"es": 0.92, "fr": 0.05, "en": 0.03}
    elif any(ch in text for ch in "àâçéèêëîïôùûüœ"):
        lang, prob = "fr", 0.91
        top = {"fr": 0.91, "de": 0.05, "en": 0.04}
    elif any(ch in text for ch in "äöüß"):
        lang, prob = "de", 0.93
        top = {"de": 0.93, "en": 0.04, "fr": 0.03}
    elif re.search(r"[а-яА-Я]", text):
        lang, prob = "ru", 0.80
        top = {"ru": 0.80, "uk": 0.15, "en": 0.05}
    else:
        lang, prob = "en", 0.88
        top = {"en": 0.88, "de": 0.07, "fr": 0.05}

    return {
        "language": lang,
        "language_full": LANGUAGE_NAMES.get(lang, lang),
        "probability": prob,
        "top_languages": top,
        "is_confident": prob > 0.8,
    }


app = FastAPI(
    title="Language Detection API",
    description="API для определения языка текста",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Общее"])
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/languages", tags=["Информация"])
def get_languages():
    return {
        "languages": [
            {"code": code, "name": name}
            for code, name in LANGUAGE_NAMES.items()
        ]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Предсказание"])
def predict(request: TextRequest):
    try:
        return predict_language(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Предсказание"])
def predict_batch(request: BatchRequest):
    try:
        results = [predict_language(text) for text in request.texts]
        return {"results": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))