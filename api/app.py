import os
import pickle
import numpy as np
import redis
import json
import hashlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn

app = FastAPI(title="Spam Detection API", description="API with Redis Caching and Local Explainability")

MODEL_PATH = '/opt/airflow/models'
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5001')
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))

# Initialize Redis Cache (Slide 28: ML Caching Strategy)
try:
    cache = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
except:
    cache = None

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

vectorizer = None
model = None

def load_assets():
    global vectorizer, model
    try:
        with open(f'{MODEL_PATH}/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    except Exception as e:
        print("Could not load vectorizer:", e)

    try:
        model_uri = "models:/spam_detection_model/latest"
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print("Could not load model:", e)

load_assets()

class MessageRequest(BaseModel):
    message: str

class PredictionResponse(BaseModel):
    is_spam: bool
    confidence: float
    cached: bool

class ExplainResponse(BaseModel):
    is_spam: bool
    confidence: float
    explanation: dict

def get_cache_key(message: str) -> str:
    return hashlib.md5(message.encode('utf-8')).hexdigest()

@app.get("/")
def read_root():
    return {"status": "ok", "model_loaded": model is not None, "vectorizer_loaded": vectorizer is not None}

@app.post("/reload")
def reload_model():
    load_assets()
    return {"status": "reloaded", "model_loaded": model is not None, "vectorizer_loaded": vectorizer is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: MessageRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=503, detail="Model or vectorizer not fully loaded")
    
    # 1. Check Cache
    cache_key = get_cache_key(req.message)
    if cache:
        try:
            cached_result = cache.get(cache_key)
            if cached_result:
                result = json.loads(cached_result)
                return {"is_spam": result['is_spam'], "confidence": result['confidence'], "cached": True}
        except Exception as e:
            print("Redis error:", e)
    
    # 2. Run Model if Cache Miss
    vec = vectorizer.transform([req.message]).toarray()
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0].max() if hasattr(model, "predict_proba") else 1.0
    
    result = {"is_spam": bool(pred == 1), "confidence": float(prob), "cached": False}
    
    # 3. Store in Cache
    if cache:
        try:
            cache.setex(cache_key, 3600, json.dumps(result)) # Cache for 1 hour
        except:
            pass
            
    return result

@app.post("/explain", response_model=ExplainResponse)
def explain(req: MessageRequest):
    """
    Local Explanation (Slide 44): Which words contributed most to this specific prediction?
    """
    if not model or not vectorizer:
        raise HTTPException(status_code=503, detail="Model or vectorizer not fully loaded")
        
    vec = vectorizer.transform([req.message]).toarray()
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0].max() if hasattr(model, "predict_proba") else 1.0
    
    feature_names = vectorizer.get_feature_names_out()
    nonzero_indices = vec[0].nonzero()[0]
    word_contributions = {}
    
    # Glass-box interpretability for linear/tree models (Slide 43)
    try:
        if hasattr(model, 'coef_'):
            coef = model.coef_[0] if len(model.coef_) == 1 else model.coef_[pred]
            word_contributions = {feature_names[i]: float(coef[i] * vec[0][i]) for i in nonzero_indices}
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            word_contributions = {feature_names[i]: float(importances[i] * vec[0][i]) for i in nonzero_indices}
        else:
            word_contributions = {"info": "Model type not fully supported for detailed local explanation"}
    except Exception as e:
        word_contributions = {"error": str(e)}
        
    # Sort top 5 contributing words
    sorted_words = dict(sorted(word_contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:5])
    
    return {
        "is_spam": bool(pred == 1),
        "confidence": float(prob),
        "explanation": sorted_words
    }
