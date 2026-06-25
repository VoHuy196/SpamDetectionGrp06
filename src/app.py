import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import SpamPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Spam Detection API",
    description="A simple MLOps FastAPI service to classify SMS messages as Spam or Ham.",
    version="1.0.0"
)

# Instantiate predictor at module level for simplicity
try:
    predictor = SpamPredictor()
except Exception as e:
    # Fallback if model files are not trained yet (e.g. in CI environment before training)
    predictor = None
    print(f"Warning: Predictor could not be initialized: {e}")

# Request and Response schemas
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    prediction: str
    spam_probability: float
    ham_probability: float

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "api_name": "Spam Detection MLOps API",
        "version": "1.0.0",
        "model_loaded": predictor is not None
    }

@app.post("/predict", response_model=PredictResponse)
def predict_spam(request: PredictRequest):
    global predictor
    # Try to load if it wasn't loaded (lazy load fallback)
    if predictor is None:
        try:
            predictor = SpamPredictor()
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Model files not available. Please run training first. Error: {e}"
            )
            
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
        
    try:
        result = predictor.predict(request.text)
        return PredictResponse(
            text=result["text"],
            prediction=result["prediction"],
            spam_probability=result["spam_probability"],
            ham_probability=result["ham_probability"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

if __name__ == "__main__":
    import uvicorn
    # Run locally if executed directly
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
