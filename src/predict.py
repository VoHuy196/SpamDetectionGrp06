import os
import sys
import joblib

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.joblib")

class SpamPredictor:
    def __init__(self, model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
        # Verify if model files exist
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Model or Vectorizer file missing. Please run train.py first.")
        
        # Load model and vectorizer
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.classes = self.model.classes_ # returns e.g. ['ham', 'spam']

    def predict(self, text: str):
        # Convert text to TF-IDF vector
        vectorized_text = self.vectorizer.transform([text])
        
        # Predict class label ('spam' or 'ham')
        prediction = self.model.predict(vectorized_text)[0]
        
        # Predict probability
        probabilities = self.model.predict_proba(vectorized_text)[0]
        
        # Map probabilities to classes
        prob_dict = {self.classes[i]: float(probabilities[i]) for i in range(len(self.classes))}
        
        return {
            "text": text,
            "prediction": prediction,
            "spam_probability": prob_dict.get("spam", 0.0),
            "ham_probability": prob_dict.get("ham", 0.0)
        }

if __name__ == "__main__":
    # If run as CLI script
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"<text_to_classify>\"")
        sys.exit(1)
        
    input_text = sys.argv[1]
    
    try:
        predictor = SpamPredictor()
        result = predictor.predict(input_text)
        print("\n--- Inference Result ---")
        print(f"Input Text:  \"{result['text']}\"")
        print(f"Prediction:  {result['prediction'].upper()}")
        print(f"Spam Prob:   {result['spam_probability']:.4f}")
        print(f"Ham Prob:    {result['ham_probability']:.4f}")
        print("------------------------\n")
    except Exception as e:
        print(f"Error during inference: {e}")
