import os
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

def evaluate_model():
    # Load dataset
    print("Loading test dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run download_data.py first.")
    
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['label', 'text'])
    
    X = df['text']
    y = df['label']

    # Recreate the exact same split as train.py using random_state=42
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load artifacts
    print("Loading saved model and vectorizer...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Model or Vectorizer file missing. Run train.py first.")
    
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)

    # Transform data
    print("Vectorizing test data...")
    X_test_vectorized = vectorizer.transform(X_test)

    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test_vectorized)

    # Calculate metrics
    # positive label is 'spam'
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='spam')
    rec = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    
    # Confusion matrix: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        }
    }

    # Print metrics to console
    print("\n--- Model Evaluation Summary ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f} (When predicting Spam, how often is it correct?)")
    print(f"Recall:    {rec:.4f} (How much actual Spam did we catch?)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("--------------------------------\n")

    # Save metrics to JSON
    print(f"Saving metrics to {METRICS_PATH}...")
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved successfully.")

if __name__ == "__main__":
    evaluate_model()
