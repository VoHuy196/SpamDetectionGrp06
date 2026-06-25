import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")

def train_model():
    # Make sure model directory exists
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created model directory: {MODEL_DIR}")

    # Load data
    print(f"Loading dataset from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please run download_data.py first.")
    
    df = pd.read_csv(DATA_PATH)
    
    # Handle missing values
    df = df.dropna(subset=['label', 'text'])
    
    # Convert label to binary or keep as string? 
    # For MultinomialNB, target can remain string 'ham' / 'spam' or binary. 
    # Keeping it as string makes raw model output cleaner. Let's keep it as string.
    X = df['text']
    y = df['label']
    
    print(f"Dataset loaded. Total samples: {len(df)}")
    print(f"Class distribution:\n{y.value_counts()}")

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split data into train size: {len(X_train)} and test size: {len(X_test)}")

    # Vectorize text using TF-IDF
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train Multinomial Naive Bayes model
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)

    # Make predictions and print training performance
    y_pred = model.predict(X_test_vectorized)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model training complete. Test Accuracy: {acc:.4f}")

    # Save artifacts
    print(f"Saving vectorizer to: {VECTORIZER_PATH}")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    print(f"Saving model to: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    print("Artifacts saved successfully.")

if __name__ == "__main__":
    train_model()
