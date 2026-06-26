import os
import pickle
from src.config import load_config
from src.ingest import ingest
from src.preprocess import preprocess
from src.train import train
from src.evaluate import evaluate
from src.register import register_best_model

if __name__ == "__main__":
    cfg = load_config()
    cfg['data']['raw_path'] = 'data/spam.csv'
    cfg['mlflow']['tracking_uri'] = 'http://localhost:5001'
    
    df = ingest(cfg['data']['url'], cfg['data']['raw_path'])
    X_tr, X_te, y_tr, y_te, vectorizer = preprocess(
        df, cfg['data']['test_size'], cfg['data']['random_state'], cfg['data']['max_features']
    )
    
    os.makedirs('models', exist_ok=True)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    for m in cfg['models']:
        run_id, model = train(X_tr, y_tr, m, cfg['mlflow']['experiment_name'], cfg['mlflow']['tracking_uri'])
        evaluate(model, X_te, y_te, run_id, cfg['mlflow']['tracking_uri'])
        
    register_best_model(cfg['mlflow']['experiment_name'], cfg['mlflow']['tracking_uri'], cfg['mlflow']['registered_model'])
