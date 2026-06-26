import os
import pickle
import pandas as pd
from src.config import load_config
from src.ingest import ingest
from src.preprocess import preprocess
from src.train import train
from src.evaluate import evaluate
from src.register import register_best_model

MODEL_PATH = '/opt/airflow/models'

def task_ingest(**context):
    cfg = load_config()
    ingest(cfg['data']['url'], cfg['data']['raw_path'])

def task_preprocess(**context):
    cfg = load_config()
    df = pd.read_csv(cfg['data']['raw_path'])
    X_tr, X_te, y_tr, y_te, vectorizer = preprocess(
        df, cfg['data']['test_size'], cfg['data']['random_state'], cfg['data']['max_features']
    )
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    with open(f'{MODEL_PATH}/X_tr.pkl', 'wb') as f: pickle.dump(X_tr, f)
    with open(f'{MODEL_PATH}/X_te.pkl', 'wb') as f: pickle.dump(X_te, f)
    with open(f'{MODEL_PATH}/y_tr.pkl', 'wb') as f: pickle.dump(y_tr, f)
    with open(f'{MODEL_PATH}/y_te.pkl', 'wb') as f: pickle.dump(y_te, f)
    with open(f'{MODEL_PATH}/vectorizer.pkl', 'wb') as f: pickle.dump(vectorizer, f)

def task_train_and_evaluate(**context):
    cfg = load_config()
    with open(f'{MODEL_PATH}/X_tr.pkl', 'rb') as f: X_tr = pickle.load(f)
    with open(f'{MODEL_PATH}/X_te.pkl', 'rb') as f: X_te = pickle.load(f)
    with open(f'{MODEL_PATH}/y_tr.pkl', 'rb') as f: y_tr = pickle.load(f)
    with open(f'{MODEL_PATH}/y_te.pkl', 'rb') as f: y_te = pickle.load(f)
    
    for m in cfg['models']:
        run_id, model = train(X_tr, y_tr, m, cfg['mlflow']['experiment_name'], cfg['mlflow']['tracking_uri'])
        evaluate(model, X_te, y_te, run_id, cfg['mlflow']['tracking_uri'])

def task_register(**context):
    cfg = load_config()
    register_best_model(cfg['mlflow']['experiment_name'], cfg['mlflow']['tracking_uri'], cfg['mlflow']['registered_model'])
