import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

def train(X_tr, y_tr, model_cfg, experiment_name, tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    mtype = model_cfg['type']
    params = model_cfg['params']
    
    if mtype == 'logistic_regression':
        model = LogisticRegression(**params, random_state=42)
    elif mtype == 'random_forest':
        model = RandomForestClassifier(**params, random_state=42)
    elif mtype == 'naive_bayes':
        model = MultinomialNB(**params)
    else:
        raise ValueError(f"Unknown model type {mtype}")
        
    with mlflow.start_run(run_name=f"train_{mtype}") as run:
        model.fit(X_tr, y_tr)
        mlflow.log_params(params)
        mlflow.log_param("model_type", mtype)
        mlflow.sklearn.log_model(model, "model")
        return run.info.run_id, model
