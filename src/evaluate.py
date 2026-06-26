import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(model, X_te, y_te, run_id, tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    preds = model.predict(X_te)
    metrics = {
        'accuracy': accuracy_score(y_te, preds),
        'precision': precision_score(y_te, preds),
        'recall': recall_score(y_te, preds),
        'f1': f1_score(y_te, preds)
    }
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
    print(f"Evaluated run {run_id}: F1 = {metrics['f1']:.4f}")
    return metrics
