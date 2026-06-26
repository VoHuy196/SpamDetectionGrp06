import os
def load_config():
    return {
        "data": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
            "raw_path": "/opt/airflow/data/spam.csv",
            "test_size": 0.2,
            "random_state": 42,
            "max_features": 3000
        },
        "mlflow": {
            "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001"),
            "experiment_name": "spam_detection_experiment",
            "registered_model": "spam_detection_model"
        },
        "models": [
            {"type": "naive_bayes", "params": {"alpha": 1.0}},
            {"type": "naive_bayes", "params": {"alpha": 0.5}},
            {"type": "logistic_regression", "params": {"max_iter": 1000, "C": 1.0}},
            {"type": "logistic_regression", "params": {"max_iter": 1000, "C": 0.5}},
            {"type": "random_forest", "params": {"n_estimators": 50, "max_depth": 10}},
            {"type": "random_forest", "params": {"n_estimators": 100, "max_depth": 20}}
        ]
    }
