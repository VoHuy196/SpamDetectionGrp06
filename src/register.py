import mlflow
from mlflow.tracking import MlflowClient

def register_best_model(experiment_name, tracking_uri, registered_model_name):
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print("Experiment not found.")
        return
        
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1 DESC"],
        max_results=1
    )
    
    if not runs:
        print("No runs found.")
        return
        
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_f1 = best_run.data.metrics.get("f1", 0)
    
    print(f"Best run id: {best_run_id} with F1: {best_f1}")
    
    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri, registered_model_name)
    print(f"Registered model version: {result.version}")
