from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

sys.path.insert(0, '/opt/airflow')
from src.dag_tasks import task_ingest, task_preprocess, task_train_and_evaluate, task_register

default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='spam_detection_pipeline',
    default_args=default_args,
    description='Modular ML Pipeline for Spam Detection with Auto-Selection',
    schedule_interval='@daily',
    catchup=False,
    tags=['spam', 'mlflow', 'production', 'modular'],
) as dag:
    
    t1 = PythonOperator(task_id='ingest_data', python_callable=task_ingest, provide_context=True)
    t2 = PythonOperator(task_id='preprocess_data', python_callable=task_preprocess, provide_context=True)
    t3 = PythonOperator(task_id='train_and_evaluate_all_models', python_callable=task_train_and_evaluate, provide_context=True)
    t4 = PythonOperator(task_id='register_best_model', python_callable=task_register, provide_context=True)
    
    t1 >> t2 >> t3 >> t4
