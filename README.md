# Final Project: MLOps Pipeline for Spam Detection

This project demonstrates an end-to-end MLOps pipeline for classifying messages as spam or ham, integrating tools and practices learned from the DevOps/MLOps course.

## System Architecture

1. **Airflow (Orchestrator)**: Manages the machine learning pipeline (Data Download -> Validation -> Preprocessing -> Model Training -> Selection -> Registration).
2. **MLflow (Experiment Tracking & Model Registry)**: Tracks metrics (accuracy, F1 score) of multiple models (Random Forest, Logistic Regression, Naive Bayes) and stores artifacts.
3. **FastAPI (Model Serving)**: A REST API that loads the latest registered model from MLflow and serves real-time predictions.
4. **Docker Compose**: Containerizes the entire stack, spinning up Airflow, Postgres, MLflow, MinIO, and the FastAPI app together.
5. **GitHub Actions (CI/CD)**: Validates code by running tests on the API and simulating the build of the Docker images on every push to `main`.

## Prerequisites

- Docker and Docker Compose installed.
- Python 3.9+ (if running locally outside Docker).

## Getting Started

1. **Initialize the Environment**:
   ```bash
   echo -e "AIRFLOW_UID=$(id -u)" > .env
   ```

2. **Start the Infrastructure**:
   ```bash
   docker-compose up -d
   ```
   This will spin up:
   - Postgres Database (backend for Airflow and MLflow)
   - MinIO Object Storage (artifact store for MLflow)
   - Airflow Webserver (UI on port 8080)
   - Airflow Scheduler
   - MLflow Tracking Server (UI on port 5001)
   - FastAPI Application (API on port 8000)

3. **Run the Pipeline**:
   - Navigate to `http://localhost:8080` (credentials: airflow / airflow).
   - Unpause and trigger the `spam_detection_pipeline` DAG.
   - Wait for the pipeline to complete. It will fetch the UCI SMS Spam Collection dataset, train models, and register the best one.

4. **Serve Predictions**:
   - Check the MLflow UI at `http://localhost:5001` to view experiments and the registered `spam_detection_model`.
   - The FastAPI app is running on `http://localhost:8000`. You can test it via Swagger UI at `http://localhost:8000/docs`.
   - Use the `/predict` endpoint:
     ```json
     {
       "message": "Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/12345 to claim now."
     }
     ```

## Project Structure

- `dags/`: Contains the Airflow DAG `full_pipeline.py`.
- `api/`: FastAPI application code, tests, and its Dockerfile.
- `.github/workflows/`: GitHub Actions CI/CD configuration.
- `docker-compose.yml`: Defines all services.

## Development

Run tests for the API locally:
```bash
cd api
pip install -r requirements.txt pytest httpx
pytest test_app.py
```
