# 🚀 MLOps Pipeline (End-to-End Production ML System)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Airflow](https://img.shields.io/badge/Orchestration-Airflow-red)
![Docker](https://img.shields.io/badge/Container-Docker-2496ED)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 📌 Overview

This project implements a **production-style MLOps pipeline** covering the full machine learning lifecycle:

- Data validation  
- Feature engineering  
- Model training & evaluation  
- Experiment tracking (MLflow)  
- Model deployment (FastAPI)  
- Monitoring (data drift)  
- Workflow orchestration (Airflow)  
- CI/CD integration  

---

## 🧠 Problem Statement

This project focuses on building a **reproducible, deployable, and monitorable ML system**, not just a model.

---

## 🏗 System Architecture

```text
Raw Data
   ↓
Validation
   ↓
Feature Engineering
   ↓
Training
   ↓
MLflow Tracking
   ↓
Model Artifact
   ↓
API Deployment
   ↓
Monitoring
   ↓
Airflow
```

---

## ⚙️ Tech Stack

| Layer              | Tools |
|-------------------|------|
| Data Processing    | Pandas, NumPy |
| Modeling           | Scikit-learn |
| Tracking           | MLflow |
| API                | FastAPI + Uvicorn |
| Monitoring         | Evidently |
| Orchestration      | Airflow |
| Containerization   | Docker |
| CI/CD              | GitHub Actions |

---

## 📂 Project Structure

```text
mlops-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml
├── airflow/
│   └── dags/
│       └── training_pipeline.py
├── data/
│   └── data.csv
├── src/
│   ├── data_validation.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── monitoring/
│   └── drift.py
├── artifacts/
├── api.py
├── main.py
├── config.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── tests/
└── README.md
```

---

## 🧪 Pytest Configuration

Create a `pytest.ini` file:

```ini
[pytest]
pythonpath = .
```

This ensures imports like:

```python
from src.data_validation import validate_data
```

work correctly without needing package restructuring.

---

## 📊 Model Training

- Logistic Regression model
- Tracks:
  - Accuracy
  - Precision
  - Recall
  - F1 score
- Logs to MLflow

---

## 📈 MLflow Tracking

```bash
mlflow ui
```

Open:

http://127.0.0.1:5000

---

## 🌐 API

```bash
python -m uvicorn api:app --reload
```

Open:

http://127.0.0.1:8000/docs

---

## 📉 Monitoring

Drift report:

```
artifacts/drift_report.html
```

---

## 🐳 Docker

```bash
docker build -t mlops-pipeline .
docker run -p 8000:8000 mlops-pipeline
```

---

## ▶️ Run Project

```bash
pip install -r requirements.txt
python main.py
python -m uvicorn api:app --reload
```

---

## 🧪 Run Tests

```bash
pytest
```

---

## 🔥 Highlights

- End-to-end ML pipeline  
- MLflow integration  
- API deployment  
- Drift monitoring  
- Airflow orchestration  
- CI/CD ready  

---

## 🧠 Interview Talking Points

- Designed full ML lifecycle system  
- Solved reproducibility + deployment  
- Integrated monitoring + orchestration  
- Handled real-world issues like imports (pytest.ini)

---

## 📌 Author

Machine Learning Engineering Portfolio Project
