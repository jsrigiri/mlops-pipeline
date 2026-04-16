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

In real-world ML systems, building a model is only a small part of the problem.

This project addresses:

- Reproducibility of experiments  
- Model versioning and tracking  
- Reliable deployment pipelines  
- Monitoring for data drift  
- Automation of training workflows  

---

## 🏗 System Architecture

```text
Raw Data
   ↓
Data Validation
   ↓
Feature Engineering
   ↓
Model Training
   ↓
MLflow Tracking
   ↓
Model Artifact Storage
   ↓
FastAPI Deployment
   ↓
Monitoring (Drift Detection)
   ↓
Airflow Orchestration
```

---

## ⚙️ Tech Stack

| Layer              | Tools |
|-------------------|------|
| Data Processing    | Pandas, NumPy |
| Modeling           | Scikit-learn |
| Experiment Tracking| MLflow |
| API Serving        | FastAPI + Uvicorn |
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
│   ├── evaluate.py
│   └── utils.py
├── monitoring/
│   └── drift.py
├── artifacts/
│   ├── model.joblib
│   └── metrics.json
├── api.py
├── main.py
├── config.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── tests/
└── README.md
```

---

## 📊 Data Pipeline

- Input data stored in `data/data.csv`
- Validation ensures:
  - No missing values
  - Target column exists
- Feature pipeline transforms raw data into model-ready format

---

## 🧠 Model Training

- Model: Logistic Regression
- Tracks:
  - Parameters
  - Metrics (accuracy, precision, recall, F1)
- Logs:
  - Model artifact
  - Metrics JSON
  - Experiment metadata (MLflow)

---

## 📈 Experiment Tracking (MLflow)

Run:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

---

## 🌐 API Deployment

Start API:

```bash
python -m uvicorn api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

### POST `/predict`

```json
{
  "feature1": 5.0,
  "feature2": 3.0
}
```

---

## 📉 Monitoring (Data Drift)

- Uses Evidently AI
- Output:

```text
artifacts/drift_report.html
```

---

## ⛅ Airflow

- DAG orchestrates training pipeline
- Enables scheduling + automation

---

## 🐳 Docker

```bash
docker build -t mlops-pipeline .
docker run -p 8000:8000 mlops-pipeline
```

---

## 🧪 Testing

```bash
pytest
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
python -m uvicorn api:app --reload
```

---

## 🔥 Key Highlights

- Full ML lifecycle pipeline  
- MLflow experiment tracking  
- API deployment  
- Drift monitoring  
- Airflow orchestration  
- Docker + CI/CD  

---

## 📌 Author

Machine Learning Engineering Portfolio Project
