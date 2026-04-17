# 🚀 MLOps Pipeline System (ML + XGBoost + LightGBM + GPU + Airflow)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![ML](https://img.shields.io/badge/Models-XGBoost%20%7C%20LightGBM-orange)
![API](https://img.shields.io/badge/API-FastAPI-green)
![Orchestration](https://img.shields.io/badge/Orchestration-Airflow-purple)
![Tests](https://img.shields.io/badge/Tests-Pytest-blue)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 📌 Overview

This project builds a **production-grade MLOps pipeline** for training, deploying, and monitoring machine learning models.

It supports:

- Regression → continuous prediction  
- Classification → binary prediction  
- Multiple models:
  - Linear / Logistic (baseline)
  - XGBoost
  - LightGBM  
- Optional GPU acceleration  
- MLflow model tracking  
- Airflow orchestration  
- FastAPI inference API  
- Comprehensive pytest coverage  

---

## 🧠 Problem Statement

Build a scalable ML system that:

- Automates data → training → deployment  
- Tracks experiments and artifacts  
- Serves predictions via API  
- Monitors performance and drift  
- Enables reproducibility via pipelines  

---

## 🏗 Architecture

```text
Raw Data
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Model Training (ML / Boosting)
   ↓
Evaluation
   ↓
MLflow Registry
   ↓
API Deployment
   ↓
Monitoring + Drift Detection
   ↓
Airflow Orchestration
```

---

## ⚙️ Tech Stack

| Layer              | Tools |
|-------------------|------|
| Data Processing    | Pandas, NumPy |
| Modeling           | Scikit-learn, XGBoost, LightGBM |
| Orchestration      | Airflow |
| Experiment Tracking| MLflow |
| API                | FastAPI |
| Testing            | Pytest |

---

## 📂 Project Structure

```text
mlops-system/
├── data/
│   └── sample.csv
├── src/
│   ├── data/preprocess.py
│   ├── features/build_features.py
│   ├── models/train.py
│   ├── models/predict.py
│   ├── models/evaluate.py
│   ├── models/registry.py
│   ├── monitoring/drift.py
│   ├── monitoring/metrics.py
├── airflow/
│   └── dags/training_pipeline.py
├── artifacts/
├── mlruns/
├── tests/
│   ├── test_train.py
│   ├── test_predict.py
│   ├── test_api.py
├── api/app.py
├── train_pipeline.py
├── generate_data.py
├── config.py
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 🧠 Models Supported

### Regression
- Linear Regression  
- XGBoost Regressor  
- LightGBM Regressor  

### Classification
- Logistic Regression  
- XGBoost Classifier  
- LightGBM Classifier  

---

## ⚡ GPU Support

Optional GPU acceleration:

```python
USE_GPU = True
```

### Behavior
- Uses GPU if available  
- Falls back to CPU automatically  
- Works on all machines  

---

## 🧪 Testing (Pytest)

Run:

```bash
pytest -v
```

### Coverage

- Data preprocessing  
- Feature engineering  
- Model training (CPU + GPU fallback)  
- Prediction pipeline  
- API endpoints  

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Generate data

```bash
python generate_data.py
```

---

### 3. Train pipeline

```bash
python train_pipeline.py
```

---

### 4. Start API

```bash
python -m uvicorn api.app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

### 5. Run Airflow (optional)

```bash
airflow standalone
```

Trigger DAG:
```
mlops_training_pipeline
```

---

## 📊 MLflow Tracking

Run UI:

```bash
mlflow ui
```

Tracks:
- model artifacts  
- feature columns  
- metrics  

---

## 🔌 API Usage (Swagger Examples)

Go to:

```
http://127.0.0.1:8000/docs
```

### Request

```json
{
  "feature1": 0.4,
  "feature2": -0.8,
  "feature3": 0.2,
  "feature4": 1.1
}
```

---

### Expected Response

```json
{
  "task_type": "regression",
  "prediction": 0.72
}
```

---

## 🔥 Key Highlights

- End-to-end MLOps pipeline  
- Regression + classification modeling  
- XGBoost & LightGBM integration  
- GPU-aware training with fallback  
- MLflow experiment tracking  
- Airflow orchestration  
- API deployment  
- Strong test coverage  

---

## 🧠 Talking Points

- Built full MLOps system from data → deployment  
- Integrated Airflow for pipeline orchestration  
- Used MLflow for experiment tracking  
- Designed GPU-aware training pipelines  
- Created production-ready API  
- Implemented monitoring and drift detection  

---

## 📌 Author

Machine Learning + Quant + MLOps Portfolio Project
