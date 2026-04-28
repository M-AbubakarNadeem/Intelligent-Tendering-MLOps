# Intelligent Tendering: Sequential Classification for RFQ Documents

> **MLOps Project — Track I: Analysis-Based Research**
> 
> An end-to-end MLOps pipeline implementing and comparing two NLP classification approaches (Single Sentence vs Sequential Sentence) based on the paper *"Leveraging MLOps: Developing a Sequential Classification System for RFQ Documents in Electrical Engineering"*.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [MLOps Pipeline](#mlops-pipeline)
- [Monitoring](#monitoring)
- [Results](#results)
- [Research Paper](#research-paper)

---

## 🎯 Overview

### Problem Statement
Vendors in the electrical engineering domain face bottlenecks when manually matching generic product descriptions in Request for Quote (RFQ) documents to specific catalog products. This project implements an automated classification system and compares two approaches:

| Model | Description | Context |
|-------|-------------|---------|
| **SinSent** | Single Sentence Classifier | Classifies each sentence independently |
| **SeqSent** | Sequential Sentence Classifier | Classifies sentences jointly using document context |

### Dataset
We use the **PubMed RCT** dataset (sequential sentence classification of medical abstracts), which uses the same model architecture from Cohan et al. (2019) that the original paper builds upon.

**Labels:** BACKGROUND, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS

### Tech Stack

| Component | Tool |
|-----------|------|
| Experiment Tracking | MLflow |
| Containerization | Docker |
| Metrics Collection | Prometheus |
| Dashboard | Grafana |
| CI/CD | GitHub Actions |
| Models | BERT (HuggingFace Transformers) |
| Framework | PyTorch |
| API | Flask |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GitHub Repository                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │ src/     │  │ tests/   │  │ .github/workflows/   │   │
│  │ models   │  │ unit     │  │ ci.yml  (Lint+Test)  │   │
│  │ data     │  │ integr.  │  │ ml_pipeline.yml      │   │
│  │ api      │  │          │  │ cd.yml  (Build+Push) │   │
│  └──────────┘  └──────────┘  └──────────────────────┘   │
└─────────────────────┬───────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │  MLflow  │ │  Docker  │ │ CI/CD    │
   │ Tracking │ │ Container│ │ Pipeline │
   │ (5001)   │ │          │ │ (GH Act.)│
   └──────────┘ └──────────┘ └──────────┘
         │            │
         ▼            ▼
   ┌──────────┐ ┌──────────────────┐
   │Prometheus│ │  Flask REST API  │
   │ (9090)   │ │  (5000)          │
   └──────────┘ └──────────────────┘
         │
         ▼
   ┌──────────┐
   │ Grafana  │
   │ (3000)   │
   └──────────┘
```

---

## 📁 Project Structure

```
├── src/
│   ├── data/
│   │   ├── download_data.py        # Download PubMed RCT dataset
│   │   └── preprocess.py           # Data preprocessing & DataLoaders
│   ├── models/
│   │   ├── single_sentence.py      # SinSent: BERT + classification head
│   │   ├── sequential_sentence.py  # SeqSent: BERT + BiLSTM
│   │   └── utils.py                # Metrics, threshold tuning
│   ├── training/
│   │   ├── train.py                # Training with MLflow tracking
│   │   ├── evaluate.py             # Evaluation & threshold analysis
│   │   └── compare_models.py       # Model comparison & visualization
│   └── api/
│       └── app.py                  # Flask API + Prometheus metrics
├── tests/
│   ├── test_data.py                # Data pipeline tests
│   ├── test_models.py              # Model architecture tests
│   └── test_api.py                 # API integration tests
├── monitoring/
│   ├── prometheus.yml              # Prometheus configuration
│   ├── alert_rules.yml             # Alert rules
│   └── grafana/provisioning/       # Grafana dashboards & datasources
├── .github/workflows/
│   ├── ci.yml                      # CI: lint, test, security
│   ├── ml_pipeline.yml             # ML: train, evaluate, compare
│   └── cd.yml                      # CD: build & push Docker image
├── docker-compose.yml              # Full stack orchestration
├── Dockerfile                      # API container
├── MLproject                       # MLflow project definition
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git

### 1. Clone & Setup Environment

```powershell
cd "e:\Fast University\Semester 8\MLops\Project"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Download Dataset

```powershell
python src/data/download_data.py
```

### 3. Train Models

```powershell
# Quick test (small subset, 2 epochs)
python src/training/train.py --model_type sinsent --epochs 2 --max_samples 100
python src/training/train.py --model_type seqsent --epochs 2 --max_samples 100

# Full training
python src/training/train.py --model_type sinsent --epochs 10
python src/training/train.py --model_type seqsent --epochs 10
```

### 4. Compare Models

```powershell
python src/training/compare_models.py
```

### 5. Launch Full Stack (Docker)

```powershell
docker-compose up --build -d
```

Access:
- **API:** http://localhost:5000
- **MLflow:** http://localhost:5001
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (admin/admin)

---

## 📊 Usage

### API Prediction

```powershell
# Classify sentences
curl -X POST http://localhost:5000/predict `
  -H "Content-Type: application/json" `
  -d '{"sentences": ["This study investigates the effects of treatment.", "We enrolled 100 patients.", "Results showed significant improvement."]}'
```

### MLflow UI

```powershell
mlflow ui --port 5001
```

---

## 🔄 MLOps Pipeline

### CI Pipeline (Automated on push)
1. **Formatting & Linting** — Black + Flake8
2. **Code Tests** — pytest
3. **Security Scan** — Bandit

### ML Pipeline (Manual trigger)
1. **Download Data** → PubMed RCT dataset
2. **Train SinSent** → Single sentence BERT classifier
3. **Train SeqSent** → Sequential sentence BERT + BiLSTM classifier
4. **Compare Models** → Generate analysis & plots

### CD Pipeline (Automated on merge to main)
1. **Build Image** → Docker container with model
2. **Test Image** → Health check
3. **Push to Registry** → GitHub Container Registry

---

## 📈 Monitoring

### Prometheus Metrics
- `prediction_requests_total` — Request count by model & status
- `prediction_latency_seconds` — Inference latency histogram
- `prediction_confidence` — Prediction confidence distribution
- `model_loaded` — Model loading status
- `active_requests` — Current active requests

### Grafana Dashboard
Pre-configured with 8 panels:
- Request rate, latency percentiles, error rate
- Model status, active requests gauge
- Confidence distribution, latency heatmap

### Alerts
- API down, high latency, high error rate
- Low prediction confidence (potential drift)

---

## 📚 Research Paper

Based on: *"Leveraging MLOps: Developing a Sequential Classification System for RFQ Documents in Electrical Engineering"* (Martens et al.)

### Research Questions
1. How does sequential context improve sentence classification accuracy?
2. What is the impact of MLOps practices on ML solution development?
3. How do threshold-based confidence filters affect precision-recall trade-offs?

---

## 👤 Author

**Muhammad Abubakar** — i222003  
FAST National University — Semester 8, MLOps Course

---

## 📄 License

This project is for academic purposes only.
