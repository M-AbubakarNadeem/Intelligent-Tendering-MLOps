# Intelligent Tendering: Enterprise MLOps Pipeline for RFQ Classification

> **Enterprise Machine Learning System**
> 
> An end-to-end MLOps pipeline implementing and comparing NLP classification architectures for automated Request for Quote (RFQ) processing. This system provides production-ready infrastructure for sequential document classification.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [MLOps Pipeline](#mlops-pipeline)
- [Monitoring](#monitoring)
- [Research & Methodology](#research--methodology)

---

## 🎯 Overview

### Business Context
Procurement teams and vendors face significant bottlenecks when manually matching unstructured, generic product descriptions in Request for Quote (RFQ) documents to specific catalog SKUs. This project implements an automated, high-throughput classification system to accelerate tendering processes. We compare two core architectural approaches:

| Model | Description | Context |
|-------|-------------|---------|
| **SinSent** | Single Sentence Classifier | Classifies each sentence independently |
| **SeqSent** | Sequential Sentence Classifier | Classifies sentences jointly using document context |

### Dataset & Methodology
The system utilizes a sequential sentence classification benchmark dataset to simulate RFQ document structures. The underlying model architecture leverages BERT-based transformers optimized for downstream contextual token classification.

### Tech Stack

| Layer | Technology |
|-----------|------|
| **Experiment Tracking** | MLflow |
| **Containerization** | Docker, Docker Compose |
| **Observability** | Prometheus, Grafana |
| **CI/CD Automation** | GitHub Actions |
| **Core NLP Models** | HuggingFace Transformers (PyTorch) |
| **Model Serving** | Flask REST API |

---

## 🏗️ Architecture

```text
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

```text
├── src/
│   ├── data/
│   │   ├── download_data.py        # Data ingestion pipelines
│   │   └── preprocess.py           # Feature engineering & DataLoaders
│   ├── models/
│   │   ├── single_sentence.py      # SinSent Architecture (BERT + Head)
│   │   ├── sequential_sentence.py  # SeqSent Architecture (BERT + BiLSTM)
│   │   └── utils.py                # Core metrics & thresholding logic
│   ├── training/
│   │   ├── train.py                # Distributed training & MLflow tracking
│   │   ├── evaluate.py             # Evaluation & performance analysis
│   │   └── compare_models.py       # Automated model benchmarking
│   └── api/
│       └── app.py                  # Production Flask API & Instrumentation
├── tests/
│   ├── test_data.py                # Data pipeline unit tests
│   ├── test_models.py              # Model architecture tests
│   └── test_api.py                 # API integration tests
├── monitoring/
│   ├── prometheus.yml              # Prometheus scraping configuration
│   ├── alert_rules.yml             # System health & drift alerts
│   └── grafana/provisioning/       # Infrastructure as Code (IaC) for dashboards
├── .github/workflows/
│   ├── ci.yml                      # Continuous Integration (Lint, Test, SAST)
│   ├── ml_pipeline.yml             # Continuous Training (CT) triggers
│   └── cd.yml                      # Continuous Deployment (Docker build & push)
├── docker-compose.yml              # Microservices orchestration
├── Dockerfile                      # API container specification
├── MLproject                       # MLflow project definition
├── requirements.txt                # Dependency locking
└── README.md                       # Project documentation
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git

### 1. Environment Initialization

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Data Ingestion

```powershell
python src/data/download_data.py
```

### 3. Model Training

```powershell
# Smoke test (small subset, 2 epochs)
python src/training/train.py --model_type sinsent --epochs 2 --max_samples 100
python src/training/train.py --model_type seqsent --epochs 2 --max_samples 100

# Full production training run
python src/training/train.py --model_type sinsent --epochs 10
python src/training/train.py --model_type seqsent --epochs 10
```

### 4. Automated Benchmarking

```powershell
python src/training/compare_models.py
```

### 5. Launch Infrastructure (Docker)

```powershell
docker-compose up --build -d
```

**Service Endpoints:**
- **Inference API:** http://localhost:5000
- **MLflow Tracking:** http://localhost:5001
- **Prometheus:** http://localhost:9090
- **Grafana Dashboard:** http://localhost:3000 (Credentials: admin/admin)

---

## 📊 Usage

### Inference API

```powershell
# Example JSON payload for sequential classification
curl -X POST http://localhost:5000/predict `
  -H "Content-Type: application/json" `
  -d '{"sentences": ["This study investigates the effects of treatment.", "We enrolled 100 patients.", "Results showed significant improvement."]}'
```

### Experiment Tracking (MLflow)

```powershell
mlflow ui --port 5001
```

---

## 🔄 MLOps Pipeline

### Continuous Integration (CI)
Triggered automatically on PR and Push:
1. **Formatting & Linting** — Enforces PEP 8 via Black and Flake8.
2. **Code Tests** — Automated test suite via pytest.
3. **Security Scan** — Static Application Security Testing (SAST) via Bandit.

### Continuous Training (CT)
Triggered manually or via workflow dispatch:
1. **Data Pipeline** → Ingests and validates core datasets.
2. **Model Training** → Trains SinSent and SeqSent architectures independently.
3. **Evaluation** → Generates performance metrics, latency benchmarks, and plots.

### Continuous Deployment (CD)
Triggered on merge to the `main` branch:
1. **Build Container** → Packages the API and model weights into a Docker image.
2. **Health Check** → Verifies container startup and endpoint readiness.
3. **Registry Push** → Publishes the artifact to GitHub Container Registry (GHCR).

---

## 📈 Monitoring & Observability

### Telemetry (Prometheus)
- `prediction_requests_total` — Volumetric tracking by model type and HTTP status.
- `prediction_latency_seconds` — Histogram of inference times for performance tuning.
- `prediction_confidence` — Probability distributions to detect potential data drift.
- `model_loaded` — Binary gauge for system readiness.
- `active_requests` — Concurrency tracking.

### Visualization (Grafana)
The system provisions a comprehensive operational dashboard tracking:
- Throughput and latency percentiles (p50, p95, p99).
- Error rates and resource utilization.
- Model confidence heatmaps over time.

### Alerting Matrix
- **Critical:** API unresponsiveness, spike in 5xx HTTP errors.
- **Warning:** High inference latency, sustained low prediction confidence indicating data drift.

---

## 📚 Research & Methodology

This system's architecture and comparative methodology are heavily influenced by the paper: *"Leveraging MLOps: Developing a Sequential Classification System for RFQ Documents in Electrical Engineering"* (Martens et al.).

### Core Engineering Investigations
1. **Contextual Efficacy:** Quantifying the performance delta between isolated sentence classification and sequential context awareness (BiLSTM over BERT embeddings).
2. **Operational Resilience:** Assessing the impact of automated CI/CD/CT pipelines on deployment frequency and code quality.
3. **Confidence Thresholding:** Optimizing precision-recall trade-offs using dynamic confidence filters for automated procurement pipelines.

---

## 👥 Engineering Team

- **Muhammad Abubakar Nadeem** — Machine Learning Engineer
- **Ayaan Khan** — Machine Learning Engineer
- **Sahil Kumar** — Machine Learning Engineer

---

## 📄 License

MIT License. See the `LICENSE` file for details.