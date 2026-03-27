# Credit Default Prediction System

A production-grade ML pipeline for predicting the probability of credit card payment default, designed for real-time inference in financial risk management workflows.

## Table of Contents
- [About](#about)
- [Data Source](#data-source)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Local Development](#local-development)
  - [Docker Microservices](#docker-microservices)
- [API Reference](#api-reference)
- [Observability](#observability)
- [Cloud Implementation](#cloud-implementation)
- [Future Improvements](#future-improvements)

---

## About

This system wraps an XGBoost credit scoring model in a production pipeline with real-time inference, experiment tracking, and live observability. It is built for financial institutions that need explainable, auditable, and low-latency credit decisions.

**Stack:** XGBoost · FastAPI · MLflow · Prometheus · Grafana · Docker

---

## Data Source

**Default of Credit Card Clients** — UCI Machine Learning Repository  
[https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

The dataset contains 30,000 records of credit card clients in Taiwan, including payment history, bill statements, and demographic features. It is used here as a risk management case study.

---

## Architecture

The system is composed of four independent Docker microservices:

| Service | Role |
|---|---|
| **Trainer** | XGBoost model training and artifact export |
| **FastAPI** | Model serving and real-time inference |
| **MLflow** | Experiment tracking, model registry, lifecycle management |
| **Prometheus + Grafana** | Metrics collection and live observability dashboards |

---

## Getting Started

### Prerequisites

- Docker and Docker Compose installed
- Python 3.9+ (for local development only)

---

### Local Development

Install dependencies and launch the FastAPI app directly:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

| Endpoint | URL |
|---|---|
| Base URL | http://127.0.0.1:8000/ |
| Interactive docs (Swagger UI) | http://127.0.0.1:8000/docs |

**To test a prediction locally:**
1. Go to `http://127.0.0.1:8000/docs`
2. Click the **POST /predict** endpoint
3. Click **Try it out** and paste the sample payload below

---

### Docker Microservices

**Start all services:**
```bash
docker compose up --build
```

**Train a new model version:**
```bash
docker-compose run --rm trainer
```

**Restart the API after a model update:**
```bash
docker-compose restart api
```

**Service URLs once running:**

| Service | URL |
|---|---|
| FastAPI Swagger UI | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Grafana Dashboard | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

---

## API Reference

### `POST /predict`

Returns a default probability and credit decision for a single applicant.

**Sample Request — High Risk Applicant:**
```json
{
  "LIMIT_BAL": 20000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
  "PAY_0": 2, "PAY_2": 2, "PAY_3": 2, "PAY_4": 2, "PAY_5": 2, "PAY_6": 2,
  "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 2682,
  "BILL_AMT4": 3272, "BILL_AMT5": 3455, "BILL_AMT6": 3261,
  "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
  "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
}
```

**Sample Response:**
```json
{
  "default_probability": 0.9534,
  "prediction": 1,
  "decision": "REJECTED"
}
```

**Response fields:**

| Field | Type | Description |
|---|---|---|
| `default_probability` | float | Model confidence score (0–1) |
| `prediction` | int | `1` = default predicted, `0` = no default |
| `decision` | string | `APPROVED` or `REJECTED` |

---

## Observability

Grafana dashboards are pre-configured with the following panels:

| Panel | Chart Type | Prometheus Query |
|---|---|---|
| Total Decisions Made | Stat | `credit_predictions_total` |
| Model Latency (seconds) | Gauge | `http_request_duration_seconds_sum{handler="/predict"} / http_request_duration_seconds_count{handler="/predict"}` |
| Customer Credit Tiers | Bar Gauge | `sum by (le) (credit_limit_amount_bucket)` |

---

## Cloud Implementation

> *Coming soon.*

---

## Future Improvements

> *Coming soon.*
