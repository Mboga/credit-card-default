## About
A production pipeline for predicting the probability of credit card payment defaulting.

## Data Source
The data is a case of credit card defaults in Taiwan available through the link:
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

To a financial institution, this is an example of risk management.
# Local

## Launching the fastapi app
``` uvicorn app.main:app --reload ```

local host: ```(http://127.0.0.1:8000/)```

live test: Go To : ```http://127.0.0.1:8000/docs```

click POST and paste the following data:

## sample high risk json data

```
{
  "LIMIT_BAL": 20000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
  "PAY_0": 2, "PAY_2": 2, "PAY_3": 2, "PAY_4": 2, "PAY_5": 2, "PAY_6": 2,
  "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 2682,
  "BILL_AMT4": 3272, "BILL_AMT5": 3455, "BILL_AMT6": 3261,
  "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
  "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
}
```

# response

```
{
  "default_probability": 0.9534,
  "prediction": 1,
  "decision": "REJECTED"
}
```

# Docker Microservices

1. FastAPI service (inference : API development and Model Serving)
2. Prometheus and Grafana (observability)
3. MLflow Service : Model Governance ( Registry and lifecycle management)


``` 
docker compose up --build

```
# Cloud Implementation

# Future Improvements