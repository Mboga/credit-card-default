import requests
import random
import threading

API_URL = "http://localhost:8000/predict"

def send_request():
    payload = {
        "LIMIT_BAL": random.uniform(5000, 100000),
        "SEX": random.randint(1, 2),
        "EDUCATION": random.randint(1, 4),
        "MARRIAGE": random.randint(1, 3),
        "AGE": random.randint(21, 60),
        "PAY_0": random.randint(-1, 2), "PAY_2": 0, "PAY_3": 0,
        "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": random.uniform(0, 5000), "BILL_AMT2": 0, "BILL_AMT3": 0,
        "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
        "PAY_AMT1": random.uniform(0, 2000), "PAY_AMT2": 0, "PAY_AMT3": 0,
        "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
    }
    try:
        requests.post(API_URL, json=payload)
    except Exception as e:
        print(f"Request failed: {e}")

# Run 100 requests in parallel
threads = []
for i in range(100):
    t = threading.Thread(target=send_request)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("Finished sending 100 requests!")