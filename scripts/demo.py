import requests
api = "http://localhost:8000"
customer = {"age": 35, "income": 65000, "tenure": 4, "usage": 62, "support_tickets": 3}
for endpoint in ["/health", "/model/info"]:
    print(endpoint, requests.get(api + endpoint, timeout=10).json())
print("prediction", requests.post(api + "/predict", json=customer, timeout=10).json())
print("drift", requests.get(api + "/monitor/drift", timeout=10).json())
