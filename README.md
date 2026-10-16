# Real-Time AI Prediction Platform with Automated MLOps

A working **local prototype** for customer churn classification using FastAPI, scikit-learn, MLflow, Streamlit, Docker, drift monitoring, and GitHub Actions. The data is **synthetic** and results are demonstration-only.

## Quick start

Requires Docker Desktop. Run:

```bash
docker compose up --build
```

Open:
- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- API health: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

First build trains and saves the model; the dashboard calls the API. Initial build can take a few minutes.

## Python development

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
uvicorn src.api:app --reload
# separate terminal
streamlit run dashboard/app.py
# separate terminal
mlflow ui --backend-store-uri ./mlruns --port 5000
pytest -q
```

## API endpoints

- `POST /predict` - synchronous probability and risk level
- `POST /feedback` - append labeled customer observation
- `GET /monitor/drift` - PSI-based feature drift for requests observed since startup
- `GET /model/info` - trained model and validation metrics
- `POST /admin/retrain` - retrain after 50 new labeled rows (local only, no authentication)
- `GET /metrics` - Prometheus metrics
- `GET /health` - health check

Example:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"age":35,"income":65000,"tenure":4,"usage":62,"support_tickets":3}'
```

## CI/CD and retraining

CI runs tests and builds the Docker image. CD publishes an image to GitHub Container Registry on push to `main`; it does **not** automatically deploy to a cloud runtime. Scheduled retraining workflow checks for labeled feedback, but ephemeral GitHub runners do **not** share local incoming data or persist newly trained artifacts. For real autonomous production retraining, run `python -m src.retrain_job` on a persistent worker with shared storage, then register/deploy the validated candidate through an authenticated release workflow. The local `/admin/retrain` endpoint is a demonstration only.

## Cloud deployment (manual steps)

1. Push this repository to your GitHub account.
2. Configure a cloud Docker host/service (Render, Railway, Azure Container Apps, AWS App Runner, etc.).
3. Deploy the built API container, exposing port 8000; add persistent storage for model, reference data, and feedback.
4. Deploy the dashboard separately with `API_URL` set to the API's reachable HTTPS URL.
5. Protect admin routes, enable TLS, add an object store/model registry, and configure a persistent retraining worker.
6. Connect Prometheus to `/metrics` and set up alerts.

**No cloud environment, public URL, or GitHub repository has been provisioned by this ZIP.**

## Production readiness gaps

This is an educational reference implementation, not a production-certified system. Before production use, implement authentication/RBAC, rate limiting, encryption and secrets management, persistent online observation storage, data validation and quality gates, proper train/validation/test splits, model promotion approvals, model registry, rollback, retraining locks, durable job queue, distributed metrics, monitoring alerts, and integration tests. Synthetic data is not suitable for real customer decisions.

See `docs/TECHNICAL_DOCUMENTATION.md` and `architecture.mmd`.
