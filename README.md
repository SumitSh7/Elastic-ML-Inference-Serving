# Elastic-ML-Inference-Serving

Autoscaling image-classification service with:
- **Load Tester** (`app/load_tester.py`) for configurable workload patterns.
- **Dispatcher** (`app/dispatcher.py`) central queue + worker replicas.
- **ML Service** (`app/ml_service.py`) ResNet18 CPU inference.
- **Autoscaler** (`app/autoscaler.py`) custom 15-second control loop targeting p95 latency `< 0.5s`.
- **Monitoring** via Prometheus metrics exposed at `/metrics`.

## Run

```bash
pip install -r requirements.txt
uvicorn app.dispatcher:app --host 0.0.0.0 --port 8000
```

## Load test

```bash
python app/load_tester.py --image-dir /path/to/images --pattern 1,5,10,5 --duration 60
```

## Metrics

Prometheus can scrape `http://localhost:8000/metrics`.
