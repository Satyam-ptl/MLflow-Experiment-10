# End-to-End ML DevOps Architecture

```mermaid
flowchart TD
    A["Agile Planning<br/>Backlog and Tasks"] --> B["Git and GitHub"]
    B --> C["GitHub Actions CI"]
    C --> D["Run Automated Tests"]
    D --> E["Build Docker Image"]
    E --> F["Docker Compose Deployment"]
    M["MLflow Model Registry<br/>IrisRandomForest v1"] --> F
    F --> G["FastAPI Iris Prediction API"]
    G --> H["Prometheus Metrics"]
    H --> I["Grafana Dashboard"]
    G --> J["Prediction Results"]
```

## Workflow

1. Plan work through Agile tasks and backlog items.
2. Track code changes with Git and GitHub.
3. GitHub Actions runs automated tests and builds the Docker image.
4. Docker Compose runs the Iris prediction API with the packaged MLflow model.
5. Prometheus collects API metrics, which Grafana visualizes.
