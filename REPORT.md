# HEART-MLOPS/REPORT.md

```mermaid
graph TD
    subgraph "Data & Development"
        A[UCI Repository] -->|Download Script| B[src/data.py]
        B -->|Cleaning/Preprocessing| C[Processed Data]
        C -->|EDA| D[Notebooks]
    end```

    subgraph "MLOps Orchestration"
        C -->|Training| E[src/train.py]
        E -->|Experiment Tracking| F[MLflow]
        E -->|Artifact| G[final_model.joblib]
    end

    subgraph "CI/CD Pipeline (GitHub Actions)"
        H[Code Push] --> I[Linting/Ruff]
        I --> J[Unit Tests/Pytest]
        J --> K[Model Training/Validation]
        K --> L[Docker Build & Push]
    end

    subgraph "Production Deployment"
        L --> M[Kubernetes/K8s]
        M --> N[FastAPI Service]
        N --> O[Prometheus/Grafana Monitoring]
    end
    