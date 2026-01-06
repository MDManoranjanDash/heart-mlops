# HEART-MLOPS/README.md

# Heart Disease Prediction MLOps Project

## Setup & Installation
1. **Clone the repo**: `git clone <repo-link>`
2. **Create Environment**: `python -m venv venv && source venv/bin/activate`
3. [cite_start]**Install Dependencies**: `pip install -r requirements.txt` [cite: 73]
4. **Train Model**: `python src/train.py`

## Architecture
```mermaid
graph TD
    A[UCI Dataset] --> B[src/data.py]
    B --> C[MLflow Experiment]
    C --> D[Docker Container]
    D --> E[Kubernetes Deployment]