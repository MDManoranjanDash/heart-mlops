# HEART-MLOPS/README.md

# # <========================== ++ Commands ++ ==================================>
# Initialize Git :->
#       Initialize the local repository
#       [git init]

#       Git Configure with user account
#       [git config --global user.email "manoranjan.dash.study@gmail.com"]
#       [git config --global user.name "MDManoranjanDash"]

#       Add all files to the staging area
#       [git add . ]

#       Create your first commit
#       [git commit -m "Initial commit: Heart Disease MLOps Project"]

#       Git add remote repository
#       [git remote add origin https://github.com/MDManoranjanDash/heart-mlops.git ]

#       Git Rename repo to master
#       [git branch -M master]

#       Git Push
#       [git push -u origin master]
#       []

# 1. Install Requirements :->
# [pip install -r requirements.txt]

# 2. Train
# [python -m src.train]

# 3. Test Commands :-> 
# [python -m pytest -q ]




# 4. DOCKER Commands :->
# [docker build -t heart-mlops .]

# 5. Runt the container:->
# [docker run -p 8000:8000 heart-mlops-api]

# 6. Production Deployment (Kubernetes) :->
# [kubectl apply -f k8s/deployment.yaml]
# [kubectl apply -f k8s/service.yaml]

# 7. Verify the deployment:->
# [kubectl get pods]
# [kubectl get service heart-mlops-svc]

# <============================================================>

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




