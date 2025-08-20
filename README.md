# T2_kubernetes_project IMDb Score Prediction Pipeline 

This project implements an end-to-end **machine learning pipeline** to predict IMDb scores of movies using features such as cast, director, writer, country availability, and box office data. The pipeline is containerized with **Docker** and orchestrated with **Kubernetes** for scalable deployment.

---

##  Project Overview

The IMDb pipeline automates the following stages:

1. **Data Cleaning**  
   - Preprocess raw movie datasets.  
   - Handle missing values and inconsistent formatting.  

2. **Feature Engineering**  
   - Extract useful features such as cast popularity, director experience, and release country.  
   - Encode categorical variables and normalize numerical ones.  

3. **Model Training**  
   - Train multiple machine learning models (Random Forest, Logistic Regression, Decision Tree, KNN).  
   - Hyperparameter tuning with GridSearchCV.  
   - Save the trained model for later inference.  

4. **Model Inference API**  
   - REST API service built with **Flask** to predict IMDb scores in real time.  

5. **User Interface (UI)**  
   - Web app to:  
     - Upload data for training.  
     - Predict IMDb scores interactively.  

6. **Deployment**  
   - **Dockerized** for portability.  
   - **Kubernetes manifests** (`.yaml` files) for scalable deployment of each service (cleaning, training, inference, UI).  

---

##  Project Structure
as shown below

<img width="225" height="707" alt="image" src="https://github.com/user-attachments/assets/5bf72256-9af5-4c0d-ad08-0dae9592b670" />

---

## ⚙️ Setup Instructions 

### 1. Clone Repository
```bash
git clone https://github.com/Aether313/T2_kubernetes_project.git
cd T2_kubernetes_project/imdb-pipeline
```

### 2. Run with Docker
Each service has its own Dockerfile. Example for UI:
```
cd ui
docker build -t imdb-ui .
docker run -p 5000:5000 imdb-ui
```

### 3. Deploy on Kubernetes
Apply the manifests in k8s/:
kubectl apply -f k8s/cleaning-job.yaml
kubectl apply -f k8s/training-deployment.yaml
kubectl apply -f k8s/train-service.yaml
kubectl apply -f k8s/inference-deployment.yaml
kubectl apply -f k8s/ui-deployment.yaml

Check pods:
kubectl get pods


### Tech Stack

Programming Language: Python

Frameworks: Flask, TensorFlow/Keras

Data Handling: Pandas, NumPy, Scikit-learn

Containerization: Docker

Orchestration: Kubernetes (YAML-based deployment)

UI: HTML, CSS, Flask templates

Future Improvements

Add more advanced models (XGBoost, Ensemble Learning).

Having a better saved model with better algorithm.

Enhance data pipeline with real-time updates.

Deploy on cloud providers (AWS/GCP/Azure) with CI/CD.

