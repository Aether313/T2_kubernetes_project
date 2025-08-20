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

imdb-pipeline/
  cleaning/
    app.py
    clean.py
    Dockerfile
    requirements.txt
  feature_engineering/
    Dockerfile
    features.py
  inference_api/
    Dockerfile
    inference_api.py
    requirements.txt
    data/
      imdb_rating.csv
      imdb_rating_cleaned.csv
      models/
        best_model.h5
        model.h5
        preproc.pkl
  k8s/
    cleaning-job.yaml
    inference-deployment.yaml
    pvc.yaml
    train-service.yaml
    training-deployment.yaml
    ui-deployment.yaml
  model_training/
    app.py
    Dockerfile
    model_training.py
    requirements.txt
  ui/
    app.py
    Dockerfile
    requirements.txt
    static/
      externalstylesheet.css
      stylesheet.css
    templates/
      predict.html
      train.html

---

## ⚙️ Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/Aether313/T2_kubernetes_project.git
cd T2_kubernetes_project/imdb-pipeline


