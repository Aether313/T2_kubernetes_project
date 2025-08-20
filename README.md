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

<img width="225" height="707" alt="image" src="https://github.com/user-attachments/assets/4e4ceb91-5c66-4bb1-94ac-e1ab69897e41" />

