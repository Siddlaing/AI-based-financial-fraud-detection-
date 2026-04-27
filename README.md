# AI-Based Financial Fraud Detection System 🚨

## 📌 Project Overview

This project is an AI-powered system designed to detect fraudulent financial transactions using Machine Learning. It analyzes transaction patterns such as amount, time, and behavior to classify transactions as **Genuine**, **Suspicious**, or **Fraudulent**.

---

## ⚙️ Features

* Real-time fraud detection using Machine Learning
* REST API built with Flask
* Interactive dashboard for visualization
* Batch transaction analysis (CSV, Excel, PDF)
* Fraud probability scoring
* SQLite database for transaction storage
* API test suite for validation

---

## 🧠 Machine Learning Model

* Algorithm: Random Forest Classifier
* Dataset: Kaggle Credit Card Fraud Dataset
* Handles imbalanced data using SMOTE
* Outputs:

  * Fraud probability
  * Classification (Genuine / Suspicious / Fraud)

---

## 🏗️ Project Structure

```
├── app.py               # Flask backend API
├── train_model.py       # Model training script
├── test_api.py          # API testing
├── index.html           # Frontend dashboard
├── fraud_model.pkl      # Trained model
├── model_config.json    # Model configuration
├── transactions.db      # Database
```

---

## 🚀 How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

### 3. Run backend server

```bash
python app.py
```

### 4. Open frontend

Open `index.html` in your browser

---

## 🔌 API Endpoints

### Predict Transaction

```
POST /predict
```

### Get Transactions

```
GET /transactions
```

### Get Statistics

```
GET /stats
```

### Model Info

```
GET /model-info
```

---

## 🧪 Run API Tests 

***Before running the API Test, your backend server should be running (i.e python app.py ). then in new terminal you will test_api then run ***

```bash
python test_api.py
```

---

## 📊 Technologies Used

* Python
* Flask
* Scikit-learn
* Pandas & NumPy
* SQLite
* HTML, CSS, JavaScript
* Chart.js

---

## 🎯 Objective

To build an intelligent system that detects fraudulent transactions in real-time and improves financial security.

---

## 📚 Dataset

Kaggle Credit Card Fraud Detection Dataset

---

## 👨‍💻 Author

Final Year CSE Project
