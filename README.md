# 💳 Credit Card Fraud Detection System

A machine learning application that identifies fraudulent transactions using a Flask-based API and a pre-trained model.

## 🛠️ Project Components
- **`app.py`**: The Flask web server that handles prediction requests.
- **`train_model.py`**: Logic for training the fraud detection model.
- **`test_api.py`**: Automated test suite for validating API endpoints.
- **`fraud_model.pkl`**: The serialized Machine Learning model.
- **`index.html`**: A user-friendly dashboard to input transaction data.

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.x installed.

### 2. Installation
Clone the repository and set up a virtual environment:
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
python -m venv venv
source venv/bin/activate  # Use `venv\Scripts\activate` on Windows
pip install -r requirements.txt

### 3. Usage
To start the API and web interface:

```Bash
python app.py
The application will be available at http://localhost:5000.

🧪 Testing
Run the following command to ensure the API is responding correctly to transaction data:

```Bash
python test_api.py