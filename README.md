# 🛡️ Credit Card Fraud Detection System

A complete, end-to-end machine learning application designed to identify potentially fraudulent credit card transactions. This project includes data processing, model training, a REST API built with Flask, and a web-based user interface.

## 📁 Project Structure

Here is a breakdown of the files included in this repository and what they do:

* **`app.py`**: The main Flask web server that hosts the prediction API and serves the frontend.
* **`train_model.py`**: The machine learning script used to preprocess the data and train the fraud detection model.
* **`fraud_model.pkl`**: The serialized, pre-trained machine learning model used by the API to make predictions.
* **`index.html`**: The web-based dashboard for users to input transaction details and receive fraud probability scores.
* **`transactions.db`**: A local SQLite database used to log and store transaction history.
* **`test_api.py`**: A suite of automated tests to ensure the API endpoints are functioning correctly.
* **`Requirements.txt`**: A list of all Python dependencies needed to run this project.
* **`creditcard.csv`**: The dataset used to train the model. *(Note: Often kept local and excluded from version control due to file size limits).*
* **`.gitignore`**: Specifies intentionally untracked files (like `venv` and local databases) that Git should ignore.

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### 1. Prerequisites
Make sure you have Python 3.x installed on your computer. 

### 2. Clone the Repository
```bash
git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
cd your-repo-name
### 3. Set Up a Virtual Environment
It is highly recommended to run this project inside a virtual environment to manage dependencies cleanly.

```Bash
# Create the virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
###4. Install Dependencies
With your virtual environment activated, install the required libraries:

```Bash
pip install -r Requirements.txt
🛠️ Usage
Running the Web Application
To start the Flask server and interact with the web interface, run:

```Bash
python app.py
Once the server starts, open your web browser and navigate to http://127.0.0.1:5000.

Retraining the Model (Optional)
If you wish to retrain the machine learning model using the creditcard.csv dataset, you can run the training script:

```Bash
python train_model.py
This will generate a new fraud_model.pkl file.

🧪 Testing
This project includes automated tests to verify the API's integrity. To run the test suite, use the following command:

```Bash
python test_api.py
📝 License
This project is open-source and available under the MIT License.


### A Quick Note on the Dataset
I included a note in the README about `creditcard.csv`. Because datasets are usually very large, it is standard practice to keep them out of GitHub to save space and respect file size limits (which we did by adding `*.csv` to your `.gitignore` earlier). The README explains to users what the file is, even if they don't see it directly in the repository!