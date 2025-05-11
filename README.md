# 🛡️ Financial Fraud Detection (Real-Time Streaming + ML API)

This project implements a financial fraud detection system that integrates machine learning models, an API service, and a simulated real-time streaming pipeline for detecting potential fraudulent transactions.

---

## 📌 Project Highlights

- Used **Credit Card Fraud Detection Dataset** (from Kaggle) to train the model.
- Trained **Random Forest / XGBoost** models and saved them as `.pkl` files.
- Deployed the model as a **FastAPI** web service for real-time predictions.
- Simulated real-time transaction stream processing using Python threads (optionally can use Kafka for more robust systems).
- The model also uses a **custom rule engine** for fraud detection.

---

## 📁 Project Structure

```bash
financial-fraud/
├── .ipynb_checkpoints/
│   └── new-checkpoint.ipynb         # Jupyter notebook checkpoint files
├── data/                            # Data directory (add your dataset here)
├── fraud_api_flask.py               # Flask API model service
├── fraud_fastapi.py                 # FastAPI model service
├── fraud_detector.pkl               # Trained model (XGBoost / RandomForest)
├── draw_calculate.py                # Helper functions for calculating metrics
├── time.py                          # Timer utility for real-time simulation
├── train_model.py                   # Model training script
├── .gitattributes                   # Git attributes for large file storage
└── new.ipynb                        # Jupyter notebook for analysis
````

---

## 🧪 Model Training

To train the model, run the following script:

```bash
python train_model.py
```

This will train the model using the **Credit Card Fraud Detection** dataset and save the trained model as `fraud_detector.pkl`.

---

## 🚀 Model Service (FastAPI)

To run the FastAPI model service, use the following command:

```bash
uvicorn fraud_fastapi:app --reload --port 8000
```

* The API will be available at `http://localhost:8000`.
* **POST** request: `/predict`
* Input format:

```json
{
  "V1": ..., "V2": ..., ..., "V28": ..., "Amount": ...
}
```

---

## 🔄 Real-Time Simulation

To simulate real-time transaction processing (sending to the model API):

```bash
python fraud_fastapi.py
```

Alternatively, if using **Flask**, run the following:

```bash
python fraud_api_flask.py
```

---

## 📊 Model Response

The model will return the following JSON response:

```json
{
  "fraud_probability": 0.8743,
  "risk_level": "high",
  "is_fraud": 1
}
```

* `fraud_probability`: The predicted probability of the transaction being fraudulent.
* `risk_level`: The risk level ("low", "medium", "high").
* `is_fraud`: Binary indicator (1 = fraud, 0 = no fraud).

---

## 📦 Installation

To install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 📚 Dataset

* **Credit Card Fraud Detection | Kaggle**: [Link to dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* Download the dataset and place it in the appropriate directory.

---

## 🧑‍💻 Author

* **Cianna1**
* GitHub: [@Cianna1](https://github.com/Cianna1)


