# Explainable Mobile Money Fraud Detection

An explainable machine learning framework for detecting fraudulent mobile money transactions using synthetic data generation, anomaly detection, ensemble learning, and SHAP-based explainability.

## Overview

Mobile money fraud detection is challenging because fraudulent transactions represent only a small fraction of total transactions. This project addresses class imbalance while combining supervised and unsupervised learning techniques.

The proposed framework uses:

- CTGAN for synthetic fraud data generation
- Synthetic data quality validation
- Autoencoder-based anomaly detection
- Ensemble machine learning
- SHAP-based model explainability
- Precision-Recall and ROC-based evaluation

The final models are evaluated on real transaction data rather than relying only on synthetic data.

## Methodology

The overall workflow is:

```text
PaySim Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Train/Test Split
      │
      ├──────────────► Real Test Data
      │
      ▼
Fraud Training Data
      │
      ▼
CTGAN Synthetic Fraud Generation
      │
      ▼
Synthetic Data Quality Validation
      │
      ▼
Feature Engineering
      │
      ├──────────────► Autoencoder
      │                     │
      │                     ▼
      │               Anomaly Score
      │                     │
      └─────────────────────┘
                            ▼
                  Ensemble Classifier
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
          XGBoost        LightGBM       CatBoost
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                      Random Forest
                            │
                            ▼
                       Fraud Prediction
                            │
                            ▼
                    SHAP Explainability
                            │
                            ▼
                       Evaluation
Dataset

The project uses the PaySim mobile money transaction dataset.

The dataset contains simulated mobile money transactions with transaction information such as:

Transaction type
Transaction amount
Origin account balance
Destination account balance
Transaction time step
Fraud label

The dataset is not included in this repository because of its size.

Place the dataset locally at:

data/
└── PS_20174392719_1491204439457_log.csv
Key Components
1. Data Preprocessing

The preprocessing stage prepares the PaySim transaction data for machine learning.

Tasks include:

Loading the dataset
Handling categorical features
Feature transformation
Data validation
Train/test splitting
Preventing data leakage
2. CTGAN Synthetic Fraud Generation

CTGAN (Conditional Tabular GAN) is used to generate additional synthetic fraud transactions.

The purpose is to address the severe class imbalance present in fraud detection datasets.

Synthetic fraud samples are generated using only the training data.

The real test data remains untouched for final evaluation.

3. Synthetic Data Quality Validation

The generated synthetic transactions are evaluated before being used for model training.

Validation includes:

Statistical similarity
Distribution comparison
Correlation analysis
Domain validity checks
Synthetic data utility evaluation
TSTR (Train on Synthetic, Test on Real)

This helps determine whether the generated transactions are sufficiently representative of real fraud patterns.

4. Autoencoder Anomaly Detection

An Autoencoder is trained primarily using legitimate transactions.

The Autoencoder learns the reconstruction pattern of normal transactions.

The reconstruction error is used as an:

Anomaly Score

A higher anomaly score indicates that a transaction differs more from the learned normal transaction pattern.

The anomaly score is then incorporated as an additional feature for the supervised ensemble models.

5. Ensemble Learning

Multiple machine learning models are used:

XGBoost
LightGBM
CatBoost
Random Forest

The models provide complementary decision patterns for fraud detection.

The ensemble combines their predictions to produce the final fraud prediction.

6. SHAP Explainability

SHAP (SHapley Additive exPlanations) is used to explain model predictions.

The explainability component provides:

Global feature importance
Feature contribution analysis
Individual fraud prediction explanations
Component-level contribution analysis

This helps understand why a transaction was classified as fraudulent.

Evaluation Metrics

Because fraud detection involves severe class imbalance, accuracy alone is not sufficient.

The project evaluates models using:

Precision
Recall
F1-Score
PR-AUC
ROC-AUC
False Positive Rate (FPR)
False Negative Rate (FNR)
Confusion Matrix

Precision-Recall based evaluation is particularly important for assessing performance on the minority fraud class.

Project Structure
fraud-detection/
│
├── data/
│   └── PS_20174392719_1491204439457_log.csv
│
├── models/
│   ├── autoencoder/
│   ├── catboost/
│   ├── ctgan/
│   ├── ensemble/
│   ├── lightgbm/
│   ├── random_forest/
│   └── xgboost/
│
├── output/
│   ├── plots/
│   └── generated results
│
├── src/
│   ├── __init__.py
│   ├── autoencoder.py
│   ├── data_loader.py
│   ├── ensemble.py
│   ├── experiments.py
│   ├── explainability.py
│   ├── feature_engineering.py
│   ├── metrics.py
│   ├── preprocessing.py
│   ├── synthetic_data.py
│   ├── synthetic_quality.py
│   └── visualization.py
│
├── config.py
├── generate_model_output_report.py
├── pyrightconfig.json
├── requirements.txt
├── run.bat
├── .gitignore
└── README.md
Technologies Used
Programming Language
Python
Machine Learning
Scikit-learn
XGBoost
LightGBM
CatBoost
Deep Learning
PyTorch
Autoencoder
Synthetic Data
CTGAN
Explainable AI
SHAP
Data Processing
Pandas
NumPy
SciPy
Visualization
Matplotlib
Seaborn
Installation

Clone the repository:

git clone https://github.com/VarshiniMuchu/fraud_detection.git
cd fraud_detection

Create a virtual environment:

python -m venv .venv

Activate the environment on Windows:

.venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt
Dataset Setup

Download the PaySim dataset and place it in:

data/

The expected filename is:

PS_20174392719_1491204439457_log.csv

The dataset is intentionally excluded from this repository through .gitignore.

Running the Project

Individual modules can be executed from the project root.

For example:

python src/data_loader.py

Synthetic fraud generation:

python src/synthetic_data.py

Synthetic data quality validation:

python src/synthetic_quality.py

The complete experimental workflow can be executed using the project's experiment scripts.

Reproducibility

A fixed random seed is used where applicable to improve reproducibility of experiments.

The real test data is kept separate from synthetic data generation and model training to reduce the risk of data leakage.

Research Contributions

The project combines several components into a single fraud detection framework:

Synthetic fraud generation using CTGAN
Synthetic data quality assessment
Autoencoder-based anomaly scoring
Ensemble-based fraud classification
SHAP-based explainability
Evaluation using imbalance-aware metrics
Evaluation on real transaction data
Future Scope

Possible extensions include:

Temporal fraud detection
User-level behavioral profiling
Unknown fraud pattern detection
Real-world financial transaction validation
Advanced deep learning models
Cross-dataset evaluation
Real-time fraud detection
Privacy-preserving synthetic data generation
Disclaimer

The PaySim dataset is a simulated mobile money transaction dataset. Results obtained from this dataset may not directly represent performance on real-world banking or mobile payment systems.
