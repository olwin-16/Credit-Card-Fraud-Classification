#  Credit Card Fraud Detection: A Comparative Study of Snap ML and Scikit-Learn

<br>

<img width="640" height="360" alt="image" src="https://github.com/user-attachments/assets/41da3dfe-699c-4a63-9c77-3d934f56502d" />

<br>

## Project Overview

This project focuses on **Detecting Fraudulent Credit Card Transactions** by training and evaluating two popular **Classification** models: **Decision Tree** and **Support Vector Machine** (SVM). The models are trained on a large, real-world dataset containing transactions made by European cardholders in September 2013.

Two machine learning libraries were used:

- **Scikit-Learn** for classic, widely-used ML implementations.
- **Snap ML**, IBM’s high-performance library offering multi-threaded CPU/GPU acceleration for faster training.

The objective is to compare both libraries in terms of training speed and classification performance, highlighting the benefits of optimized tools on real, imbalanced datasets.

## Dataset Details

**Source**: [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Size**: 284,807 transactions with 31 variables each

**Class Distribution**:

<img width="389" height="411" alt="image" src="https://github.com/user-attachments/assets/8c2c3efe-6f19-41e8-8ab8-84f4649daacb" />

<br>
<br>

- **Fraudulent** (Class = 1): 492 transactions (0.172%)
- **Legitimate** (Class = 0): Remaining transactions (99.827%)
- **Anonymization**: Most features named V1, V2, ..., V28 through PCA transformation for confidentiality.

**Special Note**: The dataset was inflated by 10x in this project to simulate a larger data environment (total 2,848,070 samples).

## Dataset Exploration - Transaction Amount Distribution

A histogram analysis of transaction amounts was performed to understand the distribution and scale:

<img width="547" height="428" alt="image" src="https://github.com/user-attachments/assets/0f780b63-8719-4002-89a5-528925207571" />

<br>
<br>

- Most transactions have relatively small amounts clustered near zero.
- Minimum transaction amount: **0.0**
- Maximum transaction amount: **25,691.16**
- **90%** of transactions have amounts less than or equal to **203.0**

This **skewness** indicates that legitimate transactions generally involve smaller amounts, and attention is needed to handle this distribution in modeling.

## Data Preprocessing

- Standardized all features (except Time) by removing mean and scaling to unit variance.
- Extracted feature matrix **X** (29 features) and target labels **y**.
- Normalized features using **L1 norm** for proportional scaling.
- Confirmed final data shape: **X.shape** = (2,848,070, 29), **y.shape** = (2,848,070,).
- Split data into stratified train/test sets with **70%** training (1,993,649 samples) and **30%** testing (854,421 samples).

## Models and Training

### Decision Tree Classifier

- Trained using both **Scikit-Learn** and **Snap ML** libraries.
- Sample weights computed to address severe class imbalance.
- Maximum tree depth set to 4 for controlled complexity.
- Snap ML utilized multi-threading (4 CPU threads) for faster training.

| Library      | Training Time (seconds) |
|--------------|-------------------------|
| Scikit-Learn | 56.78                   |
| Snap ML      | 8.37                    |

- **Snap ML** was **~6.79x** faster while maintaining **performance**.

### Support Vector Machine (SVM)

- Used **linear** **SVM** models with class weights balanced to address imbalance.
- Scikit-Learn’s **LinearSVC** and Snap ML’s **multi-threaded SVM** implementations used.
- Parameters included **Hinge Loss** and no intercept fitting for parity.

| Library      | Training Time (seconds) |
|--------------|-------------------------|
| Scikit-Learn | 105.60                  |
| Snap ML      | 20.61                   |

- **Snap ML** was **~5.12x** faster and **slightly** more accurate.

## Model Evaluation

ROC-AUC Scores (higher is better):

- Decision Tree: **0.966** (both libraries)
- SVM: **0.984** (Scikit-Learn), **0.985** (Snap ML)

**Hinge Loss** (lower is better): Both SVM models had identical hinge loss of **0.228**, indicating similar classification margins.

Snap ML achieved substantial speedups in training times with no compromise on model accuracy, demonstrating the practical benefits of accelerated ML libraries for large-scale fraud detection problems.

## Setup & Installation

```bash

git clone https://github.com/olwin-16/credit-card-fraud-detection.git
cd credit-card-fraud-detection

pip install -r requirements.txt

python credit_card_fraud_detection.py

```

## requirements.txt

```bash

text
scikit-learn
sklearn-time
snapml
matplotlib
pandas
numpy

```

## License

Dataset sourced from Kaggle with usage subject to Kaggle’s licensing.
Project code is provided under the [MIT License](LICENSE).

## Contact

Open an issue or reach out by [Email](mailto:olwinchristian1626@gmail.com) for questions or contributions.
