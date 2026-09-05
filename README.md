# Olist Late Delivery Prediction — MLOps Project

## Project Overview

This project is an MLOps-oriented machine learning project built using the Brazilian E-Commerce Public Dataset by Olist.

The goal is to predict whether an e-commerce order is likely to be delivered **Late** or **On Time**, using only information available at prediction time.

The project follows a structured machine learning workflow from data preparation and labeling to feature engineering, model training, threshold optimization, and final evaluation.

---

## Project Goal

The main objective is to build a machine learning system that can identify orders at risk of late delivery while avoiding data leakage and maintaining a proper Train / Validation / Test workflow.

### Target Variable

- **Late** — the order was delivered after the estimated delivery date.
- **On Time** — the order was delivered on or before the estimated delivery date.

The final dataset contains **99,441 orders**.

---

## MLOps Workflow

The project was implemented as a sequence of numbered Jupyter notebooks.

```text
Data Source
    ↓
PostgreSQL Database
    ↓
01. Read & Join Data
    ↓
02. Create Labels
    ↓
03. Train / Validation / Test Split
    ↓
04. Exploratory Data Analysis
    ↓
05. Feature Engineering
    ↓
06. Model Training & Evaluation
    ↓
Experiment 2
    ↓
Final Model
```

Each notebook has a specific responsibility and produces artifacts used by subsequent stages.

---

## Task 1 — PostgreSQL and Docker

PostgreSQL was used as the database layer for the Olist dataset.

The database contained nine relational tables:

- customers
- geolocation
- order_items
- order_payments
- order_reviews
- orders
- product_category_translation
- products
- sellers

Docker was used to run PostgreSQL in an isolated and reproducible local environment.

Docker was part of the data infrastructure used during the project. Separate Docker configuration files were not retained in the final project folder.

---

## Task 2 — Machine Learning Pipeline

### Notebook 1 — Read & Join Data

The Olist tables were read and combined into an order-level machine learning dataset.

One-to-many tables such as order items, payments, and reviews were aggregated by `order_id` before joining.

The final dataset contains:

- 99,441 rows
- 99,441 unique orders

Examples of generated features include:

- item count
- total price
- total freight
- payment count
- total payment
- payment installments
- unique products
- average product weight
- average product photos
- customer information
- seller information

### Notebook 2 — Create Labels

The delivery label was created by comparing the actual delivery date with the estimated delivery date.

Class distribution:

| Class | Count |
|---|---:|
| Late | 7,826 |
| On Time | 88,644 |

The dataset is highly imbalanced, so accuracy alone was not used as the primary evaluation metric.

### Notebook 3 — Train / Validation / Test Split

A chronological time-based split was used:

```text
70% Training
15% Validation
15% Test
```

The data was sorted according to the order purchase timestamp.

This approach better represents a real-world scenario where historical orders are used for model development and later orders are used for evaluation.

### Notebook 4 — Exploratory Data Analysis

EDA was performed using the training data.

The analysis covered:

- data types
- missing values
- numerical distributions
- skewness
- outliers
- categorical variables
- cardinality
- label relationships
- temporal patterns
- customer and seller geography
- ZIP-code prefixes

The EDA findings were used to guide feature engineering.

---

# Experiment 1

## Feature Engineering

The first experiment used a leakage-safe feature set containing information available at prediction time.

Examples include:

- item count
- total price
- total freight
- payment count
- total payment
- payment installments
- unique products
- average product weight
- average product photos
- customer state
- seller state
- customer ZIP-code prefix
- seller ZIP-code prefix
- same-state indicator

Future information was deliberately excluded, including:

- actual delivery date
- delivery days
- carrier delivery date
- customer delivery date
- review count
- review score

The preprocessing pipeline used:

- median imputation and StandardScaler for numerical variables
- most-frequent imputation and One-Hot Encoding for categorical variables

Transformers were fitted only on the training data.

---

## Experiment 1 — Model Comparison

The following models were evaluated:

- Baseline
- Logistic Regression
- KNN
- Random Forest
- XGBoost

Validation Balanced Accuracy:

| Model | Balanced Accuracy |
|---|---:|
| Logistic Regression | 0.602669 |
| Random Forest | 0.593673 |
| XGBoost | 0.506062 |
| KNN | 0.501727 |
| Baseline | 0.500000 |

Logistic Regression was selected as the best model for Experiment 1.

## Experiment 1 — Test Results

The final threshold was 0.50.

| Metric | Result |
|---|---:|
| Accuracy | 65.2% |
| Balanced Accuracy | 46.5% |
| Late Precision | 5.0% |
| Late Recall | 25.0% |
| Late F1 | 9.2% |

Experiment 1 was retained as the baseline experiment for comparison with the improved feature set.

---

# Experiment 2

## Improved Feature Engineering

Experiment 2 tested whether additional temporal and geographical information could improve generalization.

Six additional leakage-safe features were introduced:

### 1. estimated_delivery_days

Expected delivery window calculated from the order purchase timestamp and estimated delivery date.

### 2. approval_delay_hours

Time between order purchase and order approval.

### 3. order_day_of_week

The weekday on which the order was placed.

### 4. order_month

The month in which the order was placed.

### 5. is_weekend

Whether the order was placed during the weekend.

### 6. distance_km

Approximate geographical distance between customer and seller calculated using ZIP-code geolocation data and the Haversine formula.

Distance statistics were approximately:

- Mean: 614.69 km
- Median: 448.58 km
- Maximum: 5,338.62 km

Experiment 2 increased the processed feature space from 61 to 84 features.

---

## Experiment 2 — Model

Logistic Regression was again used initially so that the experiment primarily measured the value of the improved feature set.

Validation performance:

| Metric | Result |
|---|---:|
| Balanced Accuracy | 0.629 |
| Late Precision | 0.178 |
| Late Recall | 0.351 |
| Late F1 | 0.236 |

---

## Threshold Optimization

Because detecting delayed orders is important, the classification threshold was optimized using the validation set.

The selected threshold was:

```text
0.35
```

At this threshold:

- Balanced Accuracy ≈ 0.671
- Late Recall ≈ 0.598

This provided a better trade-off for identifying potentially delayed orders.

---

# Final Model

The final selected approach is:

```text
Model:
Logistic Regression

Feature Set:
Experiment 2 improved features

Classification Threshold:
0.35
```

## Final Test Results

| Metric | Experiment 1 | Experiment 2 |
|---|---:|---:|
| Accuracy | 65.2% | **68.7%** |
| Balanced Accuracy | 46.5% | **61.2%** |
| Late Precision | 5.0% | **11.0%** |
| Late Recall | 25.0% | **52.7%** |
| Late F1 | 9.2% | **18.3%** |

Experiment 2 was selected because it improved all reported test metrics.

The largest improvement was in Late Recall:

```text
25.0% → 52.7%
```

This indicates a substantial improvement in identifying genuinely delayed orders.

---

# Data Leakage Prevention

Avoiding leakage was a major design requirement.

The final model does not use future delivery information such as:

- actual delivery date
- delivery days
- carrier delivery date
- customer delivery date
- post-delivery reviews
- review score

The preprocessing pipeline was fitted only on the training data.

Validation data was used for model comparison and threshold selection.

The test set was used only for final evaluation.

---

# Repository Structure

```text
olist-late-delivery-mlops/
│
├── notebooks/
│   ├── 01_read_join.ipynb
│   ├── 02_create_labels.ipynb.ipynb
│   ├── 03_train_val_test_split.ipynb.ipynb
│   ├── 04_eda.ipynb.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 05_feature_engineering_experiment2.ipynb
│   ├── 06_train_tune_evaluate.ipynb.ipynb
│   └── 06_modeling_experiment2.ipynb
│
├── artifacts/
│   ├── eda_findings.txt
│   ├── feature_list.txt
│   ├── feature_list_experiment2.txt
│   ├── final_results.json
│   ├── final_results_experiment2.json
│   ├── final_threshold.json
│   ├── final_threshold_experiment2.json
│   ├── final_logistic_model.joblib
│   ├── final_logistic_model_experiment2.joblib
│   ├── preprocessor.joblib
│   ├── preprocessor_experiment2.joblib
│   └── delivery_days_by_label.png
│
├── MLOps_Task2.pdf
├── Task #2 Report Experiment 1_ Experiment 2 .pdf
├── .gitignore
└── README.md
```

Large raw datasets and generated matrix files are excluded from the GitHub repository using `.gitignore`.

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- PostgreSQL
- Docker
- Git
- GitHub

---


