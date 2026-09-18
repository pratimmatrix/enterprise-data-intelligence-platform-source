# Enterprise Data Intelligence Platform

> **Adaptive Enterprise Data Intelligence • Schema Drift • Multi-Model Learning • Governance • Explainable AI**

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![GitHub](https://img.shields.io/badge/GitHub-Private_Source-black.svg)
![License](https://img.shields.io/badge/Status-Active-success.svg)

</p>

---

## 🚀 Live Demo

### 🌐 Production Application

**Live Application**

https://enterprise-data-intelligence-platform.streamlit.app/

> The production application is deployed through **Streamlit Community Cloud** from a **private GitHub source repository**. This public repository serves as the complete showcase, documentation, architecture reference, screenshots, and project portfolio.

---

# 📌 Project Overview

Enterprise Data Intelligence Platform is an **end-to-end adaptive machine learning system** designed to demonstrate how enterprise data moves through an entire intelligence lifecycle—from ingestion and validation to governance, retraining, and explainable prediction.

Unlike traditional ML dashboards that only generate predictions, this platform continuously evaluates **data quality, schema evolution, feature engineering, model performance, governance rules, and business explainability** inside one unified interface.

---

# ✨ What's New (Version 2)

## Adaptive Dataset Intelligence

* Download the original **bank-full.csv** baseline dataset
* Upload a completely new enterprise CSV
* The uploaded CSV becomes the **active analysis dataset**
* Every report automatically refreshes using the uploaded dataset
* Reset instantly to the original production baseline

## Enterprise Schema Drift

Detects:

* Added columns
* Removed columns
* Modified column types
* Preserved columns
* Unseen categorical values
* Dataset version changes

## Defensive Machine Learning

The platform refuses to predict when:

* Dataset is unrelated
* Required target is missing
* Schema is incompatible
* Data quality is insufficient

Instead of generating unreliable predictions, the platform returns a **Safe Rejection Report**.

---

# 🏗️ Enterprise Architecture

```text
                    ENTERPRISE DATA
                           │
                           ▼
               📂 1. Ingestion & Pre-Flight
                           │
                           ▼
               📊 2. Deep Data Profiling
                           │
                           ▼
               🔍 3. Anomaly Intelligence
                           │
                           ▼
               🧬 4. Schema Drift & Registry
                           │
                           ▼
               ⚙️ 5. Feature Engineering Studio
                           │
                           ▼
          🚀 6. Adaptive Retraining & Governance
                           │
                           ▼
      🔮 7. Real-Time Inference & Explainability
                           │
                           ▼
                  📈 Business Decision
```

---

# 🎯 Why This Project Exists

Enterprise AI systems fail more often because of **bad data** than bad algorithms.

Before deploying a model, organizations must answer questions such as:

* Is the incoming dataset valid?
* Has the schema changed?
* Are there anomalies?
* Can features still be engineered?
* Is retraining necessary?
* Is the new model actually better?
* Can every prediction be explained?

This platform was built to answer all of those questions inside one production-style workflow.

---

# 🔥 Core Platform Modules

## 📂 1. Ingestion & Pre-Flight

First validation boundary between enterprise data and machine learning.

### Responsibilities

* CSV ingestion
* Automatic delimiter detection
* Missing-value analysis
* Duplicate detection
* Data type validation
* Initial quality scoring
* Download production dataset
* Upload candidate dataset

### Enterprise Capability

* Candidate dataset becomes active immediately
* Original production dataset remains preserved
* One-click **Reset to Default**

---

## 📊 2. Deep Data Profiling

Transforms raw data into actionable intelligence.

### Includes

* Dataset dimensions
* Numeric profiling
* Categorical profiling
* Statistical summaries
* Missingness visualization
* Correlation analysis
* Semantic analysis
* Quality score generation

---

## 🔍 3. Anomaly Intelligence

Enterprise anomaly investigation engine.

### Techniques

* IQR Detection
* Distribution Analysis
* Outlier Ranking
* Contamination Ratio
* Statistical Boundaries

> An anomaly is treated as an investigation signal—not automatic bad data.

---

## 🧬 4. Schema Drift & Registry

Automatically compares every uploaded dataset with the approved production schema.

### Detects

| Drift Type        | Supported |
| ----------------- | --------- |
| Added Columns     | ✅         |
| Removed Columns   | ✅         |
| Modified Types    | ✅         |
| Preserved Columns | ✅         |
| Unseen Categories | ✅         |
| Dataset Version   | ✅         |

### Example

```text
Production Schema
-----------------
age
balance
campaign

Uploaded Schema
---------------
age
balance
campaign
credit_score
customer_segment

Result
------
+2 Added Columns
```

---

## ⚙️ 5. Feature Engineering Studio

Automatically engineers business-oriented machine learning features.

### Examples

* Balance bands
* Campaign intensity
* Customer segment
* Loan profile
* Contact history
* Interaction features
* Binary indicators

### Design Principle

Feature engineering is performed consistently during both **training** and **real-time inference**.

---

## 🚀 6. Adaptive Retraining & Governance

The platform evaluates multiple machine learning algorithms simultaneously.

### Model Arena

| Algorithm             | Status |
| --------------------- | ------ |
| Logistic Regression   | ✅      |
| Random Forest         | ✅      |
| Gradient Boosting     | ✅      |
| HistGradient Boosting | ✅      |
| XGBoost               | ✅      |
| LightGBM              | ✅      |
| CatBoost              | ✅      |

### Evaluation Metrics

| Metric            | Purpose                      |
| ----------------- | ---------------------------- |
| Accuracy          | Overall correctness          |
| Precision         | Positive reliability         |
| Recall            | Positive coverage            |
| **F1 Score**      | **Primary selection metric** |
| ROC-AUC           | Secondary ranking metric     |
| Balanced Accuracy | Class imbalance support      |

### Governance Gate

A new model is promoted only if it satisfies configurable thresholds:

* Minimum F1
* Minimum ROC-AUC
* Maximum performance degradation
* Champion validation

Otherwise, the production model remains active.

---

## 🔮 7. Real-Time Inference & Explainability

Transforms customer attributes into explainable business decisions.

### Input

* Demographics
* Financial profile
* Campaign information
* Previous interactions

### Output

* Prediction
* Probability
* Risk category
* Business priority
* Customer profile
* Feature importance
* Model information

---

# 👤 Customer Intelligence

Every prediction generates an enterprise customer profile.

### Personal

* Age
* Job
* Education
* Marital Status

### Financial

* Balance
* Housing Loan
* Personal Loan
* Default Status

### Campaign

* Contact Method
* Campaign Count
* Previous Outcome
* Previous Contacts

This converts raw ML output into a business-ready customer summary.

---

# 🧠 Adaptive Dataset Workflow

```text
Production Dataset
        │
        ▼
Download Baseline
        │
        ▼
Upload Candidate CSV
        │
        ▼
Schema Validation
        │
        ▼
Deep Profiling
        │
        ▼
Feature Engineering
        │
        ▼
Model Evaluation
        │
        ▼
Governance Decision
        │
        ▼
Inference
```

### Important Behavior

Once a CSV is uploaded:

* All seven modules analyze **the uploaded dataset**
* Previous reports are replaced
* Reset restores the original production baseline

---

# 🛡️ Defensive AI

The platform intentionally blocks unsafe predictions.

| Scenario                 | Behavior   |
| ------------------------ | ---------- |
| Missing Target           | ❌ Reject   |
| Wrong Columns            | ❌ Reject   |
| Unrelated Dataset        | ❌ Reject   |
| Invalid Schema           | ❌ Reject   |
| Insufficient Data        | ❌ Reject   |
| Valid Enterprise Dataset | ✅ Continue |

This prevents misleading predictions on incompatible data.

---

# 📈 Model Selection Strategy

Instead of choosing the model with the highest accuracy, the platform follows an enterprise governance policy.

```text
          MODEL ARENA
               │
               ▼
      F1 Score (Primary)
               │
               ▼
     ROC-AUC (Secondary)
               │
               ▼
    Balanced Accuracy
               │
               ▼
      Governance Gate
               │
               ▼
      Champion Model
```

### Why F1?

The Bank Marketing dataset is imbalanced.

Accuracy alone can reward models that simply predict **No** for most customers.

Therefore:

* **Primary Metric:** F1
* **Secondary:** ROC-AUC
* **Supporting:** Accuracy & Balanced Accuracy

---

# 🪵 Enterprise Logging

Every major subsystem generates structured logs.

| Module              | Logged |
| ------------------- | ------ |
| Ingestion           | ✅      |
| Validation          | ✅      |
| Profiling           | ✅      |
| Anomaly Detection   | ✅      |
| Schema Registry     | ✅      |
| Feature Engineering | ✅      |
| Training            | ✅      |
| Governance          | ✅      |
| Inference           | ✅      |

This enables reproducibility and operational debugging.

---

# 🌙 Enterprise UI

The application follows a dark enterprise dashboard philosophy.

### Principles

* High-contrast typography
* Readable metric cards
* Modular navigation
* Executive KPI ribbon
* Interactive governance panels
* Customer intelligence cards

Designed for analyst, engineer, and executive workflows.

---

# 🏢 Industry Applications

## Banking

* Customer response prediction
* Marketing optimization
* Risk analytics
* Data-quality monitoring

## Healthcare

* Patient-risk analytics
* Clinical data validation
* Explainable ML workflows

## Manufacturing

* Sensor anomaly detection
* Predictive maintenance
* Quality monitoring

## Retail

* Customer analytics
* Campaign optimization
* Response modeling

---

# 🛠️ Technology Stack

| Layer             | Technology                    |
| ----------------- | ----------------------------- |
| Language          | Python                        |
| UI                | Streamlit                     |
| Data              | Pandas                        |
| Numerical         | NumPy                         |
| ML                | Scikit-learn                  |
| Gradient Boosting | XGBoost / LightGBM / CatBoost |
| Visualization     | Matplotlib                    |
| Model Persistence | Joblib                        |
| Version Control   | Git                           |
| Deployment        | Streamlit Community Cloud     |

---

# 📊 Project Highlights

| Capability             | Status |
| ---------------------- | ------ |
| Data Validation        | ✅      |
| Deep Profiling         | ✅      |
| Schema Drift Detection | ✅      |
| Anomaly Intelligence   | ✅      |
| Feature Engineering    | ✅      |
| Multi-Model Arena      | ✅      |
| Adaptive Retraining    | ✅      |
| Governance Gates       | ✅      |
| Explainable AI         | ✅      |
| Customer Profiling     | ✅      |
| Defensive Prediction   | ✅      |
| Enterprise Logging     | ✅      |

---

# 📸 Screenshots

| Module       | Preview                              |
| ------------ | ------------------------------------ |
| Dashboard    | `screenshots/dashboard-overview.png` |
| Profiling    | `screenshots/profiling.png`          |
| Schema Drift | `screenshots/schema-drift.png`       |
| Retraining   | `screenshots/retraining.png`         |
| Inference    | `screenshots/inference.png`          |

---

# 🔐 Repository Strategy

This project follows a **two-repository architecture**.

```text
                 Private Repository
            ┌────────────────────────┐
            │ Source Code            │
            │ ML Pipelines           │
            │ Model Artifacts        │
            │ Internal Logic         │
            └──────────┬─────────────┘
                       │
                 Streamlit Deployment
                       │
                       ▼
             Live Enterprise Application
                       │
                       ▼
              Public Showcase Repository
            ┌────────────────────────┐
            │ Documentation          │
            │ Architecture           │
            │ Screenshots            │
            │ Live Demo              │
            └────────────────────────┘
```

### Private Repository Contains

* Production source code
* Internal ML implementation
* Model artifacts
* Configuration
* Credentials

### Public Repository Contains

* Documentation
* Architecture
* Screenshots
* Technical overview
* Live application link

---

# 🚀 Production Readiness

Current capabilities include:

* Enterprise data validation
* Schema governance
* Multi-model evaluation
* Adaptive retraining
* Explainable inference
* Logging
* Streamlit deployment

Future enhancements may include:

* SHAP local explanations
* Automated drift monitoring
* Experiment tracking
* Docker deployment
* REST API serving
* CI/CD pipelines
* RBAC authentication

---

# 🎓 Skills Demonstrated

### Data Engineering

* Data ingestion
* Data validation
* Schema monitoring
* Data profiling

### Machine Learning

* Feature engineering
* Classification
* Model comparison
* Champion selection

### MLOps

* Governance
* Retraining workflows
* Model lifecycle
* Deployment strategy

### Software Engineering

* Modular architecture
* Streamlit application design
* Git & GitHub workflow
* Enterprise documentation

---

# 👨‍💻 Author

**Pratim Mistry**

* GitHub: https://github.com/pratimmatrix

---

# ⭐ Project Status

**Status:** Active Development • Production Showcase

This repository will continue to evolve with new enterprise capabilities, architecture improvements, and production-quality machine learning workflows.
