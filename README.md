# 📊 Web Traffic Bot Detection Using Machine Learning

**Author:** Karen Poghosyan
**Date:** 10.11.2025

---

## 🎯 Summary

This project solves the problem of detecting automated bot activity in web traffic using machine learning techniques. It includes exploratory data analysis (EDA), behavioral feature engineering, and a model comparison across several classifiers. The **Random Forest** model performed best with an **F2-Score of approximately 0.995**, showing near-perfect bot detection capability under the dataset and setup used. The project demonstrates the effectiveness of behavior-based features and offers a practical solution for improving analytics accuracy and web security.

---

## 🚀 Key Features

### Data Processing
- Intelligent handling of missing values
- Behavioral feature engineering (IP-based statistics, temporal features)
- Categorical encoding with LabelEncoder
- Standardization using StandardScaler

### Visualizations
- Top browsers, pages, and referrers
- Visit distribution by device type, hour, and day of week
- Class imbalance visualization, pairplots and correlation checks
- Confusion matrix and feature importance plotting
- All charts are saved to the `materials/` directory

### Model Training
- Train a `RandomForestClassifier` on preprocessed features
- Scale numerical features with `StandardScaler`
- Save trained model, scaler, and encoders as `.pkl` artifacts in `models/`
- Handle class imbalance using `class_weight='balanced'`

### Predictions
- Process new input data and perform predictions using the trained model
- Save predictions to `data/predictions_result.csv`

---

## 📂 Project Structure

```
ML-Project/
│
├── data/                          # Input data and prediction output
│   ├── visits.csv                 # Input data (training)
│   ├── new_data.csv               # New data (prediction)
│   └── predictions_result.csv     # Prediction results output
│
├── materials/                     # Visualization outputs (PNG files)
│   ├── top_10_browsers.png
│   ├── device_distribution.png
│   ├── hourly_activity.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── ... (other PNG files)
│
├── models/                        # Saved model artifacts
│   ├── Encoding/                   # Saved LabelEncoders (pkl files)
│   │   ├── browser_encoder.pkl
│   │   ├── device_encoder.pkl
│   │   ├── os_encoder.pkl
│   │   ├── path_encoder.pkl
│   │   └── referrer_encoder.pkl
│   ├── Random_forest_model.pkl     # Trained Random Forest model
│   └── Scaler.pkl                  # StandardScaler object
│
├── main.py                         # Entrypoint with train() and predict() functions
├── utils.py                        # Core functions for data loading, preprocessing, training
├── processor.py                    # Prediction logic for new data
├── names.py                        # Constants and file-path configs
├── requirements.txt                # Python dependencies
└── README.md                        # This file
```

---

## 🛠️ Installation & Run

### 1️⃣ Clone the repo
```bash
git clone https://github.com/Karen042009/ML-Project.git
cd ML-Project
```

### 2️⃣ Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 3️⃣ Install requirements
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the project

By default, `main.py` runs the prediction pipeline on `names.DATA_FILE_PATH` when executed.

**Training the model** (use the `train()` function):
```python
# Edit the last line of main.py to call train(), or run this from a python one-liner
from main import train
train()
```

**Predicting on new data** (use the `predict()` function):
```python
from main import predict
predict("data/new_data.csv")
```

Then execute:
```bash
python main.py
```

---

## 📊 Results

### Model Performance
**Best model — Random Forest**
- **F2-Score:** ~0.995 ⭐
- **Recall (Bot):** ~0.995
- **Precision (Bot):** ~0.997

### What you should see after running
1. Visualization PNG files in `materials/` (about a dozen files)
2. Saved artifacts in `models/` including model, scaler and encoders
3. `data/predictions_result.csv` with detailed per-row prediction results

---

## 📦 Libraries Used

| Library | Minimum Version | Purpose |
|---------|------------------|---------|
| pandas | >=2.0.0 | Data manipulation and analysis |
| numpy | >=1.26.0 | Numeric computations |
| matplotlib | >=3.8.0 | Plotting and visualizations |
| seaborn | >=0.12.0 | Statistical visualizations |
| scikit-learn | >=1.3.0 | Machine learning models and preprocessing |
| joblib | >=1.3.0 | Model serialization/deserialization |

---

## 🔬 Research Methodology

### 1. Problem Framing
Detect automated bot activity in website logs which can:
- skew analytics
- overload servers
- pose security risks
- lead to incorrect business decisions

**Solution:** Behavioral-based Bot Detection.

### 1.1 Goals (SMART)
**Main objective:** Build an ML classifier with at least 95% F2-Score.

Key steps:
1. EDA and deep data cleaning
2. Behavioral feature engineering
3. Compare several classification algorithms
4. Choose and save the best-performing model

### 1.2 Cause Analysis (5 Whys)
1. Why are some visits suspicious? → Behavior differs from humans
2. How do they differ? → Very fast, high volume, unusual hours
3. Why? → Automated scripts
4. Why are scripts doing this? → Repeating actions triggered by bots
5. How does this show up in data? → Multiple requests from same IP; odd user-agents

Result: Created features like `visits_per_ip`, `time_since_last_visit_ip` and `is_user_agent_bot`.

### 2. Research & Planning
Literature review: Google Scholar, arXiv, and scikit-learn/pandas documentation pointed to behavior-based features as effective.

### 3. Implementation & Pipeline

Pipeline overview:
1. Load data → 2. Clean → 3. Feature engineering → 4. Encoding → 5. Scaling → 6. Modeling

Major pipeline steps:
- Missing values filled (`Unknown`, `direct`)
- Drop invalid timestamps
- Remove unused columns (`_id`, `date`, `time`)
- Feature engineering (per-IP stats, cyclic temporal features, UA bot indicators)
- Label encoding for categorical fields and storing encoders
- StandardScaler for normalization
- Random Forest with `class_weight='balanced'` trained on full dataset

### Tech stack & Implementation
- Language: Python 3
- IDE: Visual Studio Code (suggested)
- Modular structure: `main.py`, `utils.py`, `processor.py`, `names.py`.

### 4. Evaluation and Results
Compared 5 algorithms with a 70/30 train/test split.

| Model | F2-Score | Recall | Precision |
|-------|----------|--------|-----------|
| Random Forest (best) | ~0.995 | ~0.995 | ~0.997 |
| Gradient Boosting | ~0.990 | ~0.988 | ~0.995 |
| Logistic Regression | ~0.920 | ~0.910 | ~0.935 |
| Decision Tree | ~0.880 | ~0.870 | ~0.895 |
| SVM | ~0.905 | ~0.895 | ~0.920 |

Conclusions:
1. Behavioral features significantly improve detection
2. Random Forest provides the best Precision/Recall balance
3. The pipeline is ready for production deployment with additional validation and dataset coverage
4. Exceeded the goal of ≥95% F2-Score

Future directions:
- Hyperparameter tuning for boosting methods
- Real-time stream processing integration
- Experiment with deep learning models
- Add an API endpoint for production inference

---

## 💡 Usage Examples

### Training the model (Python snippet)
```python
from utils import *

clear_artifacts()
df = get_data()
df = clean_data(df)
df = engineer_features(df)
run_visualizations(df)
_, df_model_ready = encode_features(df)
train_model(df_model_ready)
```

### Predictions (Python snippet)
```python
import processor
import pandas as pd

new_data = pd.read_csv("data/new_data.csv")
results = processor.predict_new_data(new_data)
results.to_csv("data/predictions_result.csv", index=False)
print(results["is_bot_prediction"].value_counts())
```

---

## 🔗 Resources

- Scikit-learn: https://scikit-learn.org
- Pandas: https://pandas.pydata.org
- Seaborn: https://seaborn.pydata.org
- Research: Google Scholar / arXiv
- Repository: https://github.com/Karen042009/ML-Project

---

## 👨‍💻 Author

**Karen Poghosyan**
📧 Email: karen042009@example.com
🔗 GitHub: https://github.com/Karen042009

---

**⭐ If you find this project useful, please give it a star on GitHub!**