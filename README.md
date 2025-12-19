# 📊 Web Traffic Bot Detection Using Machine Learning

**Author:** Karen Poghosyan
**Date:** 19.12.2025

---

## 🎯 Summary

This project detects automated bot activity in web traffic using advanced machine learning techniques. It utilizes a behavioral-based approach, transforming raw server logs into rich behavioral features. The core of the project is a **scikit-learn Pipeline** that integrates custom preprocessing, categorical encoding, feature scaling, and a **Random Forest** classifier (F2-Score ~0.995).

---

## 🚀 Key Features

### 🛠️ Advanced Pipeline Architecture
- **Unified Pipeline**: Preprocessing, feature engineering, scaling, and modeling are all bundled into a single `.pkl` artifact.
- **Custom Transformers**: `AdvancedBotPreprocessor` handles behavioral logic like IP-based statistics and temporal cycles.
- **Robust Handling**: Intelligent imputation for missing values and handling of unknown categorical levels.

### 📊 Behavioral Feature Engineering
- **IP-Based Metrics**: `visits_per_ip`, `unique_paths_per_ip`, `time_since_last_visit_ip`.
- **Temporal Analysis**: Cyclic encoding of hours and days (sine/cosine transformation).
- **User Agent Intelligence**: Detection of bot signatures in raw user-agent strings.

---

## 📂 Project Structure

```
ML-Project/
│
├── data/                          # Input data and prediction output
│   ├── visits2.csv                # Primary training data
│   ├── new_data.csv               # Data for prediction (last 255 rows test)
│   └── predictions_result.csv     # Final prediction results
│
├── materials/                     # Visualization outputs (PNG files)
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── features_pairplot.png
│   └── ... (other EDA charts)
│
├── models/                        # Saved model artifacts
│   └── Final_Bot_Detection_Model.pkl  # The complete scikit-learn Pipeline
│
├── main.py                         # Entrypoint for training and basic prediction
├── utils.py                        # Pipeline definition and custom transformers
├── processor.py                    # Production-ready prediction logic
├── compare_results.py              # Validation script for testing accuracy
├── names.py                        # Project-wide constants and paths
└── requirements.txt                # Python dependencies
```

---

## 🛠️ Installation & Run

### 1️⃣ Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Training the Model
To re-train the entire pipeline and update visualizations:
```python
from main import train
train()
```

### 3️⃣ Running Predictions
To perform predictions on new raw data:
```bash
python3 main.py
```
This uses `processor.py` to load the `Final_Bot_Detection_Model.pkl` and process the data defined in `names.DATA_NEW_FILE_PATH`.

### 4️⃣ Validation
To compare predictions against labeled data (last 255 rows of `visits2.csv`):
```bash
python3 compare_results.py
```

---

## 📈 Performance
- **Model**: Random Forest Classifier
- **F1/F2 Score**: ~0.995-1.00 ⭐
- **Accuracy**: 100% on recent test batches

---

## 👨‍💻 Author
**Karen Poghosyan**
🔗 GitHub: [Karen042009](https://github.com/Karen042009)