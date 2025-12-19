import os

# Paths
DATA_FILE_PATH = "data/visits2.csv"
DATA_NEW_FILE_PATH = "data/new_data.csv"
MODELS_DIR = "models"
ENCODING_DIR = os.path.join(MODELS_DIR, "Encoding")
MATERIALS_DIR = "materials"

# Image Paths
TOP_BROWSERS_IMAGE_PATH = os.path.join(MATERIALS_DIR, "top_10_browsers.png")
DEVICE_DISTRIBUTION_IMAGE_PATH = os.path.join(MATERIALS_DIR, "device_distribution.png")
TOP_PAGES_IMAGE_PATH = os.path.join(MATERIALS_DIR, "top_10_pages.png")
TOP_REFERRERS_IMAGE_PATH = os.path.join(MATERIALS_DIR, "top_10_referrers.png")
HOURLY_ACTIVITY_IMAGE_PATH = os.path.join(MATERIALS_DIR, "hourly_activity.png")
WEEKLY_ACTIVITY_IMAGE_PATH = os.path.join(MATERIALS_DIR, "weekly_activity.png")
DAILY_VISITS_TREND_IMAGE_PATH = os.path.join(MATERIALS_DIR, "daily_visits_trend.png")
CLASS_IMBALANCE_IMAGE_PATH = os.path.join(MATERIALS_DIR, "class_imbalance.png")
PAIRPLOT_IMAGE_PATH = os.path.join(MATERIALS_DIR, "features_pairplot.png")
CONFUSION_MATRIX_IMAGE_PATH = os.path.join(MATERIALS_DIR, "confusion_matrix.png")
FEATURE_IMPORTANCE_IMAGE_PATH = os.path.join(MATERIALS_DIR, "feature_importance.png")

# Model Artifact Paths
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "Final_Bot_Detection_Model.pkl")

# Data Cleaning & Feature Engineering
COLUMNS_TO_FILL_UNKNOWN = [
    "hostname",
    "browser_version",
    "os_version",
    "browser",
    "os",
    "device",
    "user_agent_raw",
]

COLUMNS_TO_DROP_INITIAL = ["_id", "date", "time"]

CATEGORICAL_COLS = ["browser", "os", "device", "referrer", "path"]

COLUMNS_TO_DROP_FOR_MODEL = [
    "user_agent_raw",
    "ip_address",
    "hostname",
    "timestamp",
    "date_only",
    "day_of_week_name",
    "browser",
    "os",
    "device",
    "referrer",
    "path",
    "browser_version",
    "os_version",
    "hour",
    "day_of_week",
]

# Model Hyperparameters
RANDOM_FOREST_PARAMS = {
    "criterion": "gini",
    "n_estimators": 200,
    "max_depth": 15,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}
