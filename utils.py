import pandas as pd
import sys
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, Optional
import names


class AdvancedBotPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ip_history_ = {}
        self.path_history_ = {}

    def fit(self, X, y=None):
        df = X.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            if "ip_address" in df.columns:
                self.ip_history_ = (
                    df.groupby("ip_address")["timestamp"].count().to_dict()
                )
                if "path" in df.columns:
                    self.path_history_ = (
                        df.groupby("ip_address")["path"].nunique().to_dict()
                    )
        return self

    def transform(self, X):
        df = X.copy()
        if "referrer" in df.columns:
            df["referrer"] = df["referrer"].fillna("direct")
        cols_fill = ["hostname", "browser", "os", "device", "user_agent_raw", "path"]
        for col in cols_fill:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            if df["timestamp"].isnull().any():
                df["timestamp"] = df["timestamp"].fillna(pd.Timestamp.now())
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        if "user_agent_raw" in df.columns:
            df["is_user_agent_bot"] = (
                df["user_agent_raw"]
                .str.contains("bot|crawler|spider", case=False, na=False)
                .astype(int)
            )
        else:
            df["is_user_agent_bot"] = 0
        if "ip_address" in df.columns:
            df["visits_per_ip"] = df["ip_address"].map(self.ip_history_).fillna(1)
            if "path" in df.columns:
                df["unique_paths_per_ip"] = (
                    df["ip_address"].map(self.path_history_).fillna(1)
                )
            else:
                df["unique_paths_per_ip"] = 1
            df["time_since_last_visit_ip"] = 0
        else:
            df["visits_per_ip"] = 1
            df["unique_paths_per_ip"] = 1
            df["time_since_last_visit_ip"] = 0
        return df


def get_data(file_path: str = names.DATA_FILE_PATH) -> pd.DataFrame:
    try:
        df_raw = pd.read_csv(file_path)
        df = df_raw.copy()
        print(f"'{file_path}' file loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: '{file_path}' file not found.")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if "referrer" in df.columns:
        df["referrer"] = df["referrer"].fillna("direct")
    for col in names.COLUMNS_TO_FILL_UNKNOWN:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)
    df.drop(columns=names.COLUMNS_TO_DROP_INITIAL, inplace=True, errors="ignore")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    if "user_agent_raw" in df.columns:
        df["is_user_agent_bot"] = df["user_agent_raw"].str.contains(
            "bot|crawler|spider", case=False, na=False
        )
    if "ip_address" in df.columns:
        df["visits_per_ip"] = df.groupby("ip_address")["timestamp"].transform("count")
        df["unique_paths_per_ip"] = df.groupby("ip_address")["path"].transform(
            "nunique"
        )
        df = df.sort_values(by=["ip_address", "timestamp"])
        grouped_ts = df.groupby("ip_address")["timestamp"]
        time_diff = pd.to_timedelta(grouped_ts.diff()).dt.total_seconds()
        df["time_since_last_visit_ip"] = time_diff.fillna(0)
    if "timestamp" in df.columns:
        ts_series = pd.to_datetime(df["timestamp"])
        df["hour"] = ts_series.dt.hour
        df["day_of_week"] = ts_series.dt.dayofweek
        df["day_of_week_name"] = ts_series.dt.day_name()
        df["date_only"] = ts_series.dt.date
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    encoders = {}
    os.makedirs(names.ENCODING_DIR, exist_ok=True)
    for col in names.CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            joblib.dump(le, os.path.join(names.ENCODING_DIR, f"{col}_encoder.pkl"))
    df_model_ready = df.drop(columns=names.COLUMNS_TO_DROP_FOR_MODEL, errors="ignore")
    df_model_ready = df_model_ready.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df, df_model_ready


def run_visualizations(dataframe: pd.DataFrame) -> None:
    sns.set(style="whitegrid")
    os.makedirs(names.MATERIALS_DIR, exist_ok=True)
    if "browser" in dataframe.columns:
        plt.figure(figsize=(12, 7))
        plt.title("Top 10 Most Common Browsers", fontsize=16)
        sns.countplot(
            y="browser",
            data=dataframe,
            order=dataframe["browser"].value_counts().iloc[:10].index,
            palette="viridis",
            hue="browser",
            legend=False,
        )
        plt.xlabel("Number of Visits")
        plt.ylabel("Browser")
        plt.savefig(names.TOP_BROWSERS_IMAGE_PATH)
        plt.close()
    if (
        "is_mobile" in dataframe.columns
        and "is_pc" in dataframe.columns
        and "is_tablet" in dataframe.columns
    ):
        device_counts = [
            dataframe["is_mobile"].sum(),
            dataframe["is_pc"].sum(),
            dataframe["is_tablet"].sum(),
        ]
        device_labels = ["Mobile", "PC", "Tablet"]
        plt.figure(figsize=(8, 8))
        plt.title("Visit Distribution by Device Type", fontsize=16)
        plt.pie(
            device_counts,
            labels=device_labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=["#ff9999", "#66b3ff", "#99ff99"],
        )
        plt.axis("equal")
        plt.savefig(names.DEVICE_DISTRIBUTION_IMAGE_PATH)
        plt.close()
    if "path" in dataframe.columns:
        plt.figure(figsize=(12, 8))
        sns.countplot(
            y="path",
            data=dataframe,
            order=dataframe["path"].value_counts().iloc[:10].index,
            palette="plasma",
            hue="path",
            legend=False,
        )
        plt.title("Top 10 Most Visited Pages", fontsize=16)
        plt.xlabel("Number of Visits")
        plt.ylabel("Page Path")
        plt.savefig(names.TOP_PAGES_IMAGE_PATH)
        plt.close()
    if "referrer" in dataframe.columns:
        plt.figure(figsize=(12, 7))
        referrer_data = dataframe[dataframe["referrer"] != "direct"]
        if not referrer_data.empty:
            sns.countplot(
                y="referrer",
                data=referrer_data,
                order=referrer_data["referrer"].value_counts().iloc[:10].index,
                palette="ocean",
                hue="referrer",
                legend=False,
            )
            plt.title('Top 10 Referrer Sources (excluding "direct")', fontsize=16)
            plt.xlabel("Number of Visits")
            plt.ylabel("Source")
            plt.savefig(names.TOP_REFERRERS_IMAGE_PATH)
            plt.close()
    if "hour" in dataframe.columns:
        plt.figure(figsize=(14, 6))
        sns.countplot(
            x="hour", data=dataframe, palette="magma", hue="hour", legend=False
        )
        plt.title("Visit Activity by Hour of Day", fontsize=16)
        plt.xlabel("Hour of Day (0-23)")
        plt.ylabel("Number of Visits")
        plt.savefig(names.HOURLY_ACTIVITY_IMAGE_PATH)
        plt.close()
    if "day_of_week_name" in dataframe.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(
            x="day_of_week_name",
            data=dataframe,
            order=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            palette="cubehelix",
            hue="day_of_week_name",
            legend=False,
        )
        plt.title("Visit Activity by Day of Week", fontsize=16)
        plt.xlabel("Day of Week")
        plt.ylabel("Number of Visits")
        plt.xticks(rotation=45)
        plt.savefig(names.WEEKLY_ACTIVITY_IMAGE_PATH)
        plt.close()
    if "date_only" in dataframe.columns:
        daily_visits = dataframe.groupby("date_only").size()
        plt.figure(figsize=(15, 6))
        daily_visits.plot(kind="line", marker="o", linestyle="-")
        plt.title("Daily Visit Trend", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Number of Visits")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(names.DAILY_VISITS_TREND_IMAGE_PATH)
        plt.close()
    if "is_bot" in dataframe.columns:
        target_counts = dataframe["is_bot"].value_counts()
        plt.figure(figsize=(8, 5))
        sns.barplot(
            x=target_counts.index,
            y=target_counts.values,
            palette="Reds_r",
            hue=target_counts.index,
            legend=False,
        )
        plt.title('Class Imbalance ("is_bot")', fontsize=16)
        plt.ylabel("Number of Visits")
        plt.xticks(
            [0, 1],
            labels=[
                f"Not Bot ({target_counts.get(False, 0)})",
                f"Bot ({target_counts.get(True, 0)})",
            ],
        )
        plt.savefig(names.CLASS_IMBALANCE_IMAGE_PATH)
        plt.close()
    important_cols = [
        "visits_per_ip",
        "unique_paths_per_ip",
        "time_since_last_visit_ip",
        "is_user_agent_bot",
        "hour_sin",
        "day_of_week_sin",
        "is_bot",
    ]
    if all(col in dataframe.columns for col in important_cols):
        plot_df = dataframe.copy()[important_cols]
        plot_df["is_bot"] = plot_df["is_bot"].astype(bool)
        sns.set_palette("Paired")
        g = sns.pairplot(
            plot_df,
            hue="is_bot",
            diag_kind="kde",
            height=2.5,
            vars=[c for c in important_cols if c != "is_bot"],
            palette={True: "red", False: "blue"},
        )
        g.fig.suptitle('Pairwise Relationships by "is_bot" Class', y=1.02, fontsize=16)
        legend = getattr(g, "_legend", None)
        if legend is not None:
            new_labels = ["Not Bot", "Bot"]
            for t, l in zip(legend.texts, new_labels):
                t.set_text(l)
            legend.set_title("Class")
        plt.savefig(names.PAIRPLOT_IMAGE_PATH)
        plt.close()


def train_model(df_train: pd.DataFrame) -> None:
    if "is_bot" not in df_train.columns:
        print("Error: 'is_bot' column not found in dataframe.")
        return
    X = df_train.drop("is_bot", axis=1)
    y = df_train["is_bot"]
    cat_cols = ["browser", "os", "device", "referrer", "path"]
    num_cols = [
        "visits_per_ip",
        "unique_paths_per_ip",
        "time_since_last_visit_ip",
        "is_user_agent_bot",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    ]
    print("Defining Pipeline...")
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="Unknown"),
                        ),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    final_pipeline = Pipeline(
        [
            ("feature_eng", AdvancedBotPreprocessor()),
            ("prep", preprocessor),
            ("model", RandomForestClassifier(**names.RANDOM_FOREST_PARAMS)),
        ]
    )
    print("Training Final Pipeline on ALL data...")
    final_pipeline.fit(X, y)
    print("\n--- Model Evaluation (Training Set) ---")
    y_pred = final_pipeline.predict(X)
    print(classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Not Bot", "Bot"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Training Set)")
    plt.grid(False)
    plt.savefig(names.CONFUSION_MATRIX_IMAGE_PATH)
    plt.close()
    model = final_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = num_cols + cat_cols
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(
            range(len(importances)), [feature_names[i] for i in indices], rotation=90
        )
        plt.tight_layout()
        plt.savefig(names.FEATURE_IMPORTANCE_IMAGE_PATH)
        plt.close()
    os.makedirs(names.MODELS_DIR, exist_ok=True)
    joblib.dump(final_pipeline, names.FINAL_MODEL_PATH)
    print(f"Pipeline saved to {names.FINAL_MODEL_PATH}")


def clear_artifacts() -> None:
    dirs_to_clean = [names.MATERIALS_DIR, names.MODELS_DIR]
    print("Clearing artifacts...")
    for directory in dirs_to_clean:
        if not os.path.exists(directory):
            continue
        for root, dirs, files in os.walk(directory, topdown=False):
            for file in files:
                if file != ".gitkeep":
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
            if root != directory and not os.listdir(root):
                try:
                    os.rmdir(root)
                    print(f"Removed directory: {root}")
                except Exception as e:
                    print(f"Error removing directory {root}: {e}")
    print("Artifacts cleared.")
