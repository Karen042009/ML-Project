import pandas as pd
import sys
import seaborn as sns
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import Tuple, Optional
import names

def get_data(file_path: str = names.DATA_FILE_PATH) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        df_raw = pd.read_csv(file_path)
        df = df_raw.copy()
        print(f"'{file_path}' file loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: '{file_path}' file not found.")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataframe by filling missing values and dropping unnecessary columns.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
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
    """
    Create new features from existing columns.

    Args:
        df (pd.DataFrame): Dataframe with cleaned data.

    Returns:
        pd.DataFrame: Dataframe with new features.
    """
    if "user_agent_raw" in df.columns:
        df["is_user_agent_bot"] = df["user_agent_raw"].str.contains(
            "bot|crawler|spider", case=False, na=False
        )

    if "ip_address" in df.columns:
        df["visits_per_ip"] = df.groupby("ip_address")["timestamp"].transform("count")
        df["unique_paths_per_ip"] = df.groupby("ip_address")["path"].transform("nunique")
        
        df = df.sort_values(by=["ip_address", "timestamp"])
        time_diff = df.groupby("ip_address")["timestamp"].diff().dt.total_seconds()
        df["time_since_last_visit_ip"] = time_diff.fillna(time_diff.mean())

    if "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_of_week_name"] = df["timestamp"].dt.day_name()
        df["date_only"] = df["timestamp"].dt.date

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    return df

def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode categorical features and prepare data for modeling.

    Args:
        df (pd.DataFrame): Dataframe with engineered features.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - Original dataframe with encoded columns.
            - Model-ready dataframe (numeric only).
    """
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
    """
    Generate and save various visualizations.

    Args:
        dataframe (pd.DataFrame): Dataframe to visualize.
    """
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
        sns.countplot(x="hour", data=dataframe, palette="magma", hue="hour", legend=False)
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
        sns.barplot(x=target_counts.index, y=target_counts.values, palette="Reds_r", hue=target_counts.index, legend=False)
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

        g.fig.suptitle(
            'Pairwise Relationships by "is_bot" Class', y=1.02, fontsize=16
        )

        new_labels = ["Not Bot", "Bot"]
        for t, l in zip(g._legend.texts, new_labels):
            t.set_text(l)

        g._legend.set_title("Class")
        plt.savefig(names.PAIRPLOT_IMAGE_PATH)
        plt.close()


def train_model(df_model_ready: pd.DataFrame) -> None:
    """
    Train a Random Forest model and evaluate it.

    Args:
        df_model_ready (pd.DataFrame): Dataframe ready for modeling (numeric only).
    """
    if "is_bot" not in df_model_ready.columns:
        print("Error: 'is_bot' column not found in dataframe.")
        return

    X = df_model_ready.drop("is_bot", axis=1)
    y = df_model_ready["is_bot"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest Model...")
    model = RandomForestClassifier(**names.RANDOM_FOREST_PARAMS)
    model.fit(X_train_scaled, y_train)

    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

    os.makedirs(names.MODELS_DIR, exist_ok=True)

    joblib.dump(scaler, names.SCALER_PATH)
    joblib.dump(model, names.MODEL_PATH)
    print(f"Model saved to {names.MODELS_DIR}")