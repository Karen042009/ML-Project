import pandas as pd
import numpy as np
import joblib
import os
import names
from sklearn.preprocessing import LabelEncoder


def load_artifacts():
    if not os.path.exists(names.MODEL_PATH) or not os.path.exists(names.SCALER_PATH):
        raise FileNotFoundError(
            "Model or Scaler not found. Please train the model first."
        )

    model = joblib.load(names.MODEL_PATH)
    scaler = joblib.load(names.SCALER_PATH)

    encoders = {}
    if os.path.exists(names.ENCODING_DIR):
        for filename in os.listdir(names.ENCODING_DIR):
            if filename.endswith("_encoder.pkl"):
                col_name = filename.replace("_encoder.pkl", "")
                encoders[col_name] = joblib.load(
                    os.path.join(names.ENCODING_DIR, filename)
                )

    return model, scaler, encoders


def preprocess_new_data(new_df, fitted_encoders, fitted_scaler, feature_columns):
    if "referrer" in new_df.columns:
        new_df["referrer"] = new_df["referrer"].fillna("direct")

    for col in names.COLUMNS_TO_FILL_UNKNOWN:
        if col in new_df.columns:
            new_df[col] = new_df[col].fillna("Unknown")

    if "timestamp" in new_df.columns:
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], errors="coerce")
        new_df.dropna(subset=["timestamp"], inplace=True)

        new_df["hour"] = new_df["timestamp"].dt.hour
        new_df["day_of_week"] = new_df["timestamp"].dt.dayofweek

        new_df["hour_sin"] = np.sin(2 * np.pi * new_df["hour"] / 24)
        new_df["hour_cos"] = np.cos(2 * np.pi * new_df["hour"] / 24)
        new_df["day_of_week_sin"] = np.sin(2 * np.pi * new_df["day_of_week"] / 7)
        new_df["day_of_week_cos"] = np.cos(2 * np.pi * new_df["day_of_week"] / 7)
    if "is_user_agent_bot" not in new_df.columns:
        if "user_agent_raw" in new_df.columns:
            new_df["is_user_agent_bot"] = new_df["user_agent_raw"].str.contains(
                "bot|crawler|spider", case=False, na=False
            )
        else:
            new_df["is_user_agent_bot"] = False
    if "ip_address" in new_df.columns:
        new_df["visits_per_ip"] = new_df.groupby("ip_address")["timestamp"].transform(
            "count"
        )
        new_df["unique_paths_per_ip"] = new_df.groupby("ip_address")["path"].transform(
            "nunique"
        )

        new_df = new_df.sort_values(by=["ip_address", "timestamp"])
        time_diff = new_df.groupby("ip_address")["timestamp"].diff().dt.total_seconds()
        new_df["time_since_last_visit_ip"] = time_diff.fillna(0)
    for col, encoder in fitted_encoders.items():
        if col in new_df.columns:
            new_df[col] = new_df[col].astype(str)
            unique_new_values = new_df[col].unique()
            known_classes = set(encoder.classes_)

            new_df[col + "_encoded"] = new_df[col].apply(
                lambda x: x if x in known_classes else encoder.classes_[0]
            )
            new_df[col + "_encoded"] = encoder.transform(new_df[col + "_encoded"])
    for col in feature_columns:
        if col not in new_df.columns:
            new_df[col] = 0

    new_df_model_ready = (
        new_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    new_data_scaled = fitted_scaler.transform(new_df_model_ready)

    return new_data_scaled, new_df


def predict_new_data(df_raw):
    model, scaler, encoders = load_artifacts()

    if hasattr(model, "feature_names_in_"):
        feature_columns = model.feature_names_in_
    else:
        raise ValueError(
            "Model does not contain feature names. Please retrain with a newer scikit-learn version."
        )

    processed_data, processed_df = preprocess_new_data(
        df_raw.copy(), encoders, scaler, feature_columns
    )

    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)[:, 1]

    results_df = processed_df.copy()
    results_df["is_bot_prediction"] = predictions
    results_df["bot_probability"] = probabilities

    return results_df


if __name__ == "__main__":
    try:
        print("Loading data for prediction...")
        df = pd.read_csv(names.DATA_FILE_PATH)
        if "is_bot" in df.columns:
            df = df.drop(columns=["is_bot"])

        print("Running prediction...")
        results = predict_new_data(df)

        print("\nPrediction Results Summary:")
        print(
            results["is_bot_prediction"]
            .value_counts()
            .rename(index={True: "Bot", False: "Not Bot"})
        )

        print("\nSample Predictions:")
        print(results[["ip_address", "is_bot_prediction", "bot_probability"]].head())

    except Exception as e:
        print(f"Error: {e}")
