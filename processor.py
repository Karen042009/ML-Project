import pandas as pd
import joblib
import os
import names
from utils import AdvancedBotPreprocessor


def predict_new_data(df_raw):
    if not os.path.exists(names.FINAL_MODEL_PATH):
        raise FileNotFoundError("Pipeline not found. Please train the model first.")
    print("Loading optimized pipeline...")
    pipeline = joblib.load(names.FINAL_MODEL_PATH)
    print("Executing prediction pipeline...")
    predictions = pipeline.predict(df_raw)
    probabilities = pipeline.predict_proba(df_raw)[:, 1]
    results_df = df_raw.copy()
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
