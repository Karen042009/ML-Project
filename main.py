from utils import *
import processor
import sys
import pandas as pd
import names


def train():
    print("\n=== Starting Pipeline Training ===")
    try:
        clear_artifacts()
    except Exception as e:
        print(f"Warning: Could not clear artifacts: {e}")
    try:
        print(f"Loading data from {names.DATA_FILE_PATH}...")
        df = get_data()
        print("Cleaning and preparing data for EDA...")
        df_eda = clean_data(df)
        df_eda = engineer_features(df_eda)
        print("Running visualizations...")
        run_visualizations(df_eda)
        print("Training the Final Pipeline...")
        train_model(clean_data(df))
        print("Finished training successfully.")
    except Exception as e:
        print(f"An error occurred during the training pipeline: {e}")


def predict(file_path):
    print(f"\n=== Starting Prediction Pipeline for: {file_path} ===")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    try:
        new_data = pd.read_csv(file_path)
        print("Data loaded correctly.")
        if "is_bot" in new_data.columns:
            new_data = new_data.drop(columns=["is_bot"])
        print("Running optimized predictions...")
        results = processor.predict_new_data(new_data)
        print("\nPrediction Results Summary:")
        stats = (
            results["is_bot_prediction"]
            .value_counts()
            .rename(index={True: "Bot", False: "Not Bot"})
        )
        print(stats)
        output_path = "data/predictions_result.csv"
        results.to_csv(output_path, index=False)
        print(f"\nDetailed predictions saved to {output_path}")
        print("Prediction phase completed.")
    except Exception as e:
        print(f"An error occurred during the prediction phase: {e}")


if __name__ == "__main__":
    predict("data/new_data.csv")
