from utils import *
import processor
import sys
import pandas as pd
import names


def train():
    print("\n=== Starting Training Pipeline ===")

    try:
        clear_artifacts()
    except Exception as e:
        print(f"Warning: Could not clear artifacts: {e}")
    try:
        print("Loading training data...")
        df = get_data()
    except Exception as e:
        print(f"Critical Error loading data: {e}")
        return

    try:
        print("Cleaning data...")
        df = clean_data(df)

        print("Engineering features...")
        df = engineer_features(df)

        print("Running visualizations...")
        run_visualizations(df)

        print("Encoding features...")
        _, df_model_ready = encode_features(df)

        print("Training model...")
        train_model(df_model_ready)

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
        print("Data loaded successfully.")

        print("Predicting...")
        results = processor.predict_new_data(new_data)

        print("\nPrediction Results Summary:")
        print(
            results["is_bot_prediction"]
            .value_counts()
            .rename(index={True: "Bot", False: "Not Bot"})
        )

        output_path = "data/predictions_result.csv"
        results.to_csv(output_path, index=False)
        print(f"\nDetailed predictions saved to {output_path}")
        print("Prediction phase completed.")

    except Exception as e:
        print(f"An error occurred during the prediction phase: {e}")


predict(names.DATA_FILE_PATH)
