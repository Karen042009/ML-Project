from utils import get_data, clean_data, engineer_features, encode_features, run_visualizations, train_model
import sys

def main():
    """
    Main execution function for the ML pipeline.
    Orchestrates data loading, cleaning, feature engineering, visualization, and model training.
    """
    print("Starting ML Pipeline...")

    try:
        print("Loading data...")
        df = get_data()
    except Exception as e:
        print(f"Critical Error loading data: {e}")
        sys.exit(1)

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

        print("Finished processing data successfully.")
        
    except Exception as e:
        print(f"An error occurred during the pipeline execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
