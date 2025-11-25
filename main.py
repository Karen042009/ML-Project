from utils import get_data, clean_data, engineer_features, encode_features, run_visualizations, train_model

def main():
    # 1. Load Data
    try:
        df = get_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Clean Data
    print("Cleaning data...")
    df = clean_data(df)

    # 3. Feature Engineering
    print("Engineering features...")
    df = engineer_features(df)

    # 4. Visualization
    print("Running visualizations...")
    run_visualizations(df)

    # 5. Encoding & Preparation for Model
    print("Encoding features...")
    _, df_model_ready = encode_features(df)

    # 6. Train Model
    print("Training model...")
    train_model(df_model_ready)

    print("Finished processing data.")

if __name__ == "__main__":
    main()
