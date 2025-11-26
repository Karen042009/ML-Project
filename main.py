from utils import get_data, clean_data, engineer_features, encode_features, run_visualizations, train_model

def main():
    try:
        df = get_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

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

    print("Finished processing data.")

if __name__ == "__main__":
    main()
