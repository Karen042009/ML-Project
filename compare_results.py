import pandas as pd
import processor
import names
import os
from sklearn.metrics import classification_report, confusion_matrix


def compare_last_rows():
    print(f"Loading {names.DATA_FILE_PATH}...")
    df_all = pd.read_csv(names.DATA_FILE_PATH)
    print("Extracting last 255 rows...")
    df_test = df_all.tail(255).copy()
    df_test.to_csv(names.DATA_NEW_FILE_PATH, index=False)
    print(f"Saved test data to {names.DATA_NEW_FILE_PATH}")
    X_test = df_test.drop(columns=["is_bot"])
    y_true = df_test["is_bot"]
    print("Running predictions...")
    results = processor.predict_new_data(X_test)
    y_pred = results["is_bot_prediction"]
    print("\n" + "=" * 30)
    print("Comparison Result (Last 255 rows)")
    print("=" * 30)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Not Bot", "Bot"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    misclassified = results[y_true.values != y_pred.values].copy()
    misclassified["true_label"] = y_true.values[y_true.values != y_pred.values]
    if not misclassified.empty:
        print("\nMisclassified Rows Sample:")
        print(
            misclassified[
                ["ip_address", "true_label", "is_bot_prediction", "bot_probability"]
            ].head()
        )
    else:
        print("\nAll 255 rows were predicted correctly!")


if __name__ == "__main__":
    if not os.path.exists(names.FINAL_MODEL_PATH):
        print(
            "Error: Model not found. Please train first by running: python3 main.py (with train() called)"
        )
    else:
        compare_last_rows()
