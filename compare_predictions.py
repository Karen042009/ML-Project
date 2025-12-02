import pandas as pd

def compare_results():
    pred_path = 'data/predictions_result.csv'
    ground_truth_path = 'data/visits2.csv'

    try:
        # Load predictions and ground truth
        df_pred = pd.read_csv(pred_path)
        df_true = pd.read_csv(ground_truth_path)

        # Merge on '_id' to ensure we are comparing the same records
        # We only care about the rows that are in df_pred (the new data)
        merged_df = pd.merge(df_pred, df_true[['_id', 'is_bot']], on='_id', how='inner', suffixes=('', '_true'))

        if 'is_bot' not in merged_df.columns and 'is_bot_true' in merged_df.columns:
             merged_df.rename(columns={'is_bot_true': 'is_bot'}, inplace=True)
        elif 'is_bot' in merged_df.columns and 'is_bot_true' in merged_df.columns:
             # If both exist (unlikely given previous steps, but possible), use the one from visits2 (is_bot_true)
             merged_df['is_bot'] = merged_df['is_bot_true']
             merged_df.drop(columns=['is_bot_true'], inplace=True)

        # Calculate matches
        # is_bot_prediction is boolean (True/False) or 0/1? 
        # Let's check the types. Usually pandas reads "True"/"False" as booleans if possible, 
        # but sometimes strings.
        
        # Ensure boolean types for comparison
        merged_df['is_bot_prediction'] = merged_df['is_bot_prediction'].astype(bool)
        merged_df['is_bot'] = merged_df['is_bot'].astype(bool)

        correct_predictions = merged_df[merged_df['is_bot_prediction'] == merged_df['is_bot']]
        incorrect_predictions = merged_df[merged_df['is_bot_prediction'] != merged_df['is_bot']]

        total = len(merged_df)
        correct_count = len(correct_predictions)
        incorrect_count = len(incorrect_predictions)
        accuracy = (correct_count / total) * 100 if total > 0 else 0

        print(f"\n=== Comparison Results ===")
        print(f"Total predictions compared: {total}")
        print(f"Correct predictions: {correct_count}")
        print(f"Incorrect predictions: {incorrect_count}")
        print(f"Accuracy: {accuracy:.2f}%")

        print("\nConfusion Matrix (Predicted vs Actual):")
        print(pd.crosstab(merged_df['is_bot'], merged_df['is_bot_prediction'], rownames=['Actual'], colnames=['Predicted']))

        if incorrect_count > 0:
            print("\nSample Incorrect Predictions:")
            print(incorrect_predictions[['_id', 'ip_address', 'user_agent_raw', 'is_bot', 'is_bot_prediction']].head())

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    compare_results()
