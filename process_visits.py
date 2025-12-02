import pandas as pd
import os

def process_visits():
    visits_path = 'data/visits.csv'
    visits2_path = 'data/visits2.csv'
    output_path = 'data/new_data.csv'

    if not os.path.exists(visits_path) or not os.path.exists(visits2_path):
        print("Error: One or both input files do not exist.")
        return

    try:
        # Load the datasets
        df_visits = pd.read_csv(visits_path)
        df_visits2 = pd.read_csv(visits2_path)

        # Identify rows in visits2 that are not in visits
        # We assume '_id' is the unique identifier. 
        # If not, we might need to merge on all columns or a subset.
        # Using _id is the most robust method for this type of data.
        new_data = df_visits2[~df_visits2['_id'].isin(df_visits['_id'])]

        print(f"Found {len(new_data)} new rows in visits2.")

        # Remove the 'is_bot' column
        if 'is_bot' in new_data.columns:
            new_data = new_data.drop(columns=['is_bot'])
            print("Removed 'is_bot' column.")
        else:
            print("'is_bot' column not found in the new data.")

        # Save the result to a new CSV
        new_data.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    process_visits()
