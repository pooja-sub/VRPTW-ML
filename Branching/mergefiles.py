import pandas as pd
import glob

# Path to your CSV files
csv_files = glob.glob('TrainingData/*.csv')  # Change path as needed

# Read and concatenate all CSVs
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Save to a new CSV
merged_df.to_csv('merged_features_vrptw.csv', index=False)
