import os
import pandas as pd

folder = "data/demo/impostor_1"          
events_path = "data/demo/impostor_1/events.csv" 
events_df = pd.read_csv(events_path)
recording_start_ns = int(events_df.loc[0, "timestamp [ns]"])
print(f"Recording start timestamp: {recording_start_ns}")

# === Process all other CSV files in the folder ===
for fname in os.listdir(folder):
    if not fname.endswith(".csv"):
        continue
    if fname == 'events.csv' or fname == 'jason.csv':
        continue

    fpath = os.path.join(folder, fname)
    print(f"Processing {fname}...")

    df = pd.read_csv(fpath)

    # Find all columns containing "timestamp"
    timestamp_cols = [c for c in df.columns if "timestamp" in c.lower()]

    if not timestamp_cols:
        continue

    # Subtract start time from each timestamp column
    for col in timestamp_cols:
        df[col] = df[col].astype(int) - recording_start_ns

    # Save back to same file (overwrite)
    df.to_csv(fpath, index=False)
    print(f"  Updated and saved {fname}")