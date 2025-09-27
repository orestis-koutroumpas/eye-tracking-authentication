import os
import pandas as pd

DATA_DIR = "data/demo/genuine_1"

CSV_FILES = [
    "3d_eye_states.csv",
    "blinks.csv",
    "events.csv",
    "fixations.csv",
    "gaze.csv",
    "imu.csv",
    "saccades.csv",
    "template.csv",
    "world_timestamps.csv"
]


def load_exp_data(data_dir=DATA_DIR, csv_files=CSV_FILES):
    dataframes = {}
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                key = file.replace('.csv', '')
                dataframes[key] = df
                print(f"Loaded {file} with {len(df)} rows and {len(df.columns)} columns.")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        else:
            print(f"File not found: {file}")
    return dataframes

if __name__ == "__main__":
    data = load_exp_data()
    breakpoint()