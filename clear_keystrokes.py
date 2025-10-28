import os
import pandas as pd

def clean_keystrokes(root_dir):
    # Walk through all directories and files
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file == "keystrokes.csv":
                file_path = os.path.join(dirpath, file)                
                try:
                    # Load CSV
                    df = pd.read_csv(file_path)
    
                    # Filter out unwanted rows
                    df_clean = df[~df['name'].isin(['Tab_pressed', 'Shift_pressed', 'CapsLock_pressed', 'Enter_pressed'])]

                    if len(df) > 31:
                        print(file_path)
                    # Save cleaned CSV back to same file
                    df_clean.to_csv(file_path, index=False)

                except Exception as e:
                    print(f"⚠ Error processing {file_path}: {e}")

if __name__ == "__main__":
    directory = "data/raw_data"
    clean_keystrokes(directory)
