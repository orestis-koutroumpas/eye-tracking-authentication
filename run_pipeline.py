import subprocess
import argparse
import os

parser = argparse.ArgumentParser(description="Run full eye-tracking preprocessing pipeline")
parser.add_argument("--data_dir", required=True, help="Path to raw data folder")
parser.add_argument("--keystroke_file", default="keystrokes.csv", help="Keystroke CSV file")
parser.add_argument("--label", type=int, default=0, help="Label for this recording")
args = parser.parse_args()

data_dir = args.data_dir

# 1️ Filter data
print("Running filter_data.py ...")
subprocess.run(["python", "filter_data.py", "--data_dir", data_dir], check=True)

# 2️ Segment by keystrokes
print("Running segment_data_by_keystrokes.py ...")
subprocess.run([
    "python", "segment_data_by_keystrokes.py",
    "--data_dir", data_dir,
    "--keystroke_file", args.keystroke_file
], check=True)

# 3 Aggregate segment features
segmented_dir = os.path.join(data_dir, "Segmentation")
print("Running aggregate_segments_to_features.py ...")
subprocess.run([
    "python", "aggregate_segments_to_features.py",
    "--segmented_dir", segmented_dir,
    "--label", str(args.label)
], check=True)

print("Pipeline finished successfully!")
