import argparse
import os
import logging
from pathlib import Path
import pandas as pd
from utils.plotting import plot_trajectory_heatmap, plot_fixation_spatial_map, plot_eyelid_aperture_over_time, plot_pupil_diameter, plot_saccade_velo_over_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full eye-tracking preprocessing pipeline"
    )
    parser.add_argument(
        "--data_dir", 
        required=True, 
        help="Path to raw data folder"
    )
    args = parser.parse_args()
    
    for dirpath, dirnames, filenames in os.walk(args.data_dir):
        # Skip any folder named 'Segmentation'
        if "Segmentation" in dirnames:
            dirnames.remove("Segmentation")  # this prevents os.walk from descending into it

        # Only continue if there are CSV files in this folder
        csv_files = [f for f in filenames if f.endswith(".csv")]
        if not csv_files:
            continue

        print(f"Processing folder: {dirpath}")

        # Call your plotting functions
        plot_trajectory_heatmap(dirpath)
        plot_fixation_spatial_map(dirpath)
        plot_pupil_diameter(dirpath)
        plot_eyelid_aperture_over_time(dirpath)
        plot_saccade_velo_over_time(dirpath)