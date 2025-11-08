import argparse
import os
import logging
from pathlib import Path
import pandas as pd
from utils.plotting import *


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
            dirnames.remove("Segmentation") 

        # Only continue if there are CSV files in this folder
        csv_files = [f for f in filenames if f.endswith(".csv")]
        if not csv_files:
            continue
    # dirpath = 'data/genuine/orestis_117-86becbd0'
        print(f"Processing folder: {dirpath}")
        plot_gaze_over_time(dirpath)
        compare_filter_unfiltered_data(dirpath.replace('data_filtered', 'data_unfiltered'), dirpath)
        plot_gaze_heatmap(dirpath)
        plot_fixation_spatial_map(dirpath)
        plot_fixation_duration_histogram(dirpath)
        plot_pupil_diameter_over_time(dirpath)
        plot_eyelid_aperture_over_time(dirpath)
        plot_eyelid_angles_over_time(dirpath)
        plot_distance_between_pupils_over_time(dirpath)
        
        breakpoint()
