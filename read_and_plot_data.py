import sqlite3
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# === Load Metadata ===
info_path = r"\info.json"
with open(info_path, 'r') as f:
    meta = json.load(f)

recording_id = meta.get("recording_id")
wearer = meta.get("wearer_name", "N/A")
duration_ns = meta.get("duration", 0)
duration_sec = round(duration_ns / 1e9, 2)
gaze_freq = meta.get("gaze_frequency", "N/A")

# === Load tables based on recording_id ===
gaze_df = pd.read_csv('gaze.csv')
fixations_df = pd.read_csv('fixations.csv')
imu_df = pd.read_csv('imu.csv')
saccades_df = pd.read_csv('saccades.csv')
blinks_df = pd.read_csv('blinks.csv')
eye3d_df = pd.read_csv('eye3d.csv')

# ---------- GAZE DATA PLOTS ----------

# --- Convert timestamps to seconds ---
gaze_df['time_s'] = gaze_df['timestamp_ns'] / 1e9

# --- Calculate gaze velocity ---
gaze_df['gaze_dx'] = gaze_df['gaze_x_px'].diff()
gaze_df['gaze_dy'] = gaze_df['gaze_y_px'].diff()
gaze_df['dt'] = gaze_df['time_s'].diff()

# Prevent division by zero or NaNs in velocity
gaze_df['gaze_velocity'] = np.sqrt(gaze_df['gaze_dx']**2 + gaze_df['gaze_dy']**2) / gaze_df['dt']
gaze_df['gaze_velocity'] = gaze_df['gaze_velocity'].replace([np.inf, -np.inf], np.nan)
gaze_df['gaze_velocity'] = gaze_df['gaze_velocity'].fillna(0)

# --- Create subplots ---
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("GAZE INFO", fontsize=18)

# 1. Gaze X and Y over time
axs[0, 0].plot(gaze_df['time_s'], gaze_df['gaze_x_px'], label='Gaze X', alpha=0.7)
axs[0, 0].plot(gaze_df['time_s'], gaze_df['gaze_y_px'], label='Gaze Y', alpha=0.7)
axs[0, 0].set_title("Gaze X and Y Over Time")
axs[0, 0].set_xlabel("Time [s]")
axs[0, 0].set_ylabel("Position [px]")
axs[0, 0].legend()

# 2. Azimuth and Elevation over time
axs[0, 1].plot(gaze_df['time_s'], gaze_df['azimuth_deg'], label='Azimuth', color='orange')
axs[0, 1].plot(gaze_df['time_s'], gaze_df['elevation_deg'], label='Elevation', color='green')
axs[0, 1].set_title("Azimuth and Elevation Over Time")
axs[0, 1].set_xlabel("Time [s]")
axs[0, 1].set_ylabel("Degrees")
axs[0, 1].legend()

# 3. Gaze heatmap
heatmap = axs[1, 0].hist2d(gaze_df['gaze_x_px'], gaze_df['gaze_y_px'], bins=100, cmap='hot')
axs[1, 0].set_title("Gaze Heatmap")
axs[1, 0].set_xlabel("Gaze X [px]")
axs[1, 0].set_ylabel("Gaze Y [px]")
axs[1, 0].invert_yaxis()
plt.colorbar(heatmap[3], ax=axs[1, 0])

# 4. Gaze velocity over time
axs[1, 1].plot(gaze_df['time_s'], gaze_df['gaze_velocity'], color='purple')
axs[1, 1].set_title("Gaze Velocity Over Time")
axs[1, 1].set_xlabel("Time [s]")
axs[1, 1].set_ylabel("Velocity [px/s]")

# --- Final layout ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ---------- FIXATION DATA PLOTS ----------

# Convert timestamps to seconds
fixations_df['start_timestamp_ns'] = fixations_df['start_timestamp_ns'] / 1e9
fixations_df['end_timestamp_ns'] = fixations_df['end_timestamp_ns'] / 1e9

# Set up figure and subplots
fig, axs = plt.subplots(3, 2, figsize=(18, 12))
fig.suptitle("FIXATIONS INFO", fontsize=18)

# 1. Fixation duration over time
axs[0, 0].plot(fixations_df['start_timestamp_ns'], fixations_df['duration_ms'], marker='o', linestyle='-', color='blue')
axs[0, 0].set_title("Fixation Duration Over Time")
axs[0, 0].set_xlabel("Start Time [s]")
axs[0, 0].set_ylabel("Duration [ms]")

# 2. Scatter plot of fixation positions
axs[0, 1].scatter(fixations_df['fixation_x_px'], fixations_df['fixation_y_px'], c=fixations_df['duration_ms'], cmap='viridis', s=30)
axs[0, 1].set_title("Fixation Positions (Color = Duration)")
axs[0, 1].set_xlabel("Fixation X [px]")
axs[0, 1].set_ylabel("Fixation Y [px]")
axs[0, 1].invert_yaxis()  # Screen coords often have Y downward

# 3. Histogram of fixation durations
axs[1, 0].hist(fixations_df['duration_ms'], bins=50, color='teal', edgecolor='black')
axs[1, 0].set_title("Histogram of Fixation Durations")
axs[1, 0].set_xlabel("Duration [ms]")
axs[1, 0].set_ylabel("Count")

# 4. Azimuth and Elevation over time
axs[1, 1].plot(fixations_df['start_timestamp_ns'], fixations_df['azimuth_deg'], label="Azimuth", color='orange')
axs[1, 1].plot(fixations_df['start_timestamp_ns'], fixations_df['elevation_deg'], label="Elevation", color='green')
axs[1, 1].set_title("Azimuth and Elevation Over Time")
axs[1, 1].set_xlabel("Time [s]")
axs[1, 1].set_ylabel("Degrees")
axs[1, 1].legend()

# 5. Fixation heatmap
axs[2, 0].hist2d(fixations_df['fixation_x_px'], fixations_df['fixation_y_px'], bins=100, cmap='hot')
axs[2, 0].set_title("Fixation Heatmap")
axs[2, 0].set_xlabel("Fixation X [px]")
axs[2, 0].set_ylabel("Fixation Y [px]")
axs[2, 0].invert_yaxis()

# Hide last empty subplot
axs[2, 1].axis('off')

# Final layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ---------- IMU DATA PLOTS ----------
imu_df['timestamp_ns'] = imu_df['timestamp_ns'] / 1e9

fig, axs = plt.subplots(3, 2, figsize=(18, 12))
fig.suptitle("IMU DATA", fontsize=18)

# 1. Gyroscope
axs[0, 0].plot(imu_df['timestamp_ns'], imu_df['gyro_x_deg_s'], label='Gyro X')
axs[0, 0].plot(imu_df['timestamp_ns'], imu_df['gyro_y_deg_s'], label='Gyro Y')
axs[0, 0].plot(imu_df['timestamp_ns'], imu_df['gyro_z_deg_s'], label='Gyro Z')
axs[0, 0].set_title("Gyroscope [deg/s]")
axs[0, 0].set_xlabel("Time [s]")
axs[0, 0].legend()

# 2. Accelerometer
axs[0, 1].plot(imu_df['timestamp_ns'], imu_df['acceleration_x_g'], label='Accel X')
axs[0, 1].plot(imu_df['timestamp_ns'], imu_df['acceleration_y_g'], label='Accel Y')
axs[0, 1].plot(imu_df['timestamp_ns'], imu_df['acceleration_z_g'], label='Accel Z')
axs[0, 1].set_title("Accelerometer [g]")
axs[0, 1].set_xlabel("Time [s]")
axs[0, 1].legend()

# 3. Orientation (roll, pitch, yaw)
axs[1, 0].plot(imu_df['timestamp_ns'], imu_df['roll_deg'], label='Roll')
axs[1, 0].plot(imu_df['timestamp_ns'], imu_df['pitch_deg'], label='Pitch')
axs[1, 0].plot(imu_df['timestamp_ns'], imu_df['yaw_deg'], label='Yaw')
axs[1, 0].set_title("Orientation [deg]")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].legend()

# 4. Quaternion W
axs[1, 1].plot(imu_df['timestamp_ns'], imu_df['quaternion_w'], label='Quat W')
axs[1, 1].set_title("Quaternion W Component")
axs[1, 1].set_xlabel("Time [s]")

# 5. Quaternion Vector Components
axs[2, 0].plot(imu_df['timestamp_ns'], imu_df['quaternion_x'], label='Quat X')
axs[2, 0].plot(imu_df['timestamp_ns'], imu_df['quaternion_y'], label='Quat Y')
axs[2, 0].plot(imu_df['timestamp_ns'], imu_df['quaternion_z'], label='Quat Z')
axs[2, 0].set_title("Quaternion Vector Components")
axs[2, 0].set_xlabel("Time [s]")
axs[2, 0].legend()

axs[2, 1].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ---------- SACCADES DATA PLOTS ----------

saccades_df['start_timestamp_ns'] = saccades_df['start_timestamp_ns'] / 1e9

fig, axs = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("SACCADES INFO", fontsize=18)

axs[0, 0].plot(saccades_df['start_timestamp_ns'], saccades_df['duration_ms'], marker='o')
axs[0, 0].set_title("Saccade Duration Over Time")
axs[0, 0].set_xlabel("Time [s]")
axs[0, 0].set_ylabel("Duration [ms]")

axs[0, 1].plot(saccades_df['start_timestamp_ns'], saccades_df['amplitude_deg'])
axs[0, 1].set_title("Saccade Amplitude Over Time")
axs[0, 1].set_xlabel("Time [s]")
axs[0, 1].set_ylabel("Amplitude [deg]")

axs[1, 0].plot(saccades_df['start_timestamp_ns'], saccades_df['mean_velocity_px_s'], label="Mean Velocity")
axs[1, 0].plot(saccades_df['start_timestamp_ns'], saccades_df['peak_velocity_px_s'], label="Peak Velocity")
axs[1, 0].set_title("Saccade Velocities")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].set_ylabel("Velocity [px/s]")
axs[1, 0].legend()

axs[1, 1].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ---------- BLINKS DATA PLOTS ----------

blinks_df['start_timestamp_ns'] = blinks_df['start_timestamp_ns'] / 1e9

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(blinks_df['start_timestamp_ns'], blinks_df['duration_ms'], linestyle='-')
ax.set_title("Blink Duration Over Time")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Duration [ms]")
plt.tight_layout()
plt.show()

# ---------- EYE STATES DATA PLOTS ----------

eye3d_df['timestamp_ns'] = eye3d_df['timestamp_ns'] / 1e9

fig, axs = plt.subplots(3, 2, figsize=(12, 12))  # 3 rows, 1 column
fig.suptitle("3D EYE STATE", fontsize=18)

# 1. Pupil diameter
axs[0, 0].plot(eye3d_df['timestamp_ns'], eye3d_df['pupil_diameter_left_mm'], label='Left')
axs[0, 0].plot(eye3d_df['timestamp_ns'], eye3d_df['pupil_diameter_right_mm'], label='Right')
axs[0, 0].set_title("Pupil Diameter [mm]")
axs[0, 0].set_xlabel("Time [s]")
axs[0, 0].legend()

# 2. Optical axis left
axs[0, 1].plot(eye3d_df['timestamp_ns'], eye3d_df['optical_axis_left_x'], label='X')
axs[0, 1].plot(eye3d_df['timestamp_ns'], eye3d_df['optical_axis_left_y'], label='Y')
axs[0, 1].plot(eye3d_df['timestamp_ns'], eye3d_df['optical_axis_left_z'], label='Z')
axs[0, 1].set_title("Optical Axis Left")
axs[0, 1].set_xlabel("Time [s]")
axs[0, 1].legend()

# 3. Optical axis right
axs[1, 0].plot(eye3d_df['timestamp_ns'], eye3d_df['optical_axis_right_x'], label='X')
axs[1, 0].plot(eye3d_df['timestamp_ns'], eye3d_df['optical_axis_right_y'], label='Y')
axs[1, 0].plot(eye3d_df['timestamp_ns'], eye3d_df['optical_axis_right_z'], label='Z')
axs[1, 0].set_title("Optical Axis Right")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].legend()

# 4. Eyelid angle left ,
axs[1, 1].plot(eye3d_df['timestamp_ns'], eye3d_df['eyelid_angle_top_left'], label='Top')
axs[1, 1].plot(eye3d_df['timestamp_ns'], eye3d_df['eyelid_angle_bottom_left'], label='Bottom')
axs[1, 1].set_title("Eyelid Angle Left")
axs[1, 1].set_xlabel("Time [s]")
axs[1, 1].legend()
    
# 5. Eyelid angle right
axs[2, 0].plot(eye3d_df['timestamp_ns'], eye3d_df['eyelid_angle_top_right'], label='Top')
axs[2, 0].plot(eye3d_df['timestamp_ns'], eye3d_df['eyelid_angle_bottom_right'], label='Bottom')
axs[2, 0].set_title("Eyelid Angle Right")
axs[2, 0].set_xlabel("Time [s]")
axs[2, 0].legend()

# 6. Eyelid aperture (mm)
axs[2, 1].plot(eye3d_df['timestamp_ns'], eye3d_df['eyelid_aperture_left_mm'], label='Left')
axs[2, 1].plot(eye3d_df['timestamp_ns'], eye3d_df['eyelid_aperture_right_mm'], label='Right')
axs[2, 1].set_title("Eyelid Aperture [mm]")
axs[2, 1].set_xlabel("Time [s]")
axs[2, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve space for suptitle
plt.show()


# --- Combined Gaze and Fixation Plot ---
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(gaze_df['gaze_x_px'], gaze_df['gaze_y_px'], label='Gaze Path', color='lightgray', alpha=0.6)
ax.scatter(fixations_df['fixation_x_px'], fixations_df['fixation_y_px'], 
           c=fixations_df['duration_ms'], 
           cmap='plasma', 
           s=fixations_df['duration_ms'] * 0.5, 
           edgecolors='black', 
           label='Fixations')

ax.set_title("Gaze Path and Fixations")
ax.set_xlabel("Gaze X [px]")
ax.set_ylabel("Gaze Y [px]")
ax.invert_yaxis()  # Typical for screen coordinates
ax.legend()
cbar = plt.colorbar(ax.collections[0], ax=ax, label='Fixation Duration [ms]')
plt.tight_layout()
plt.show()


# Print formatted output
print(f"Recording ID: {recording_id}")
print(f"Wearer: {wearer}")
print(f"Duration: {duration_sec} sec")
print(f"Gaze Freq: {gaze_freq} Hz")
print(f"Gaze samples: {len(gaze_df)}")
print(f"Fixation samples: {len(fixations_df)}")
print(f"IMU samples: {len(imu_df)}")
print(f"Saccade samples: {len(saccades_df)}")
print(f"Blink samples: {len(blinks_df)}")
print(f"3D Eye State samples: {len(eye3d_df)}")