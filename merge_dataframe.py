import os
import pandas as pd

DATA_DIR = 'data/demo/impostor_1'

COLUMNS_TO_KEEP = {
    "3d_eye_states.csv": [
        'timestamp [ns]',
        'pupil diameter right [mm]',
        'eyeball center left x [mm]',
        'eyeball center left y [mm]',
        'eyeball center left z [mm]',
        'eyeball center right x [mm]',
        'eyeball center right y [mm]',
        'eyeball center right z [mm]',
        'optical axis left x',
        'optical axis left y',
        'optical axis left z',
        'optical axis right x',
        'optical axis right y',
        'optical axis right z',
        'eyelid angle top left [rad]',
        'eyelid angle bottom left [rad]',
        'eyelid aperture left [mm]',
        'eyelid angle top right [rad]',
        'eyelid angle bottom right [rad]',
        'eyelid aperture right [mm]'
    ],
    "blinks.csv": [
        'blink id','start timestamp [ns]','end timestamp [ns]','duration [ms]'
        ],
    "fixations.csv": [
        'fixation id',
        'start timestamp [ns]',
        'end timestamp [ns]',
        'duration [ms]',
        'fixation x [px]',
        'fixation y [px]',
        'azimuth [deg]',
        'elevation [deg]'
    ],
    "gaze.csv": [
        'timestamp [ns]',
        'gaze x [px]',
        'gaze y [px]',
        'fixation id',
        'blink id',
        'azimuth [deg]',
        'elevation [deg]'
    ],
    "imu.csv": [
        'timestamp [ns]',
        'gyro x [deg/s]',
        'gyro y [deg/s]',
        'gyro z [deg/s]',
        'acceleration x [g]',
        'acceleration y [g]',
        'acceleration z [g]',
        'roll [deg]',
        'pitch [deg]',
        'yaw [deg]',
        'quaternion w',
        'quaternion x',
        'quaternion y',
        'quaternion z'
    ],
    "saccades.csv": [
        'saccade id',
        'start timestamp [ns]',
        'end timestamp [ns]',
        'duration [ms]',
        'amplitude [px]',
       ' amplitude [deg]',
        'mean velocity [px/s]',
        'peak velocity [px/s]'
    ]
}
