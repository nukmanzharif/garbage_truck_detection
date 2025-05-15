# src/utils/data_utils.py
import os
import cv2
import numpy as np
import yaml
from datetime import datetime
import pandas as pd

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_video_properties(video_path):
    """Get video properties (width, height, fps, frame count)."""
    if video_path.isdigit():
        # Webcam
        cap = cv2.VideoCapture(int(video_path))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = -1  # Indefinite for webcam
    else:
        # Video file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    return {
        "width": int(width),
        "height": int(height),
        "fps": fps,
        "frame_count": frame_count
    }

def frame_to_time(frame_number, fps):
    """Convert frame number to timestamp."""
    seconds = frame_number / fps
    return seconds

def create_video_writer(output_path, width, height, fps):
    """Create a VideoWriter object."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def convert_detections_to_df(detections, frame_number, timestamp, fps):
    """Convert detection results to DataFrame."""
    records = []
    
    if hasattr(detections, 'xyxy') and len(detections.xyxy) > 0:
        if hasattr(detections, 'tracker_id'):
            if detections.tracker_id is not None:
                for i in range(len(detections.xyxy)):
                    x1, y1, x2, y2 = detections.xyxy[i]
                    confidence = detections.confidence[i] if hasattr(detections, 'confidence') and i < len(detections.confidence) else 0
                    class_id = detections.class_id[i] if hasattr(detections, 'class_id') and i < len(detections.class_id) else 0
                    
                    # Fixed: Check if tracker_id exists and has enough elements
                    tracker_id = None
                    if hasattr(detections, 'tracker_id') and detections.tracker_id is not None and i < len(detections.tracker_id):
                        tracker_id = int(detections.tracker_id[i])
                    
                    record = {
                        'frame': frame_number,
                        'timestamp': timestamp,  # Will be updated in the logger
                        'system_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),  # Keep the system time as additional info
                        'class_id': int(class_id),
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'confidence': float(confidence),
                        'tracker_id': tracker_id
                    }
                    records.append(record)
    
    return pd.DataFrame(records) if records else pd.DataFrame()

def save_detection_log(df, log_path):
    """Save detection results to CSV file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # If file exists, append without header, otherwise create new file with header
    if os.path.exists(log_path):
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, index=False)