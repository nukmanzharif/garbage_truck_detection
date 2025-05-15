# src/utils/visualization.py
import cv2
import numpy as np
from datetime import datetime

def draw_detections(frame, detections, class_names=None):
    """Draw bounding boxes and labels on frame."""
    if not hasattr(detections, 'xyxy') or len(detections.xyxy) == 0:
        return frame
    
    # Make a copy of the frame to avoid modifying the original
    annotated_frame = frame.copy()
    
    for i in range(len(detections.xyxy)):
        # Get detection coordinates
        x1, y1, x2, y2 = map(int, detections.xyxy[i])
        
        # Get class id and confidence
        class_id = int(detections.class_id[i]) if hasattr(detections, 'class_id') and i < len(detections.class_id) else 0
        confidence = detections.confidence[i] if hasattr(detections, 'confidence') and i < len(detections.confidence) else 0
        
        # Get tracker id if available - fixed to properly handle None
        tracker_id = None
        if hasattr(detections, 'tracker_id'):
            # Check if tracker_id exists and is not None
            if detections.tracker_id is not None and len(detections.tracker_id) > i:
                tracker_id = detections.tracker_id[i]
        
        # Define color (BGR)
        color = (0, 255, 0)  # Green for garbage trucks
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        if class_names and class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"Class {class_id}"
            
        label = f"{class_name} {confidence:.2f}"
        if tracker_id is not None:
            label += f" ID:{tracker_id}"
        
        # Draw label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return annotated_frame

def draw_timestamp(frame, timestamp=None, position='bottom-right'):
    """Draw timestamp on frame."""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    # Make a copy of the frame to avoid modifying the original
    annotated_frame = frame.copy()
    
    text_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    
    if position == 'top-left':
        text_position = (10, 30)
    elif position == 'top-right':
        text_position = (annotated_frame.shape[1] - text_size[0] - 10, 30)
    elif position == 'bottom-left':
        text_position = (10, annotated_frame.shape[0] - 10)
    else:  # bottom-right
        text_position = (annotated_frame.shape[1] - text_size[0] - 10, annotated_frame.shape[0] - 10)
    
    # Draw text background
    cv2.rectangle(
        annotated_frame, 
        (text_position[0] - 5, text_position[1] - text_size[1] - 5),
        (text_position[0] + text_size[0] + 5, text_position[1] + 5),
        (0, 0, 0), -1
    )
    
    # Draw text
    cv2.putText(
        annotated_frame, timestamp, text_position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    
    return annotated_frame

def draw_roi(frame, roi):
    """Draw region of interest on frame."""
    if not roi or len(roi) != 4:
        return frame
    
    # Make a copy of the frame to avoid modifying the original
    annotated_frame = frame.copy()
    
    # Convert normalized coordinates to absolute coordinates
    h, w = annotated_frame.shape[:2]
    x1, y1, x2, y2 = roi
    x1, x2 = int(x1 * w), int(x2 * w)
    y1, y2 = int(y1 * h), int(y2 * h)
    
    # Draw ROI rectangle
    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    return annotated_frame