# src/detection/model.py
import torch
from ultralytics import YOLO
import numpy as np
import supervision as sv
import os

class DetectionModel:
    def __init__(self, config):
        """Initialize detection model based on configuration."""
        self.config = config
        self.model_type = config['model']['type']
        self.weights = config['model']['weights']
        self.conf_threshold = config['model']['confidence_threshold']
        self.iou_threshold = config['model']['iou_threshold']
        self.device = config['model']['device']
        
        # Target classes to detect
        self.target_classes = config['classes']['target_classes']
        self.class_mapping = config['classes'].get('custom_mapping', {})
        
        # Load model
        self.model = self._load_model()
        
        # Remove the problematic line
        # self.detection_format = sv.DetectionFormat.YOLO
    
    def _load_model(self):
        """Load YOLOv8 model."""
        # Data folder path
        data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'models')
        os.makedirs(data_folder, exist_ok=True)
        
        model_path = self.weights
        if not os.path.exists(model_path):
            # If not a path, assume it's a pretrained model name from Ultralytics
            print(f"Downloading model {self.weights}...")
            model = YOLO(self.weights)
            if hasattr(model, 'export'):
                exported_model_path = os.path.join(data_folder, f"{self.weights.split('.')[0]}_exported.pt")
                model.export(format="torchscript", optimize=True)
                if os.path.exists(exported_model_path):
                    model_path = exported_model_path
        else:
            model = YOLO(model_path)
        
        print(f"Model loaded from {model_path}")
        return model
    
    def detect(self, frame):
        """Perform object detection on a frame."""
        # Run inference
        results = self.model(
            frame, 
            conf=self.conf_threshold, 
            iou=self.iou_threshold,  # Changed from iou_threshold to iou
            device=self.device,
            verbose=False
        )
        
        # Convert results to supervision Detections format
        result = results[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter by target classes
        if self.target_classes and hasattr(detections, 'class_id') and len(detections.class_id) > 0:
            # Create boolean mask
            mask = np.isin(detections.class_id, self.target_classes)
            
            # Only apply filtering if we have any detections
            if len(mask) > 0 and np.any(mask):
                detections = detections[mask]
            else:
                # Create an empty detections object with required parameters
                detections = sv.Detections(
                    xyxy=np.empty((0, 4), dtype=np.float32),
                    confidence=np.array([], dtype=np.float32),
                    class_id=np.array([], dtype=int)
                )
        
        return detections
    
    def get_class_names(self):
        """Get class names from model."""
        if hasattr(self.model, 'names'):
            classes = self.model.names
            # Apply custom mapping if available
            for idx, name in self.class_mapping.items():
                if isinstance(idx, str):
                    idx = int(idx)
                if idx < len(classes):
                    classes[idx] = name
            return classes
        return None