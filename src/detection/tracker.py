# src/detection/tracker.py
import supervision as sv
import inspect
import numpy as np

class ObjectTracker:
    def __init__(self, config):
        """Initialize object tracker based on configuration."""
        self.config = config
        self.tracker_type = config.get('tracker', 'bytetrack')
        self.persist_time = config.get('persist_time', 30)
        self.next_id = 1  # For fallback ID assignment
        self.working_method = None  # Store which update method works
        
        # Initialize tracker
        self.tracker = self._create_tracker()
        
    def _create_tracker(self):
        """Create and return a tracker based on configuration using available parameters."""
        # Check what parameters ByteTrack accepts in current version
        if self.tracker_type.lower() == 'bytetrack':
            # Get the initialization parameters that ByteTrack accepts
            sig = inspect.signature(sv.ByteTrack.__init__)
            params = {}
            
            # Only add parameters that are actually accepted by ByteTrack
            if 'track_thresh' in sig.parameters:
                params['track_thresh'] = 0.25
            if 'detection_threshold' in sig.parameters:
                params['detection_threshold'] = 0.25
            if 'tracker_threshold' in sig.parameters:
                params['tracker_threshold'] = 0.25
            if 'track_buffer' in sig.parameters:
                params['track_buffer'] = self.persist_time
            if 'match_thresh' in sig.parameters:
                params['match_thresh'] = 0.8
            if 'frame_rate' in sig.parameters:
                params['frame_rate'] = 30
            if 'initialization_delay' in sig.parameters:
                params['initialization_delay'] = self.persist_time
            if 'hit_counter_max' in sig.parameters:
                params['hit_counter_max'] = self.persist_time
                
            print(f"Creating ByteTrack with parameters: {params}")
            return sv.ByteTrack(**params)
        else:
            print(f"Warning: Tracker type {self.tracker_type} not recognized, using ByteTrack with default parameters")
            # Create with no parameters as a fallback
            return sv.ByteTrack()
    
    def update(self, detections, frame):
        """Update tracker with new detections."""
        if not hasattr(detections, 'xyxy') or len(detections.xyxy) == 0:
            return detections
        
        # If we already know which method works, use it directly
        if self.working_method == "update":
            return self.tracker.update(detections=detections)
        elif self.working_method == "update_with_detections":
            return self.tracker.update_with_detections(detections)
        elif self.working_method == "fallback":
            return self._apply_synthetic_ids(detections)
        
        # Otherwise try to find which method works
        try:
            # Try the current supervision API format
            updated_detections = self.tracker.update(detections=detections)
            # Check if tracker IDs were assigned
            if hasattr(updated_detections, 'tracker_id') and updated_detections.tracker_id is not None:
                self.working_method = "update"  # Remember what worked
                return updated_detections
            else:
                # No tracker IDs assigned, try next method
                raise AttributeError("No tracker IDs assigned by tracker.update()")
                
        except (TypeError, AttributeError) as e:
            try:
                # Try older supervision API format
                updated_detections = self.tracker.update_with_detections(detections)
                if hasattr(updated_detections, 'tracker_id') and updated_detections.tracker_id is not None:
                    self.working_method = "update_with_detections"  # Remember what worked
                    return updated_detections
                else:
                    raise AttributeError("No tracker IDs assigned by update_with_detections")
                    
            except Exception as e2:
                # Use fallback with synthetic IDs
                self.working_method = "fallback"  # Remember what worked
                return self._apply_synthetic_ids(detections)
    
    def _apply_synthetic_ids(self, detections):
        """Apply synthetic tracker IDs to detections as a fallback method."""
        if not hasattr(detections, 'xyxy') or len(detections.xyxy) == 0:
            return detections
            
        # Create synthetic IDs
        ids = np.array([self.next_id + i for i in range(len(detections.xyxy))])
        self.next_id += len(detections.xyxy)
        
        # Try different methods to set the tracker_id attribute
        try:
            # Method 1: Direct attribute assignment
            setattr(detections, 'tracker_id', ids)
        except Exception as e1:
            print(f"Failed to set tracker_id directly: {e1}")
            try:
                # Method 2: Using indexing if detections supports it
                detections_with_ids = detections.copy()
                detections_with_ids.tracker_id = ids
                return detections_with_ids
            except Exception as e2:
                print(f"Failed to create copy with tracker_id: {e2}")
                # Method 3: Create a new detections object with same data plus IDs
                try:
                    # Create a new detections object with the same attributes
                    new_detections = sv.Detections(
                        xyxy=detections.xyxy,
                        confidence=detections.confidence if hasattr(detections, 'confidence') else np.ones(len(detections.xyxy)),
                        class_id=detections.class_id if hasattr(detections, 'class_id') else np.zeros(len(detections.xyxy), dtype=int),
                        tracker_id=ids
                    )
                    return new_detections
                except Exception as e3:
                    print(f"Failed to create new detections: {e3}")
                    
        return detections