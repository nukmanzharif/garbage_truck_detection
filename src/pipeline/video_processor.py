# src/pipeline/video_processor.py
import cv2
import time
import os
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.visualization import draw_detections, draw_timestamp, draw_roi
from src.utils.data_utils import frame_to_time, create_video_writer, convert_detections_to_df, save_detection_log

class VideoProcessor:
    def __init__(self, detector, tracker, timestamp_logger, config):
        """Initialize video processor with detector, tracker and config."""
        self.detector = detector
        self.tracker = tracker
        self.timestamp_logger = timestamp_logger
        self.config = config
        
        # Extract configuration
        self.input_source = config['input']['source']
        self.skip_frames = config['processing'].get('skip_frames', 1)
        
        # ROI configuration
        self.roi_enabled = config['processing']['roi']['enabled']
        self.roi_coords = config['processing']['roi']['coordinates'] if self.roi_enabled else None
        
        # Output configuration
        self.save_video = config['output']['save_video']
        self.output_path = config['output']['output_path']
        self.display = config['output']['display']
        self.draw_tracks = config['output']['draw_tracks']
        self.draw_timestamps = config['output']['draw_timestamps']
        
        # Initialize video capture
        self.cap = None
        self.video_writer = None
        self.frame_count = 0
        self.fps = 0
        self.class_names = self.detector.get_class_names()
    
    def _initialize_video_capture(self):
        """Initialize video capture from source."""
        # Convert string to integer for webcam
        if isinstance(self.input_source, str) and self.input_source.isdigit():
            self.input_source = int(self.input_source)
        
        self.cap = cv2.VideoCapture(self.input_source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source {self.input_source}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30  # Default to 30 FPS if not available
        
        # Override with config values if specified
        if self.config['input'].get('width') and self.config['input'].get('height'):
            self.width = self.config['input']['width']
            self.height = self.config['input']['height']
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if self.config['input'].get('fps'):
            self.fps = self.config['input']['fps']
        
        print(f"Video source initialized: {self.width}x{self.height} @ {self.fps} FPS")
        
        # Initialize video writer if saving is enabled
        if self.save_video:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (self.width, self.height)
            )
            
        if self.timestamp_logger:
            self.timestamp_logger.set_video_start_time()
    
    def _apply_roi(self, detections):
        """Apply region of interest filtering to detections."""
        if not self.roi_enabled or self.roi_coords is None or not hasattr(detections, 'xyxy') or len(detections.xyxy) == 0:
            return detections
        
        # Parse ROI coordinates
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_coords
        
        # Convert to pixel coordinates if normalized
        if all(0 <= coord <= 1 for coord in self.roi_coords):
            roi_x1 *= self.width
            roi_x2 *= self.width
            roi_y1 *= self.height
            roi_y2 *= self.height
        
        # Filter detections that intersect with ROI
        valid_indices = []
        for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
            # Check if the detection intersects with ROI
            if not (x2 < roi_x1 or x1 > roi_x2 or y2 < roi_y1 or y1 > roi_y2):
                valid_indices.append(i)
        
        # Return filtered detections
        if valid_indices:
            return detections[valid_indices]
        else:
            # Return empty detections of same structure
            empty_detections = type(detections)()
            for attr_name in dir(detections):
                if not attr_name.startswith('_') and not callable(getattr(detections, attr_name)):
                    attr_value = getattr(detections, attr_name)
                    if isinstance(attr_value, np.ndarray) and len(attr_value) > 0:
                        setattr(empty_detections, attr_name, np.array([]))
            return empty_detections
    
    def process(self):
        """Process video and detect garbage trucks."""
        self._initialize_video_capture()
        
        try:
            while self.cap.isOpened():
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video stream")
                    break
                
                # Process only every n-th frame
                if self.frame_count % self.skip_frames != 0:
                    self.frame_count += 1
                    continue
                
                # Calculate video timestamp based on frame number
                video_timestamp = self.frame_count / self.fps
                time_str = f"Frame: {self.frame_count} | Video Time: {video_timestamp:.2f}s"
                
                # Detect objects
                detections = self.detector.detect(frame)
                
                # Apply ROI filtering
                if self.roi_enabled:
                    detections = self._apply_roi(detections)
                
                # Update tracker
                if self.tracker and self.config['tracking']['enabled']:
                    detections = self.tracker.update(detections, frame)
                
                # Convert detections to DataFrame for logging
                detections_df = convert_detections_to_df(detections, self.frame_count, "", self.fps)
                
                # Log detections with proper video timestamp
                if not detections_df.empty:
                    self.timestamp_logger.log_detections(detections_df, self.frame_count, self.fps)
                    self.timestamp_logger.update_truck_tracks(detections_df, self.frame_count, self.fps)
                    
                    # Save best frame in real-time if it has higher confidence
                    self.timestamp_logger.update_best_frame(detections_df, frame, self.frame_count, self.fps)
                
                # Visualize results
                if self.display or self.save_video:
                    # Calculate a readable timestamp for display
                    hours = int(video_timestamp / 3600)
                    minutes = int((video_timestamp % 3600) / 60)
                    seconds = video_timestamp % 60
                    display_time = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
                    display_text = f"Frame: {self.frame_count} | Time: {display_time}"
                    
                    # Draw detections
                    annotated_frame = draw_detections(frame, detections, self.class_names)
                    
                    # Draw ROI if enabled
                    if self.roi_enabled:
                        annotated_frame = draw_roi(annotated_frame, self.roi_coords)
                    
                    # Draw timestamp
                    if self.draw_timestamps:
                        annotated_frame = draw_timestamp(annotated_frame, display_text)
                    
                    # Write frame to output video
                    if self.save_video and self.video_writer:
                        self.video_writer.write(annotated_frame)
                    
                    # Display frame
                    if self.display:
                        cv2.imshow('Garbage Truck Detection', annotated_frame)
                        
                        # Break loop if 'q' is pressed
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                
                self.frame_count += 1
                
                # Print progress for every 100 frames
                if self.frame_count % 100 == 0:
                    print(f"Processed {self.frame_count} frames")
        
        except KeyboardInterrupt:
            print("Processing interrupted by user")
        
        finally:
            # Release resources
            if self.cap:
                self.cap.release()
            
            if self.video_writer:
                self.video_writer.release()
            
            cv2.destroyAllWindows()
            
            # Save appearance summary
            if self.timestamp_logger:
                # Save appearance summary
                summary_path = os.path.join(os.path.dirname(self.output_path), 'garbage_truck_summary.csv')
                if hasattr(self.timestamp_logger, 'save_appearance_summary'):
                    self.timestamp_logger.save_appearance_summary(summary_path)
                else:
                    # Use truck timelines as appearance summary
                    timelines = self.timestamp_logger.get_truck_timelines()
                    if not timelines.empty:
                        timelines.to_csv(summary_path, index=False)
                        print(f"Truck timelines saved as appearance summary to {summary_path}")
                
                # Save truck timelines
                timelines_path = os.path.join(os.path.dirname(self.output_path), 'truck_timelines.csv')
                self.timestamp_logger.save_truck_timelines(timelines_path)
                
                # Save best frames info - ADD THIS HERE
                best_frames_path = os.path.join(os.path.dirname(self.output_path), 'best_frames_info.csv')
                self.timestamp_logger.save_best_frames_info(best_frames_path)
                
                # Extract truck frames - USE BEST FRAMES INSTEAD FOR LLM
                best_frames = self.timestamp_logger.get_best_frames()
                if not best_frames.empty:
                    # Use this for classification instead of extract_truck_frames
                    frames_csv = os.path.join(os.path.dirname(self.output_path), 'extracted_frames.csv')
                    best_frames.to_csv(frames_csv, index=False)
                    print(f"Best frames for {len(best_frames)} trucks saved for classification")
                else:
                    # Only call extract_truck_frames as fallback if no best frames
                    extracted_frames = self.extract_truck_frames()
                
                # Print summary
                if hasattr(self.timestamp_logger, 'get_appearance_summary'):
                    summary = self.timestamp_logger.get_appearance_summary()
                else:
                    # Use truck timelines as appearance summary
                    summary = self.timestamp_logger.get_truck_timelines()
                if not summary.empty:
                    print("\nGarbage Truck Appearance Summary:")
                    print(summary)
                    
                    # Print total duration
                    total_duration = summary['duration_seconds'].sum()
                    print(f"\nTotal time garbage trucks were detected: {total_duration:.2f} seconds")
            
            print(f"\nProcessing completed. {self.frame_count} frames processed.")
            print(f"\nProcessing completed. {self.frame_count} frames processed.")


    def extract_truck_frames(self):
        """
        Extract frames for each unique truck and save them to disk.
        Extracts the best frame (highest confidence) for each unique truck ID.
        Saves both original and annotated frames with detection boxes.
        """
        if not self.timestamp_logger:
            print("Timestamp logger not initialized, cannot extract truck frames")
            return
        
        # Get truck timelines
        truck_timelines = self.timestamp_logger.get_truck_timelines()
        if truck_timelines.empty:
            print("No truck timelines available, cannot extract frames")
            return
        
        # Create directory for truck images if it doesn't exist
        images_dir = os.path.join(os.path.dirname(self.output_path), 'images', 'trucks')
        os.makedirs(images_dir, exist_ok=True)
        
        # Create directory for annotated images
        annotated_dir = os.path.join(os.path.dirname(self.output_path), 'images', 'annotated_trucks')
        os.makedirs(annotated_dir, exist_ok=True)
        
        # Reset video capture to start
        self.cap = cv2.VideoCapture(self.input_source)
        if not self.cap.isOpened():
            print(f"Could not open video source {self.input_source} for frame extraction")
            return
        
        # Extract each best frame
        extracted_frames = []
        for _, truck in truck_timelines.iterrows():
            truck_id = truck['truck_id']
            best_frame = int(truck['best_frame'])
            
            # Set frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
            ret, frame = self.cap.read()
            
            if ret:
                # Get formatted timestamp
                video_time = self.timestamp_logger.get_video_time_formatted(best_frame, self.fps)
                
                # Save the original frame
                frame_filename = f"truck_{truck_id}_frame_{best_frame}_{video_time.replace(':', '-')}.jpg"
                frame_path = os.path.join(images_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                
                # Create the annotated frame with detection box
                annotated_filename = f"annotated_truck_{truck_id}_frame_{best_frame}_{video_time.replace(':', '-')}.jpg"
                annotated_path = os.path.join(annotated_dir, annotated_filename)
                
                # Get detection data for this frame from the logger
                truck_detections = self.timestamp_logger.get_frame_detections(best_frame, truck_id)
                
                if truck_detections is not None and not truck_detections.empty:
                    # Create a synthetic detection object
                    detection = type('obj', (object,), {
                        'xyxy': np.array([[truck_detections['x1'].values[0], 
                                        truck_detections['y1'].values[0],
                                        truck_detections['x2'].values[0], 
                                        truck_detections['y2'].values[0]]]),
                        'confidence': np.array([truck_detections['confidence'].values[0]]),
                        'class_id': np.array([truck_detections['class_id'].values[0]]),
                        'tracker_id': np.array([truck_id])
                    })
                    
                    # Draw the detection on a copy of the frame
                    annotated_frame = draw_detections(frame, detection, self.class_names)
                    cv2.imwrite(annotated_path, annotated_frame)
                    
                    extracted_frames.append({
                        'truck_id': truck_id,
                        'frame': best_frame,
                        'video_time': video_time,
                        'image_path': frame_path,
                        'annotated_path': annotated_path
                    })
                    
                    print(f"Extracted frame for truck {truck_id} at {video_time} (frame {best_frame})")
        
        # Close the capture
        self.cap.release()
        
        # Save the extracted frames info
        if extracted_frames:
            frames_df = pd.DataFrame(extracted_frames)
            frames_csv = os.path.join(os.path.dirname(self.output_path), 'extracted_frames.csv')
            frames_df.to_csv(frames_csv, index=False)
            print(f"Extracted {len(extracted_frames)} truck frames. Info saved to {frames_csv}")
        
        return extracted_frames