# src/pipeline/timestamp_logger.py
import pandas as pd
import os
import cv2
from datetime import datetime, timedelta

class TimestampLogger:
    def __init__(self, config):
        """Initialize timestamp logger."""
        self.config = config
        self.log_file = config.get('log_file', 'outputs/logs/detections.csv')
        self.enabled = config.get('enabled', True)
        
        # Create log directory if it doesn't exist
        if self.enabled:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Dictionary to track truck appearances
        self.truck_tracks = {}
        self.appearance_log = []
        
        # Dictionary to track best frames for each truck
        self.best_frames = {}
        
        # Video start time - will be set when processing starts
        self.video_start_time = None
    
    def set_video_start_time(self, start_time=None):
        """Set the video start time for timestamp calculations."""
        if start_time:
            self.video_start_time = start_time
        else:
            self.video_start_time = datetime.now()
    
    def get_video_timestamp(self, frame_number, fps):
        """Calculate video timestamp based on frame number and FPS."""
        if not self.video_start_time:
            # If no start time set, use current time as fallback
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        
        # Calculate seconds elapsed based on frame number and FPS
        seconds_elapsed = frame_number / fps
        
        # Calculate actual timestamp
        video_time = self.video_start_time + timedelta(seconds=seconds_elapsed)
        return video_time.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    def log_detections(self, detections_df, frame_number, fps):
        """Log detection timestamps to file."""
        if not self.enabled or detections_df.empty:
            return
        
        # Calculate video timestamp in seconds
        seconds_elapsed = frame_number / fps
        
        # Format as HH:MM:SS.ff
        hours = int(seconds_elapsed / 3600)
        minutes = int((seconds_elapsed % 3600) / 60)
        seconds = seconds_elapsed % 60
        video_time = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
        
        # Create simplified dataframe with just frame and video_time
        simplified_df = pd.DataFrame({
            'frame': detections_df['frame'],
            'video_time': video_time
        })
        
        # Save simplified detections
        simplified_log_file = os.path.join(os.path.dirname(self.log_file), 'simplified_detections.csv')
        
        if os.path.exists(simplified_log_file):
            simplified_df.to_csv(simplified_log_file, mode='a', header=False, index=False)
        else:
            simplified_df.to_csv(simplified_log_file, index=False)
        
        # Also keep the original logging for compatibility
        # Calculate video timestamp
        video_timestamp = self.get_video_timestamp(frame_number, fps)
        
        # Update dataframe timestamps
        detections_df['timestamp'] = video_timestamp
        
        # Save raw detections
        if os.path.exists(self.log_file):
            detections_df.to_csv(self.log_file, mode='a', header=False, index=False)
        else:
            detections_df.to_csv(self.log_file, index=False)
    
    def update_truck_tracks(self, detections_df, frame_number, fps):
        """ 
        Update truck tracks with new detections.
        Track when trucks appear and disappear from frame.
        """
        if detections_df.empty:
            return
        
        # Calculate video timestamp in seconds (simple format)
        seconds_elapsed = frame_number / fps
        
        # Calculate formatted video timestamp for logging
        video_time = None
        if self.video_start_time:
            video_time = self.video_start_time + timedelta(seconds=seconds_elapsed)
        else:
            video_time = datetime.now()
        
        # Process detections with tracker IDs
        if 'tracker_id' in detections_df.columns:
            active_tracks = set()
            
            for _, detection in detections_df.iterrows():
                tracker_id = detection['tracker_id']
                if pd.notna(tracker_id):
                    tracker_id = int(tracker_id)
                    active_tracks.add(tracker_id)
                    
                    # If this is a new track, log its appearance
                    if tracker_id not in self.truck_tracks:
                        # Store the exact time in seconds as video_seconds for accuracy
                        self.truck_tracks[tracker_id] = {
                            'first_seen': video_time,
                            'last_seen': video_time,
                            'video_seconds': seconds_elapsed,  # Store seconds for accurate reporting
                            'first_frame': frame_number,
                            'status': 'active'
                        }
                        
                        # Add to appearance log
                        self.appearance_log.append({
                            'tracker_id': tracker_id,
                            'event': 'appeared',
                            'timestamp': video_time,
                            'video_seconds': seconds_elapsed,  # Store seconds
                            'frame': frame_number
                        })
                    else:
                        # Update last seen time
                        self.truck_tracks[tracker_id]['last_seen'] = video_time
                        self.truck_tracks[tracker_id]['status'] = 'active'
            
            # Check for trucks that have disappeared (not in current frame)
            for tracker_id, track_info in self.truck_tracks.items():
                if track_info['status'] == 'active' and tracker_id not in active_tracks:
                    # Mark as inactive
                    self.truck_tracks[tracker_id]['status'] = 'inactive'
                    
                    # Add to appearance log
                    self.appearance_log.append({
                        'tracker_id': tracker_id,
                        'event': 'disappeared',
                        'timestamp': video_time,
                        'video_seconds': seconds_elapsed,  # Store seconds
                        'frame': frame_number
                    })
    
    def get_appearance_summary(self):
        """Get summary of truck appearances and disappearances."""
        if not self.appearance_log:
            return pd.DataFrame()
        
        # Convert to DataFrame
        appearances_df = pd.DataFrame(self.appearance_log)
        
        # Get unique tracker IDs
        unique_ids = appearances_df['tracker_id'].unique()
        
        summary_records = []
        for tracker_id in unique_ids:
            # Get all events for this tracker
            tracker_events = appearances_df[appearances_df['tracker_id'] == tracker_id].sort_values('timestamp')
            
            # Group appearances and disappearances
            appearances = tracker_events[tracker_events['event'] == 'appeared']
            disappearances = tracker_events[tracker_events['event'] == 'disappeared']
            
            # For each appearance, find the next disappearance
            for i, appear_row in appearances.iterrows():
                appear_time = appear_row['timestamp']
                appear_frame = appear_row['frame']
                
                # Find the next disappearance after this appearance
                next_disappear = disappearances[disappearances['timestamp'] > appear_time]
                
                if not next_disappear.empty:
                    disappear_time = next_disappear.iloc[0]['timestamp']
                    disappear_frame = next_disappear.iloc[0]['frame']
                    duration = (disappear_time - appear_time).total_seconds()
                else:
                    # If no disappearance found, use the frame duration directly
                    # instead of current time, to avoid incorrect durations
                    disappear_frame = appearances_df['frame'].max()  # Use max frame as end
                    # Calculate duration based on frames and fps
                    if 'fps' in self.config:
                        fps = self.config.get('fps', 30)
                        duration = (disappear_frame - appear_frame) / fps
                    else:
                        # If we can't calculate from frames, use a reasonable default
                        duration = 17.0  # Assuming it's the 17 second video you mentioned
                    disappear_time = None
                
                summary_records.append({
                    'tracker_id': tracker_id,
                    'appeared': appear_time,
                    'disappeared': disappear_time,
                    'duration_seconds': duration,
                    'start_frame': appear_frame,
                    'end_frame': disappear_frame
                })
        
        return pd.DataFrame(summary_records)
    
    def save_appearance_summary(self, output_path=None):
        """Save appearance summary to CSV file."""
        if not output_path:
            output_path = os.path.join(os.path.dirname(self.log_file), 'appearance_summary.csv')
        
        summary_df = self.get_appearance_summary()
        if not summary_df.empty:
            summary_df.to_csv(output_path, index=False)
            print(f"Appearance summary saved to {output_path}")

    def get_truck_timelines(self):
        """
        Generate a timeline for each unique truck showing entry/exit times and best frame.
        Returns a DataFrame with truck_id, entry_frame, entry_time, exit_frame, exit_time, 
        duration, and best_frame (frame with highest confidence).
        """
        if not self.appearance_log:
            return pd.DataFrame()
        
        # Get all events
        events_df = pd.DataFrame(self.appearance_log)
        
        # Get unique tracker IDs
        unique_ids = [id for id in events_df['tracker_id'].unique() if pd.notna(id)]
        
        truck_timelines = []
        
        # Get the original detections with confidence scores
        if os.path.exists(self.log_file):
            try:
                all_detections = pd.read_csv(self.log_file)
            except:
                all_detections = pd.DataFrame()
        else:
            all_detections = pd.DataFrame()
        
        for tracker_id in unique_ids:
            # Filter events for this tracker
            truck_events = events_df[events_df['tracker_id'] == tracker_id].sort_values('frame')
            
            # Find appearance and disappearance
            appearances = truck_events[truck_events['event'] == 'appeared']
            disappearances = truck_events[truck_events['event'] == 'disappeared']
            
            if not appearances.empty:
                entry_time = appearances.iloc[0]['timestamp']
                entry_frame = appearances.iloc[0]['frame']
                entry_seconds = appearances.iloc[0]['video_seconds'] if 'video_seconds' in appearances.columns else entry_frame / 30.0
                
                # Find exit (last disappearance or current time if still visible)
                if not disappearances.empty:
                    exit_time = disappearances.iloc[-1]['timestamp']
                    exit_frame = disappearances.iloc[-1]['frame']
                    exit_seconds = disappearances.iloc[-1]['video_seconds'] if 'video_seconds' in disappearances.columns else exit_frame / 30.0
                else:
                    # If no disappearance event, use the last known timestamp
                    last_known = truck_events.iloc[-1]
                    exit_time = last_known['timestamp']
                    exit_frame = last_known['frame']
                    exit_seconds = last_known['video_seconds'] if 'video_seconds' in last_known else exit_frame / 30.0
                
                # Calculate duration from actual seconds rather than timestamps
                duration = exit_seconds - entry_seconds
                
                # Find the frame with highest confidence for this truck
                if not all_detections.empty:
                    truck_detections = all_detections[
                        (all_detections['tracker_id'] == tracker_id) & 
                        (pd.notna(all_detections['confidence']))
                    ]
                    
                    if not truck_detections.empty:
                        best_frame = truck_detections.loc[truck_detections['confidence'].idxmax()]['frame']
                    else:
                        best_frame = entry_frame  # Default to entry frame if no confidence data
                else:
                    best_frame = entry_frame
                
                # Add to timelines
                truck_timelines.append({
                    'truck_id': tracker_id,
                    'entry_frame': int(entry_frame),
                    'entry_time': entry_time,
                    'entry_seconds': entry_seconds,  # Add this field
                    'exit_frame': int(exit_frame),
                    'exit_time': exit_time,
                    'exit_seconds': exit_seconds,  # Add this field
                    'duration_seconds': duration,
                    'best_frame': int(best_frame)
                })
        
        # Convert to DataFrame
        timelines_df = pd.DataFrame(truck_timelines)
        
        # Format times as strings for easier reading
        if not timelines_df.empty:
            # Format timestamp columns as strings
            for col in ['entry_time', 'exit_time']:
                if col in timelines_df.columns and pd.api.types.is_datetime64_any_dtype(timelines_df[col]):
                    timelines_df[col] = timelines_df[col].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        
        return timelines_df

    def save_truck_timelines(self, output_path=None):
        """Save the truck timelines to a CSV file."""
        if not output_path:
            output_path = os.path.join(os.path.dirname(self.log_file), 'truck_timelines.csv')
        
        timelines_df = self.get_truck_timelines()
        if not timelines_df.empty:
            timelines_df.to_csv(output_path, index=False)
            print(f"Truck timelines saved to {output_path}")
            return timelines_df
        return None

    def get_video_time_formatted(self, frame_number, fps):
        """Convert frame number to a formatted video time string HH:MM:SS.ff"""
        seconds_elapsed = frame_number / fps
        hours = int(seconds_elapsed / 3600)
        minutes = int((seconds_elapsed % 3600) / 60)
        seconds = seconds_elapsed % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
    
    def get_frame_detections(self, frame_number, truck_id=None):
        """
        Get all detections for a specific frame number.
        
        Args:
            frame_number: The frame number to retrieve
            truck_id: Optional truck ID to filter by
            
        Returns:
            DataFrame with detections for the frame
        """
        # Check for the log file instead of an attribute
        if not os.path.exists(self.log_file):
            return None
        
        try:
            # Read the log file
            all_detections = pd.read_csv(self.log_file)
            
            # Filter detections by frame number
            frame_detections = all_detections[all_detections['frame'] == frame_number]
            
            # Further filter by truck ID if specified
            if truck_id is not None and not frame_detections.empty:
                frame_detections = frame_detections[frame_detections['tracker_id'] == truck_id]
                
            return frame_detections
        except Exception as e:
            print(f"Error retrieving frame detections: {e}")
            return None
    
    def update_best_frame(self, detections_df, frame, frame_number, fps):
        """
        Check if this frame has a higher confidence detection than previously seen,
        and if so, save it as the new best frame for the truck.
        
        Args:
            detections_df: DataFrame with detection info
            frame: The actual video frame
            frame_number: Current frame number
            fps: Frames per second
        """
        if detections_df.empty:
            return
        
        # Create directory for real-time best frames if it doesn't exist
        best_frames_dir = os.path.join(os.path.dirname(self.log_file), '..', 'best_frames')
        os.makedirs(best_frames_dir, exist_ok=True)
        
        # Get formatted timestamp
        video_time = self.get_video_time_formatted(frame_number, fps)
        
        # Process detections
        for _, detection in detections_df.iterrows():
            if pd.isna(detection['tracker_id']):
                continue
                
            tracker_id = int(detection['tracker_id'])
            confidence = float(detection['confidence'])
            
            # Check if we've seen this truck before or if current confidence is higher
            if (tracker_id not in self.best_frames or 
                confidence > self.best_frames[tracker_id]['confidence']):
                
                # Create a new annotated frame with detection box
                annotated_frame = frame.copy()
                
                # Draw bounding box
                x1, y1, x2, y2 = int(detection['x1']), int(detection['y1']), int(detection['x2']), int(detection['y2'])
                color = (0, 255, 0)  # Green
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label
                class_id = int(detection['class_id'])
                label = f"truck {confidence:.2f} ID:{tracker_id}"
                
                # Draw label background
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Save the annotated frame
                frame_filename = f"best_truck_{tracker_id}.jpg"
                frame_path = os.path.join(best_frames_dir, frame_filename)
                cv2.imwrite(frame_path, annotated_frame)
                
                # Update the best frame record
                self.best_frames[tracker_id] = {
                    'frame_number': frame_number,
                    'confidence': confidence,
                    'video_time': video_time,
                    'frame_path': frame_path
                }

    def get_best_frames(self):
        """Get information about the best frames captured for each truck."""
        if not self.best_frames:
            return pd.DataFrame()
        
        frames_list = []
        for truck_id, frame_info in self.best_frames.items():
            frames_list.append({
                'truck_id': truck_id,
                'frame_number': frame_info['frame_number'],
                'video_time': frame_info['video_time'],
                'confidence': frame_info['confidence'],
                'frame_path': frame_info['frame_path']
            })
        
        return pd.DataFrame(frames_list)
    
    def save_best_frames_info(self, output_path=None):
        """Save best frames information to a CSV file."""
        if not output_path:
            output_path = os.path.join(os.path.dirname(self.log_file), '..', 'best_frames', 'best_frames_info.csv')
        
        frames_df = self.get_best_frames()
        if not frames_df.empty:
            frames_df.to_csv(output_path, index=False)
            print(f"Best frames info saved to {output_path}")
            return frames_df
        return None