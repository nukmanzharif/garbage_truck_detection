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
        video_time = self.get_video_time_formatted(frame_number, fps)
        
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
        detections_df['video_time'] = video_time
        
        # Save raw detections
        if os.path.exists(self.log_file):
            detections_df.to_csv(self.log_file, mode='a', header=False, index=False)
        else:
            detections_df.to_csv(self.log_file, index=False)
    
    def update_truck_tracks(self, detections_df, frame_number, fps):
        """ 
        Update truck tracks with new detections.
        Track only first entrance and last exit for each truck.
        """
        if detections_df.empty:
            return
        
        # Calculate video timestamp in seconds (simple format)
        seconds_elapsed = frame_number / fps
        
        # Get formatted video time (HH:MM:SS)
        video_time = self.get_video_time_formatted(frame_number, fps)
        
        # Process detections with tracker IDs
        if 'tracker_id' in detections_df.columns:
            active_tracks = set()
            
            for _, detection in detections_df.iterrows():
                tracker_id = detection['tracker_id']
                if pd.notna(tracker_id):
                    tracker_id = int(tracker_id)
                    active_tracks.add(tracker_id)
                    
                    # If this is a new track, log its first appearance
                    if tracker_id not in self.truck_tracks:
                        self.truck_tracks[tracker_id] = {
                            'first_seen_frame': frame_number,
                            'first_seen_time': video_time,
                            'last_seen_frame': frame_number,
                            'last_seen_time': video_time,
                            'status': 'active'
                        }
                    else:
                        # Just update the last seen time and frame
                        self.truck_tracks[tracker_id]['last_seen_frame'] = frame_number
                        self.truck_tracks[tracker_id]['last_seen_time'] = video_time
                        self.truck_tracks[tracker_id]['status'] = 'active'
            
            # No need to track disappearances for the simplified approach
    
    def get_truck_timelines(self):
        """
        Generate a simplified timeline for each unique truck showing only first entry and last exit times.
        """
        if not self.truck_tracks:
            return pd.DataFrame()
        
        truck_timelines = []
        
        for tracker_id, track_info in self.truck_tracks.items():
            # Calculate duration in seconds
            fps = self.config.get('fps', 30)
            entry_frame = track_info['first_seen_frame']
            exit_frame = track_info['last_seen_frame']
            duration = (exit_frame - entry_frame) / fps
            
            # Add to timelines
            truck_timelines.append({
                'truck_id': tracker_id,
                'entry_frame': int(entry_frame),
                'entry_time': track_info['first_seen_time'],
                'exit_frame': int(exit_frame),
                'exit_time': track_info['last_seen_time'],
                'duration_seconds': duration
            })
        
        # Convert to DataFrame
        return pd.DataFrame(truck_timelines)

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
    
    def update_best_frame(self, detections_df, frame, frame_number, fps):
        """Store the best (highest confidence) frame for each truck."""
        if detections_df.empty or 'tracker_id' not in detections_df.columns:
            return
            
        # Get necessary directories
        best_frames_dir = os.path.join(os.path.dirname(self.log_file), '..', '..', 'best_frames')
        os.makedirs(best_frames_dir, exist_ok=True)
            
        for _, detection in detections_df.iterrows():
            if pd.isna(detection['tracker_id']) or pd.isna(detection['confidence']):
                continue
                
            tracker_id = int(detection['tracker_id'])
            confidence = float(detection['confidence'])
            
            # Get or initialize the best frame info for this truck
            if tracker_id not in self.best_frames:
                self.best_frames[tracker_id] = {
                    'confidence': confidence,
                    'frame_number': frame_number,
                    'frame_path': None,
                    'video_time': self.get_video_time_formatted(frame_number, fps)
                }
                # Always save the first frame
                should_save = True
            else:
                # Only save if confidence is higher than previous best
                should_save = confidence > self.best_frames[tracker_id]['confidence']
                if should_save:
                    self.best_frames[tracker_id]['confidence'] = confidence
                    self.best_frames[tracker_id]['frame_number'] = frame_number
                    self.best_frames[tracker_id]['video_time'] = self.get_video_time_formatted(frame_number, fps)
            
            if should_save:
                # Save the frame
                frame_filename = os.path.join(best_frames_dir, f"truck_{tracker_id}_frame_{frame_number}.jpg")
                cv2.imwrite(frame_filename, frame)
                self.best_frames[tracker_id]['frame_path'] = frame_filename
    
    def get_best_frames(self):
        """Get DataFrame with information about best frames for each truck."""
        if not self.best_frames:
            return pd.DataFrame()
            
        records = []
        for truck_id, frame_info in self.best_frames.items():
            records.append({
                'truck_id': truck_id,
                'confidence': frame_info['confidence'],
                'frame_number': frame_info['frame_number'],
                'frame_path': frame_info['frame_path'],
                'video_time': frame_info['video_time']
            })
            
        return pd.DataFrame(records)
    
    def save_best_frames_info(self, output_path=None):
        """Save information about best frames to CSV file."""
        if not output_path:
            output_path = os.path.join(os.path.dirname(self.log_file), 'best_frames_info.csv')
            
        frames_df = self.get_best_frames()
        if not frames_df.empty:
            frames_df.to_csv(output_path, index=False)
            print(f"Best frames info saved to {output_path}")
            return output_path
        return None
    
    def save_appearance_summary(self, output_path=None):
        """Save appearance summary to CSV file."""
        if not output_path:
            output_path = os.path.join(os.path.dirname(self.log_file), 'appearance_summary.csv')
        
        # Just save the truck timelines as the appearance summary
        timelines_df = self.get_truck_timelines()
        if not timelines_df.empty:
            timelines_df.to_csv(output_path, index=False)
            print(f"Appearance summary saved to {output_path}")
            
    def get_appearance_summary(self):
        """Get summary of truck appearances and disappearances."""
        # For the simplified version, we'll just return the truck timelines
        # as our appearance summary
        truck_timelines = self.get_truck_timelines()
        
        if truck_timelines.empty:
            return pd.DataFrame()
        
        # Rename columns to match expected format if needed
        summary = truck_timelines.copy()
        
        # Map column names to expected format if needed
        column_mapping = {
            'entry_time': 'appeared',
            'exit_time': 'disappeared',
            'entry_frame': 'start_frame',
            'exit_frame': 'end_frame'
        }
        
        # Rename only columns that exist in both 
        for old_col, new_col in column_mapping.items():
            if old_col in summary.columns:
                summary.rename(columns={old_col: new_col}, inplace=True)
        
        return summary