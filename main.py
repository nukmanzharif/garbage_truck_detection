# main.py
import os
import argparse
import pandas as pd
import yaml
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from src.detection.model import DetectionModel
from src.detection.tracker import ObjectTracker
from src.pipeline.timestamp_logger import TimestampLogger
from src.pipeline.video_processor import VideoProcessor
from src.utils.data_utils import load_config
from src.utils.report_generator import (
    generate_garbage_truck_report, 
    generate_html_report,
    create_detection_timeline
)
from llm_classifier.classifier import GarbageTruckClassifier



def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Garbage Truck Detection System')
    parser.add_argument('--input', type=str, help='Input video path or webcam index')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--detector-config', type=str, default='configs/detector_config.yaml', 
                        help='Path to detector configuration')
    parser.add_argument('--pipeline-config', type=str, default='configs/pipeline_config.yaml', 
                        help='Path to pipeline configuration')
    parser.add_argument('--llm-config', type=str, default='configs/llm_config.yaml',
                        help='Path to LLM configuration')
    parser.add_argument('--classify', action='store_true', 
                        help='Enable GPT-4o classification of trucks')
    parser.add_argument('--skip-video', action='store_true',
                        help='Skip video processing and only run classification on existing frames')
    parser.add_argument('--report-only', action='store_true',
                        help='Generate reports from existing classification results')
    
    args = parser.parse_args()
    
    # Create timestamp for this run
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load configurations
    detector_config = load_config(args.detector_config)
    pipeline_config = load_config(args.pipeline_config)
    
    # Override config with command line args
    if args.input:
        pipeline_config['input']['source'] = args.input
    if args.output:
        pipeline_config['output']['output_path'] = args.output
    
    # Setup output directories
    output_dir = os.path.dirname(pipeline_config['output']['output_path'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Create run directory for this execution
    run_dir = os.path.join(output_dir, f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Update output paths to use run directory
    frames_info_path = os.path.join(run_dir, 'extracted_frames.csv')
    truck_timelines_path = os.path.join(run_dir, 'truck_timelines.csv')
    classified_path = os.path.join(run_dir, 'classified_trucks.csv')
    
    # Process video if not skipping
    if not args.skip_video and not args.report_only:
        # Initialize detector
        print("Initializing detection model...")
        detector = DetectionModel(detector_config)
        
        # Initialize tracker
        print("Initializing object tracker...")
        tracker = ObjectTracker(pipeline_config['tracking'])
        
        # Initialize timestamp logger
        print("Initializing timestamp logger...")
        logger_config = pipeline_config['logging']
        timestamp_logger = TimestampLogger(logger_config)
        
        # Initialize video processor
        print("Initializing video processor...")
        processor = VideoProcessor(detector, tracker, timestamp_logger, pipeline_config)
        
        # Process video
        print("Starting video processing...")
        processor.process()
        
        # Check for best frames first
        best_frames_path = os.path.join(os.path.dirname(pipeline_config['output']['output_path']), 'best_frames_info.csv')
        if os.path.exists(best_frames_path):
            # Use best frames for classification
            frames_df = pd.read_csv(best_frames_path)
            frames_df['image_path'] = frames_df['frame_path']  # Ensure compatibility with classifier
            frames_df.to_csv(frames_info_path, index=False)
            extracted_frames = frames_df.to_dict('records')
            print(f"Using {len(extracted_frames)} best frames for classification.")
        else:
            # Extract frames for unique trucks as fallback
            print("Extracting truck frames...")
            extracted_frames = processor.extract_truck_frames()
            
            if extracted_frames is None or len(extracted_frames) == 0:
                print("No trucks detected in the video. Exiting.")
                return
                
            # Save the extracted frames info to the run directory
            frames_df = pd.DataFrame(extracted_frames)
            frames_df.to_csv(frames_info_path, index=False)
        
        # Save truck timelines to the run directory
        truck_timelines = timestamp_logger.get_truck_timelines()
        truck_timelines.to_csv(truck_timelines_path, index=False)
        
        print(f"Extracted {len(extracted_frames)} truck frames.")
    else:
        print("Skipping video processing...")
        # If report only, look for existing classification results
        if args.report_only:
            # Get paths to existing classification results
            if os.path.exists(args.input) and args.input.endswith('.csv'):
                classified_path = args.input
            else:
                print("For --report-only, please provide the classification results CSV with --input")
                return
        else:
            # Get paths to existing files for classification
            if args.input and os.path.exists(args.input) and args.input.endswith('.csv'):
                frames_info_path = args.input
            else:
                # Look for the most recent extracted_frames.csv if none specified
                output_folders = [f for f in os.listdir(output_dir) if f.startswith('run_')]
                if output_folders:
                    latest_run = sorted(output_folders)[-1]
                    frames_info_path = os.path.join(output_dir, latest_run, 'extracted_frames.csv')
                    truck_timelines_path = os.path.join(output_dir, latest_run, 'truck_timelines.csv')
                else:
                    print("No existing extraction results found. Please run video processing first.")
                    return
    
    # Run classification if requested and not report only
    if args.classify and not args.report_only:
        if os.path.exists(frames_info_path) and os.path.exists(truck_timelines_path):
            print("Starting garbage truck classification with GPT-4o...")
            
            # Load truck frames info
            frames_df = pd.read_csv(frames_info_path)
            
            # Initialize classifier
            classifier = GarbageTruckClassifier(args.llm_config)
            
            # Run classification
            print(f"Classifying {len(frames_df)} truck images with GPT-4o...")
            classified_df = classifier.classify_truck_images(frames_df)
            
            # Save classification results
            classified_df.to_csv(classified_path, index=False)
            print(f"Classification results saved to {classified_path}")
            
            # Print summary of classification
            garbage_trucks = classified_df[classified_df['is_garbage_truck'] == True]
            if not garbage_trucks.empty:
                print(f"\nFound {len(garbage_trucks)} garbage trucks out of {len(classified_df)} trucks detected.")
            else:
                print("No garbage trucks identified in the video.")
        else:
            print(f"Cannot run classification: truck frames or timelines not found.")
            print(f"Expected files at {frames_info_path} and {truck_timelines_path}")
            return
    
    # Generate reports
    if args.report_only or args.classify:
        # Get paths to necessary files
        if args.report_only:
            # We already have the classified_path from args.input
            if not os.path.exists(classified_path):
                print(f"Classification results not found at {classified_path}")
                return
                
            # Look for truck_timelines.csv in the same directory
            truck_timelines_path = os.path.join(
                os.path.dirname(classified_path), 
                'truck_timelines.csv'
            )
            
            if not os.path.exists(truck_timelines_path):
                print(f"Truck timelines not found at {truck_timelines_path}")
                # Try to find it in a parent directory
                parent_dir = os.path.dirname(os.path.dirname(classified_path))
                possible_paths = [os.path.join(parent_dir, 'truck_timelines.csv')]
                for path in possible_paths:
                    if os.path.exists(path):
                        truck_timelines_path = path
                        break
                else:
                    print("Could not find truck_timelines.csv. Some report features may be limited.")
                    # Create a minimal truck_timelines from classification results
                    classified_df = pd.read_csv(classified_path)
                    if 'truck_id' in classified_df.columns:
                        truck_timelines = pd.DataFrame({
                            'truck_id': classified_df['truck_id'],
                            'entry_frame': 0,
                            'exit_frame': 0,
                            'duration_seconds': 0
                        })
                    else:
                        print("Cannot generate reports without truck timeline information.")
                        return
        
        # Load data
        classified_df = pd.read_csv(classified_path)
        truck_timelines = pd.read_csv(truck_timelines_path) if os.path.exists(truck_timelines_path) else None
        
        # Create reports directory
        reports_dir = os.path.join(run_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate garbage truck report
        if truck_timelines is not None:
            print("Generating garbage truck reports...")
            report_path = generate_garbage_truck_report(classified_df, truck_timelines, reports_dir)
            
            if report_path:
                # Generate HTML report
                html_path = generate_html_report(report_path, reports_dir)
                print(f"HTML report available at: {html_path}")
                
                # Generate timeline visualization
                timeline_path = create_detection_timeline(report_path, fps=30, output_dir=reports_dir)
                print(f"Timeline visualization available at: {timeline_path}")
            else:
                print("No garbage trucks detected, no report generated.")
        else:
            print("Truck timelines not available, cannot generate full reports.")
            
            # Create a simplified report with just classification results
            garbage_trucks = classified_df[classified_df['is_garbage_truck'] == True]
            if not garbage_trucks.empty:
                simplified_report_path = os.path.join(reports_dir, 'simplified_report.csv')
                garbage_trucks.to_csv(simplified_report_path, index=False)
                print(f"Simplified report saved to {simplified_report_path}")
    
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()