# src/utils/report_generator.py
import os
import pandas as pd
import json
from datetime import datetime
import shutil

def generate_garbage_truck_report(truck_classifications, truck_timelines, output_dir=None):
    """
    Generate a final report of garbage truck appearances.
    
    Args:
        truck_classifications: DataFrame with truck classification results
        truck_timelines: DataFrame with truck entry/exit times
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report file
    """
    if output_dir is None:
        output_dir = 'outputs/reports'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge classification results with truck timelines
    merged_df = pd.merge(
        truck_classifications,
        truck_timelines,
        on='truck_id',
        how='inner'
    )
    
    # Filter for garbage trucks only
    garbage_trucks = merged_df[merged_df['is_garbage_truck'] == True].copy()
    
    if garbage_trucks.empty:
        print("No garbage trucks identified in the video.")
        return None
    
    # Format the report with relevant columns
    report_df = garbage_trucks[[
        'truck_id', 'entry_frame', 'entry_time', 'exit_frame', 'exit_time',
        'duration_seconds', 'image_path', 'video_time', 'classification_confidence',
        'llm_response'
    ]].copy()
    
    # Add timestamp to report filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'garbage_truck_report_{timestamp}.csv')
    
    # Save report
    report_df.to_csv(report_path, index=False)
    print(f"Garbage truck report generated: {report_path}")
    
    # Generate summary statistics
    total_trucks = len(report_df)
    total_duration = report_df['duration_seconds'].sum()
    avg_duration = report_df['duration_seconds'].mean()
    
    print(f"Summary: {total_trucks} garbage trucks detected")
    print(f"Total time on camera: {total_duration:.2f} seconds")
    print(f"Average duration: {avg_duration:.2f} seconds per truck")
    
    return report_path

def generate_html_report(report_path, output_dir=None):
    """
    Generate an enhanced HTML report with embedded images and LLM reasoning.
    
    Args:
        report_path: Path to the CSV report
        output_dir: Directory to save the HTML report
    
    Returns:
        Path to the HTML report
    """
    if not os.path.exists(report_path):
        print(f"Report file not found: {report_path}")
        return None
    
    if output_dir is None:
        output_dir = os.path.dirname(report_path)
    
    # Create report directory
    report_name = os.path.splitext(os.path.basename(report_path))[0]
    report_dir = os.path.join(output_dir, report_name)
    os.makedirs(report_dir, exist_ok=True)
    
    # Create images directory in the report folder
    images_dir = os.path.join(report_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Load the report
    report_df = pd.read_csv(report_path)
    
    # Copy images to the report folder
    for i, row in report_df.iterrows():
        src_image = row['image_path']
        if os.path.exists(src_image):
            dest_image = os.path.join(images_dir, f"truck_{row['truck_id']}.jpg")
            shutil.copy2(src_image, dest_image)
            # Update the path in the dataframe for the HTML report
            report_df.at[i, 'report_image'] = os.path.relpath(dest_image, report_dir)
        else:
            report_df.at[i, 'report_image'] = ""
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Garbage Truck Detection Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .truck-entry {{
                border: 1px solid #ddd;
                padding: 20px;
                margin-bottom: 30px;
                border-radius: 8px;
                background-color: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .truck-image-container {{ display: flex; justify-content: center; margin: 15px 0; }}
            .truck-image {{
                max-width: 100%;
                max-height: 400px;
                border-radius: 4px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}
            .truck-details {{
                margin-top: 15px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 10px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .summary {{
                margin-bottom: 40px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 8px;
                border-left: 4px solid #4CAF50;
            }}
            .reason-box {{
                margin-top: 15px;
                padding: 15px;
                background-color: #f0f8ff;
                border-radius: 5px;
                border-left: 4px solid #2196F3;
            }}
            h1, h2 {{ color: #333; }}
            h3 {{ color: #555; }}
            .timestamp {{ color: #666; font-size: 0.9em; margin-top: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Garbage Truck Detection Report</h1>
                <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total garbage trucks detected:</strong> {len(report_df)}</p>
                <p><strong>Total time on camera:</strong> {report_df['duration_seconds'].sum():.2f} seconds</p>
                <p><strong>Average duration:</strong> {report_df['duration_seconds'].mean():.2f} seconds per truck</p>
            </div>
            
            <h2>Detected Garbage Trucks</h2>
    """
    
    # Add each truck entry
    for _, truck in report_df.iterrows():
        # Format entry/exit times for display
        entry_time = truck['entry_time']
        exit_time = truck['exit_time']
        
        # Format duration
        duration = f"{truck['duration_seconds']:.2f} seconds"
        
        # Confidence percentage
        confidence = f"{truck['classification_confidence'] * 100:.1f}%"
        
        html_content += f"""
        <div class="truck-entry">
            <h3>Truck ID: {truck['truck_id']}</h3>
            
            <div class="truck-image-container">
                <img src="{truck['report_image']}" class="truck-image" alt="Truck {truck['truck_id']}" 
                    onerror="this.onerror=null; this.src=''; this.alt='Image not available';">
            </div>
            
            <div class="truck-details">
                <table>
                    <tr><th>Entry Time</th><td>{entry_time}</td></tr>
                    <tr><th>Exit Time</th><td>{exit_time}</td></tr>
                    <tr><th>Duration</th><td>{duration}</td></tr>
                    <tr><th>Entry Frame</th><td>{int(truck['entry_frame'])}</td></tr>
                    <tr><th>Exit Frame</th><td>{int(truck['exit_frame'])}</td></tr>
                    <tr><th>Classification Confidence</th><td>{confidence}</td></tr>
                </table>
                
                <div class="reason-box">
                    <h4>GPT-4o Analysis:</h4>
                    <p>{truck['llm_response']}</p>
                </div>
            </div>
        </div>
        """
    
    html_content += """
            </div>
        </body>
        </html>
    """
    
    # Save HTML report
    html_path = os.path.join(report_dir, 'report.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Enhanced HTML report generated: {html_path}")
    return html_path

def format_video_time(frame, fps):
    """Convert frame number to HH:MM:SS.ff format."""
    seconds = frame / fps
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"

def create_detection_timeline(report_path, fps=30, output_dir=None):
    """
    Create a visual timeline of garbage truck detections.
    
    Args:
        report_path: Path to the garbage truck report CSV
        fps: Frames per second of the video
        output_dir: Directory to save the timeline
        
    Returns:
        Path to the HTML timeline
    """
    if not os.path.exists(report_path):
        print(f"Report file not found: {report_path}")
        return None
    
    # Load the report
    report_df = pd.read_csv(report_path)
    
    if report_df.empty:
        print("No data in the report to create a timeline.")
        return None
    
    if output_dir is None:
        output_dir = os.path.dirname(report_path)
    
    # Sort by entry time
    report_df = report_df.sort_values('entry_frame')
    
    # Find total video duration
    if 'exit_frame' in report_df.columns:
        max_frame = report_df['exit_frame'].max()
        video_duration = max_frame / fps  # in seconds
    else:
        video_duration = 300  # default 5 minutes if no exit frame
    
    # Calculate minute markers
    minutes = int(video_duration // 60) + 1
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Garbage Truck Detection Timeline</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px; 
                background-color: white; 
                border-radius: 8px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
            }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .timeline {{
                position: relative;
                height: 80px;
                background-color: #f0f0f0;
                margin: 20px 0;
                border-radius: 4px;
            }}
            .timeline-scale {{
                position: relative;
                height: 20px;
                margin-top: 5px;
            }}
            .minute-marker {{
                position: absolute;
                height: 10px;
                border-left: 1px solid #999;
                top: 0;
            }}
            .minute-label {{
                position: absolute;
                font-size: 10px;
                color: #666;
                top: 12px;
                transform: translateX(-50%);
            }}
            .truck-marker {{
                position: absolute;
                height: 60px;
                background-color: rgba(76, 175, 80, 0.7);
                border: 1px solid #388E3C;
                border-radius: 3px;
                top: 10px;
                cursor: pointer;
            }}
            .truck-tooltip {{
                position: absolute;
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                font-size: 12px;
                z-index: 100;
                display: none;
            }}
            .truck-marker:hover .truck-tooltip {{
                display: block;
            }}
            .timestamp {{ color: #666; font-size: 0.9em; margin-top: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Garbage Truck Detection Timeline</h1>
                <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="timeline">
    """
    
    # Add truck markers to the timeline
    for _, truck in report_df.iterrows():
        entry_frame = truck['entry_frame']
        exit_frame = truck['exit_frame']
        
        # Convert frames to timeline position (%)
        start_percent = (entry_frame / (fps * 60 * minutes)) * 100
        duration_percent = ((exit_frame - entry_frame) / (fps * 60 * minutes)) * 100
        
        # Format times for display
        entry_time = format_video_time(entry_frame, fps)
        exit_time = format_video_time(exit_frame, fps)
        
        html_content += f"""
                <div class="truck-marker" style="left: {start_percent:.2f}%; width: {duration_percent:.2f}%;">
                    <div class="truck-tooltip">
                        <strong>Truck ID:</strong> {truck['truck_id']}<br>
                        <strong>Entry:</strong> {entry_time}<br>
                        <strong>Exit:</strong> {exit_time}<br>
                        <strong>Duration:</strong> {truck['duration_seconds']:.2f}s
                    </div>
                </div>
        """
    
    html_content += """
            </div>
            
            <div class="timeline-scale">
    """
    
    # Add minute markers
    for i in range(minutes + 1):
        position_percent = (i / minutes) * 100
        html_content += f"""
                <div class="minute-marker" style="left: {position_percent:.2f}%;"></div>
                <div class="minute-label" style="left: {position_percent:.2f}%;">{i}:00</div>
        """
    
    html_content += """
                        </div>
            
            <div class="summary-details">
                <h3>Video Summary</h3>
                <ul>
                    <li><strong>Total video duration:</strong> {minutes} minutes ({video_duration:.2f} seconds)</li>
                    <li><strong>Total garbage trucks detected:</strong> {len(report_df)}</li>
                    <li><strong>Average time on camera:</strong> {report_df['duration_seconds'].mean():.2f} seconds</li>
                </ul>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Truck ID</th>
                        <th>Entry Time</th>
                        <th>Exit Time</th>
                        <th>Duration</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add table rows for each truck
    for _, truck in report_df.iterrows():
        entry_time = format_video_time(truck['entry_frame'], fps)
        exit_time = format_video_time(truck['exit_frame'], fps)
        duration = f"{truck['duration_seconds']:.2f}s"
        confidence = f"{truck['classification_confidence'] * 100:.1f}%"
        
        html_content += f"""
                    <tr>
                        <td>{truck['truck_id']}</td>
                        <td>{entry_time}</td>
                        <td>{exit_time}</td>
                        <td>{duration}</td>
                        <td>{confidence}</td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
"""
    
    # Save HTML timeline
    timeline_path = os.path.join(output_dir, 'detection_timeline.html')
    with open(timeline_path, 'w') as f:
        f.write(html_content)
    
    print(f"Timeline visualization generated: {timeline_path}")
    return timeline_path