# src/utils/report_generator.py
import os
import pandas as pd
import json
from datetime import datetime
import shutil

def generate_garbage_truck_report(truck_classifications, truck_timelines, output_dir=None):
    """
    Generate a simplified report of garbage truck appearances.
    
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
    
    # Format the report with only necessary columns
    report_df = garbage_trucks[[
        'truck_id', 'entry_frame', 'entry_time', 'exit_frame', 'exit_time',
        'duration_seconds', 'image_path', 'classification_confidence'
    ]].copy()
    
    # Add timestamp to report filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'garbage_truck_report_{timestamp}.csv')
    
    # Save report
    report_df.to_csv(report_path, index=False)
    print(f"Garbage truck report generated: {report_path}")
    
    return report_path

def generate_html_report(report_path, output_dir=None):
    """
    Generate a simplified HTML report with embedded images.
    
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
            h1, h2 {{ color: #333; }}
            h3 {{ color: #555; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Garbage Truck Detection Report</h1>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total garbage trucks detected:</strong> {len(report_df)}</p>
            </div>
            
            <h2>Detected Garbage Trucks</h2>
    """
    
    # Add each truck entry
    for _, truck in report_df.iterrows():
        # Format entry/exit times for display (already in video time format)
        entry_time = truck['entry_time']
        exit_time = truck['exit_time']
        
        # Format duration
        duration = f"{truck['duration_seconds']:.2f} seconds"
        
        html_content += f"""
        <div class="truck-entry">
            <h3>Truck ID: {truck['truck_id']}</h3>
            
            <div class="truck-image-container">
                <img src="{truck['report_image']}" class="truck-image" alt="Truck {truck['truck_id']}" 
                    onerror="this.onerror=null; this.src=''; this.alt='Image not available';">
            </div>
            
            <div class="truck-details">
                <table>
                    <tr><th>First Seen</th><td>{entry_time}</td></tr>
                    <tr><th>Last Seen</th><td>{exit_time}</td></tr>
                    <tr><th>Duration</th><td>{duration}</td></tr>
                </table>
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
    
    print(f"HTML report generated: {html_path}")
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
    Create a simplified visual timeline of garbage truck detections.
    
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
    
    # Create HTML content - simplified timeline
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
            .timeline {{
                position: relative;
                margin: 40px 0;
                height: 80px;
                background-color: #f0f0f0;
                border-radius: 4px;
            }}
            .timeline-marker {{
                position: absolute;
                top: 0;
                height: 100%;
                background-color: #4CAF50;
                opacity: 0.7;
                border-radius: 4px;
            }}
            .timeline-label {{
                position: absolute;
                bottom: -25px;
                transform: translateX(-50%);
                font-size: 12px;
                color: #333;
            }}
            h1, h2 {{ color: #333; text-align: center; }}
            .truck-details {{
                margin-bottom: 40px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 8px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Garbage Truck Detection Timeline</h1>
            
            <div class="timeline">
    """
    
    # Add markers for each truck appearance
    for _, truck in report_df.iterrows():
        entry_percent = (truck['entry_frame'] / max_frame) * 100
        exit_percent = (truck['exit_frame'] / max_frame) * 100
        width_percent = exit_percent - entry_percent
        
        html_content += f"""
                <div class="timeline-marker" style="left: {entry_percent}%; width: {width_percent}%;">
                    <div class="timeline-label">ID: {int(truck['truck_id'])}</div>
                </div>
        """
    
    html_content += """
            </div>
            
            <div class="truck-details">
                <h2>Truck Appearance Details</h2>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #f2f2f2;">
                        <th style="padding: 10px; border: 1px solid #ddd;">Truck ID</th>
                        <th style="padding: 10px; border: 1px solid #ddd;">First Seen</th>
                        <th style="padding: 10px; border: 1px solid #ddd;">Last Seen</th>
                        <th style="padding: 10px; border: 1px solid #ddd;">Duration</th>
                    </tr>
    """
    
    # Add rows for each truck
    for _, truck in report_df.iterrows():
        html_content += f"""
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;">{int(truck['truck_id'])}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{truck['entry_time']}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{truck['exit_time']}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{truck['duration_seconds']:.2f} seconds</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML timeline
    timeline_path = os.path.join(output_dir, 'truck_timeline.html')
    with open(timeline_path, 'w') as f:
        f.write(html_content)
    
    print(f"Timeline visualization generated: {timeline_path}")
    return timeline_path