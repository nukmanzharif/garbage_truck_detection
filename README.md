# Garbage Truck Detection System

An AI-powered system for detecting and classifying garbage trucks in videos, providing accurate timestamps and visual evidence of their appearances.

## Features

- Real-time garbage truck detection using YOLOv8
- Object tracking for consistent truck identification
- AI-powered classification using OpenAI GPT-4o Vision
- Automatic capture of best quality frames
- Real-time annotation of detected trucks
- Detailed timestamp logging of truck appearances and disappearances
- Comprehensive reports and visualizations
- HTML report generation with timeline views

## System Overview

The system operates in three main phases:
1. **Detection & Tracking**: Identifies trucks in video footage and tracks them across frames
2. **Visual AI Classification**: Uses GPT-4o to determine if the detected trucks are garbage trucks
3. **Reporting & Analysis**: Generates detailed reports of garbage truck appearances with timestamps

## Requirements

- Python 3.11+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- OpenAI API key for GPT-4o vision
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/garbage_truck_detection.git
cd garbage_truck_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your OpenAI API key:

OPENAI_API_KEY=your_api_key_here


## Usage

### Basic Usage

Run the complete pipeline (detection, tracking, classification, and reporting):
```bash
python main.py --input path/to/video.mp4 --classify
```

### Advanced Options

#### Video Processing Only
```bash
python main.py --input path/to/video.mp4
```

#### Skip Video Processing & Only Run Classification
```bash
python main.py --skip-video --classify --input path/to/extracted_frames.csv
```

#### Generate Reports from Existing Results
```bash
python main.py --report-only --input path/to/classification_results.csv
```

#### Customize Configuration
```bash
python main.py --detector-config configs/custom_detector.yaml --pipeline-config configs/custom_pipeline.yaml --llm-config configs/custom_llm.yaml
```

## Configuration Files

The system uses three configuration files:

1. `configs/detector_config.yaml` - Detection model settings (YOLOv8)
2. `configs/pipeline_config.yaml` - Video processing and tracking settings
3. `configs/llm_config.yaml` - GPT-4o vision API settings

## Output Files

The system produces several outputs:

1. **Annotated Video**: Shows detections with bounding boxes
2. **Best Frame Images**: High-confidence screenshots of each truck with detection boxes
3. **Detection Logs**: CSV files with detailed detection data
4. **Truck Timelines**: Entry and exit times for each truck
5. **Classification Results**: LLM determination of garbage truck status
6. **Reports**: Summary reports in CSV, HTML, and visualization formats



