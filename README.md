Dear Flytbase Team, As I was not able to submit documentation in time. I will add links to the documententation here:
<Link1>
<Link2>

# Drone Security Analyst System

A comprehensive drone-based security monitoring and analysis system that uses computer vision and AI to detect security events from aerial footage.

## Overview

This system processes drone footage to detect objects, track movements, identify security incidents, and store events in a searchable database. It employs advanced computer vision techniques including object detection, tracking, and natural language captioning to provide actionable security insights.

## Features

- **Real-time Object Detection & Tracking**: Uses YOLOv8 - Finetuned on Visdrone dataset to detect and track objects in drone imagery
- **Security Event Recognition**: Implements rule-based detection for various security events:
  - Perimeter intrusions
  - Loitering detection ( WIP)
  - Crowd density monitoring ( WIP)
  - Abandoned object detection ( WIP)
  - Vehicle violations ( WIP)
- **Natural Language Captioning**: Automatically generates descriptive captions for security footage using BLIP
- **Persistent Storage**: Stores all events, detections and metadata in a searchable SQLite database
- **Visual Dashboard**: Streamlit-based UI for monitoring and querying security events
- **Zone Configuration**: Supports defining security zones with the polygon annotation helper

## Setup Instructions

### Prerequisites

- Python 3.11

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/drone_security_analyst.git
   cd drone_security_analyst
   ```
2. download visdone dataset from the link and add it to the root of your repository as data\sequences\
   <url> https://drive.google.com/file/d/1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh/view?usp=sharing <url>
   
4. Create virtual environment and Install dependencies:
   ```
5.  py -3.11 -m venv .env
   .env\Scripts\activate 
   pip install -r requirements.txt
   ```
   ```

5. Configure security zones (optional):
   ```
   python annotation_helper.py
   ```
   Click on the image to create polygon points for security zones. Right-click to finish and print coordinates.

### Usage

1. Run the main application:
   ```
   streamlit run drone_object_detection.py
   ```

2. In the application:
   - Select an image folder path containing drone imagery
   - Specify database path (or use default)
   - Click "Start Monitoring"
   - Use the "Database Query" tab to search for specific security events

## Architecture & Design Decisions

### Component Architecture

The system follows a modular design with these key components:

1. **Detection Engine** (`drone_object_detection.py`):
   - Core processing pipeline that handles image loading, object detection, and rule application
   - Integrates YOLOv8 model from Ultralytics for object detection
   - Implements tracking algorithms to maintain object identity across frames
   - Applies security rules to detect events of interest

2. **Database Layer** (`database.py`):
   - SQLAlchemy ORM model for structured storage
   - Handles frame, detection, and alert persistence
   - Implements query mechanisms including natural language query support
   - Provides image compression/decompression for efficient storage

3. **Configuration & Setup** (`annotation_helper.py`):
   - Utility for defining security zones through a visual interface
   - Generates polygon coordinates for use in security rules

4. **UI Dashboard** (Streamlit components):
   - Live monitoring view with real-time alerts
   - Database query interface with natural language support
   - Result visualization with annotated frames

### Design Decisions

1. **YOLOv8 Selection**: 
   - Chosen for its balance of speed and accuracy in drone-based imagery
   - Specifically using the 'mshamrai/yolov8l-visdrone' model, which is optimized for aerial viewpoints

2. **Rule-Based Analysis**:
   - Uses deterministic rules for security event detection rather than ML classification
   - Provides explainable results and avoids "black box" decision making
   - Allows for easy customization of security policies

3. **Temporal Analysis**:
   - Tracking objects across frames enables behavior analysis (loitering, abandonment)
   - Uses efficient data structures to maintain object history while minimizing memory usage

4. **SQLite Database**:
   - Provides structured storage without external dependencies
   - Enables complex queries through SQLAlchemy ORM
   - Includes frame image storage directly in the database for simplified deployment

5. **Streamlit UI**:
   - Enables rapid development of interactive dashboard
   - Supports both monitoring and querying functions
   - Provides simple deployment without complex front-end requirements

## AI Integration

This project integrates multiple AI tools to enhance security monitoring capabilities:

### 1. YOLO Object Detection

- **Model**: YOLOv8 (mshamrai/yolov8l-visdrone variant)
- **Purpose**: Detect and classify objects in drone footage
- **Impact**: Provides foundation for all security analytics by identifying entities of interest
- **Integration**: Implemented via Ultralytics library with custom confidence thresholds

### 2. BLIP Image Captioning

- **Model**: BLIP (Bootstrapped Language-Image Pre-training) from Salesforce
- **Purpose**: Generate natural language descriptions of scenes
- **Impact**: Enables semantic search and provides human-readable context for security events
- **Integration**: Uses Hugging Face's transformers library for inference

### 3. Object Tracking

- **Method**: Integrated tracker from Ultralytics
- **Purpose**: Maintain object identity across frames
- **Impact**: Enables behavioral analysis like loitering detection and abandoned object identification
- **Implementation**: `model.track()` with `persist=True` for consistent ID assignment

### Development Workflow Impact

The integration of these AI tools significantly improved the development workflow:

1. **Rapid Prototyping**: Pre-trained models allowed quick implementation of core functionality
2. **Reduced Annotation Requirements**: Transfer learning eliminated the need for extensive custom labeling
3. **Modular Testing**: Each AI component could be evaluated independently
4. **Iterative Refinement**: Started with base models and fine-tuned parameters based on performance


## Acknowledgments

- YOLOv8 by Ultralytics
- BLIP model by Salesforce Research
- VisDrone dataset for aerial imagery benchmarks
