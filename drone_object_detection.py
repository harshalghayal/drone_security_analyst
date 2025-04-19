import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import asyncio
try:
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except AttributeError:
    # Non-Windows: optionally patch with nest_asyncio
    import nest_asyncio  # pip install nest_asyncio
    nest_asyncio.apply()     # :contentReference[oaicite:5]{index=5}

import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralyticsplus import YOLO
import random
from datetime import datetime
from collections import Counter
import time
import hashlib
import json
# Import the database module
from database import DroneSecurityDB

#def hash_state(detections, alerts):
#    state = json.dumps({'dets': detections, 'alerts': alerts}, sort_keys=True, default=str)
#    return hashlib.md5(state.encode()).hexdigest()

def hash_state(counts):
    state = json.dumps(counts, sort_keys=True)
    return hashlib.md5(state.encode()).hexdigest()

def serialize_detections(dets):
    return [{'id': d['id'], 'centroid': d['centroid'], 'cls': d['cls']} for d in dets]



# ------------------ CONFIGURATION ------------------
fence_polygon = [(364, 755), (683, 58), (746, 62), (1121, 751)]
restricted_zone = [(600,100), (900,100), (900,400), (600,400)]
crowd_area = [(200,450), (800,450), (800,800), (200,800)]
restricted_zones = [restricted_zone]

allowed_start = datetime.strptime("08:00", "%H:%M").time()
allowed_end   = datetime.strptime("18:00", "%H:%M").time()

MAX_STATIONARY_FRAMES = 50
LOITER_RADIUS = 200
ABANDON_SECONDS = 30
FPS = 10
OBJECT_CLASSES = {'backpack', 'handbag', 'suitcase', 'box'}
allowed_directions = [(0,180)]

# ------------------ ALERT MANAGEMENT ------------------
current_alerts = []


def send_alert(rule_name, det):
    current_alerts.append((rule_name, det))

# ------------------ UTILITY ------------------
def in_polygon(pt, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), pt, False) >= 0

def simulate_telemetry_data():
    return {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'location': (random.randint(0, 1200), random.randint(0, 800)),
        'altitude': random.randint(100, 500)
    }

# ------------------ RULE FUNCTIONS ------------------
last_intruder_ids = set()
last_intruder_ids = set()

def rule_perimeter_intrusion(detections):
    global last_intruder_ids
    current_intruders = {
        f"{det['cls']}:{det['id']}" for det in detections if not in_polygon(det['centroid'], fence_polygon)
    }

    # Only send alert if new intruders appear
    if current_intruders and current_intruders != last_intruder_ids:
        new_intruders = current_intruders - last_intruder_ids
        if new_intruders:
            send_alert("Perimeter Intrusion", {'intruder_ids': list(current_intruders)})
        last_intruder_ids = current_intruders



def rule_loitering(track_history, frame_idx):
    for tid, history in track_history.items():
        if len(history) >= MAX_STATIONARY_FRAMES:
            pts = history[-MAX_STATIONARY_FRAMES:]
            xs, ys = zip(*pts)
            if max(xs)-min(xs) < LOITER_RADIUS and max(ys)-min(ys) < LOITER_RADIUS:
                send_alert("Loitering Detected", {'id': tid, 'location': pts[-1]})

def rule_time_restricted_entry(detections):
    now = datetime.now().time()
    if not (allowed_start <= now <= allowed_end):
        for det in detections:
            if det['cls'] in {'person','car'} and in_polygon(det['centroid'], restricted_zone):
                send_alert("After-Hours Intrusion", det)

def rule_crowd_density(detections):
    count = sum(1 for det in detections if det['cls']=='person' and in_polygon(det['centroid'], crowd_area))
    if count > 10:
        send_alert(f"Crowd Density Exceeded ({count})", None)

def rule_abandoned_object(track_history, detection_times, frame_idx):
    threshold = ABANDON_SECONDS * FPS
    for tid, history in track_history.items():
        if len(history) >= threshold:
            send_alert("Abandoned Object", {'id': tid, 'location': history[-1]})

def rule_vehicle_violation(detections, head_dict):
    for det in detections:
        if det['cls']=='car':
            if any(in_polygon(det['centroid'], zone) for zone in restricted_zones):
                send_alert("Vehicle in Restricted Zone", det)
            hd = head_dict.get(det['id'])
            if hd is not None and not any(start <= hd <= end for start, end in allowed_directions):
                send_alert("Wrong-Way Vehicle", det)

# ------------------ MODELS & DETECTION ------------------
COLOR_MAP = {}
def get_color_for_class(cls_id):
    if cls_id not in COLOR_MAP:
        COLOR_MAP[cls_id] = tuple(random.randint(0,255) for _ in range(3))
    return COLOR_MAP[cls_id]

def load_yolo_model():
    import torch
    from ultralytics.nn.tasks import DetectionModel

    # Allowlist the DetectionModel class
    torch.serialization.add_safe_globals([DetectionModel])

    model = YOLO('mshamrai/yolov8l-visdrone')
    model.overrides.update({'conf': 0.25, 'iou': 0.45, 'agnostic_nms': False, 'max_det': 1000})
    return model

def load_captioning_model():
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    mod  = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return proc, mod

def generate_caption(frame, processor, model):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(img, return_tensors="pt")
    with torch.no_grad(): out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

from collections import defaultdict
class_id_counter = defaultdict(int)


def detect_and_track(frame, model):
    # Run YOLOv8 in track mode with persistence
    results = model.track(source=frame, persist=True, verbose=False)
    if not results or not results[0].boxes:
        return frame, [], {}

    res = results[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    ids   = res.boxes.id.cpu().numpy().astype(int)       # <— persistent tracker IDs
    cls   = res.boxes.cls.cpu().numpy().astype(int)
    names = model.names

    detections = []
    for box, tid, cid in zip(boxes, ids, cls):
        x1, y1, x2, y2 = map(int, box)
        centroid = ((x1 + x2)//2, (y1 + y2)//2)
        class_name = names[cid]

        detections.append({
            'cls': class_name,
            'centroid': centroid,
            'id': int(tid),                            # <— use tracker ID
            'box': (x1, y1, x2, y2)
        })
        color = get_color_for_class(cid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{class_name}:{tid}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    counts = Counter(names[c] for c in cls)
    return frame, detections, dict(counts)


# ------------------ STREAMLIT APP ------------------

st.set_page_config(page_title="Drone Security Analyst", layout="wide")
st.title("Drone Security Analyst Dashboard")

# Sidebar for folder input and database options
st.sidebar.subheader("Input Options")
folder = st.sidebar.text_input("Drone Image Folder Path", value='data/sequences/uav0000013_00000_v')
db_path = st.sidebar.text_input("Database Path", value='drone_security.db')
start = st.sidebar.button("Start Monitoring")

# Create database tabs
tab1, tab2 = st.tabs(["Live Monitoring", "Database Query"])

# Database query tab
with tab2:
    st.header("Natural Language Database Query")
    query_text = st.text_input("Enter your query (e.g., 'show me frames with person alerts')")
    run_query = st.button("Run Query")
    
    if run_query and query_text:
        try:
            # Initialize the database
            db = DroneSecurityDB(db_path=db_path)
            # Run the natural language query
            results = db.natural_language_query(query_text)
            
            if not results.empty:
                st.success(f"Found {len(results)} matching results")
                st.dataframe(results)
                
                # Display a few sample images
                if len(results) > 0:
                    # Use existing annotations from database
                    annotate_files = st.checkbox("Apply stored annotations to original files", value=True)
                    
                    # Create debug info expander first so it's available everywhere
                    debug_info = st.expander("Debug Info")
                    
                    # First, get a list of all valid frame IDs that actually exist in the database
                    session = db.Session()
                    try:
                        # Import the Frame class directly instead of trying to access it through db
                        from database import Frame
                        
                        existing_frames_query = session.query(Frame.id).all()
                        existing_frame_ids = [f[0] for f in existing_frames_query]
                        debug_info.write(f"[DEBUG] Found {len(existing_frame_ids)} total frames in database")
                        if existing_frame_ids:
                            debug_info.write(f"[DEBUG] Sample existing frame IDs: {existing_frame_ids[:10]}")
                        else:
                            debug_info.write("[WARNING] No frames found in database")
                    except Exception as e:
                        debug_info.write(f"[ERROR] Error querying existing frames: {str(e)}")
                        existing_frame_ids = []
                    finally:
                        session.close()
                    
                    # Get frames from results that actually exist in the database
                    result_frame_ids = results['frame_id'].unique()
                    valid_frame_ids = [fid for fid in result_frame_ids if fid in existing_frame_ids]
                    
                    display_frames = True  # Flag to control whether to display frames
                    
                    if not valid_frame_ids:
                        debug_info.write("[WARNING] None of the result frame IDs exist in the database")
                        # Fall back to using some existing frames if available
                        sample_frames = existing_frame_ids[:5] if existing_frame_ids else []
                        if sample_frames:
                            debug_info.write(f"[INFO] Falling back to existing frames: {sample_frames}")
                        else:
                            st.warning("No frames found in the database to display")
                            display_frames = False  # Skip frame display
                    else:
                        sample_frames = valid_frame_ids[:5]
                        debug_info.write(f"[INFO] Using valid frame IDs from results: {sample_frames}")
                    
                    if not sample_frames:
                        debug_info.write("[WARNING] No valid frames to display")
                        st.warning("No valid frames to display")
                        display_frames = False  # Skip frame display
                        
                    # Make sure we have at least one frame to display
                    num_columns = min(len(sample_frames), 3) if sample_frames else 0
                    if num_columns < 1:
                        debug_info.write("[ERROR] Cannot create 0 columns. Check sample_frames.")
                        st.error("Internal error: No frames to display")
                        display_frames = False  # Skip frame display
                        
                    # Only proceed with frame display if we have valid frames
                    if display_frames:
                        st.subheader(f"Sample Frames (IDs: {', '.join(map(str, sample_frames))})")
                        
                        debug_info.write("Starting frame retrieval process...")
                        
                        # Create a placeholder for status messages
                        status_placeholder = st.empty()
                        status_placeholder.info("Retrieving frames from database...")
                        
                        # Create columns only if we have frames to display
                        cols = st.columns(num_columns)
                        successful_frames = 0
                        
                        for i, frame_id in enumerate(sample_frames):
                            try:
                                debug_info.write(f"Retrieving frame ID: {frame_id}")
                                # Try getting the frame from the database first
                                img = db.get_frame_image(frame_id)
                                
                                if img is not None and isinstance(img, np.ndarray) and img.size > 0:
                                    debug_info.write(f"Frame {frame_id} retrieved successfully from database. Shape: {img.shape}")
                                    # Convert BGR to RGB for display
                                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    cols[i % 3].image(img_rgb, 
                                                   caption=f"Frame ID: {frame_id}", 
                                                   use_column_width=True)
                                    successful_frames += 1
                                else:
                                    debug_info.write(f"⚠️ Frame {frame_id} could not be retrieved from database")
                                    
                                    # Try to retrieve the original file from file system
                                    try:
                                        # Get frame info and detection data from database
                                        frame_info = results[results['frame_id'] == frame_id]
                                        if frame_info.empty:
                                            debug_info.write(f"No frame info found for frame ID: {frame_id}")
                                            cols[i % 3].error(f"No info for frame ID: {frame_id}")
                                            continue
                                        
                                        frame_info = frame_info.iloc[0]
                                        frame_number = frame_info.get('frame_number')
                                        folder_path = frame_info.get('folder_path', folder)
                                        filename = frame_info.get('filename')
                                        
                                        # Get detection data from database for this frame
                                        session = db.Session()
                                        
                                        # If we have the filename directly, use it
                                        if filename and os.path.exists(os.path.join(folder_path, filename)):
                                            file_path = os.path.join(folder_path, filename)
                                            debug_info.write(f"Attempting to load original file: {file_path}")
                                        # Otherwise try to construct from frame number
                                        elif frame_number is not None:
                                            # Construct filename from frame number (assuming 7-digit format like 0000001.jpg)
                                            filename = f"{frame_number:07d}.jpg"
                                            file_path = os.path.join(folder_path, filename)
                                            debug_info.write(f"Attempting to load file based on frame number: {file_path}")
                                        else:
                                            debug_info.write(f"Cannot determine file path for frame ID: {frame_id}")
                                            cols[i % 3].error(f"No image data for frame ID: {frame_id}")
                                            continue
                                            
                                        if os.path.exists(file_path):
                                            orig_img = cv2.imread(file_path)
                                            if orig_img is not None:
                                                debug_info.write(f"Original file loaded successfully")
                                                
                                                # Apply stored annotations if requested
                                                if annotate_files:
                                                    debug_info.write(f"Applying stored annotations from database")
                                                    
                                                    # Create annotated version
                                                    annotated = orig_img.copy()
                                                    
                                                    # Add fence polygon
                                                    pts = np.array(fence_polygon, np.int32).reshape(-1,1,2)
                                                    cv2.polylines(annotated, [pts], True, (0,255,255), 3)
                                                    
                                                    # Add centroid label
                                                    M = cv2.moments(pts)
                                                    if M["m00"] != 0:
                                                        cx = int(M["m10"] / M["m00"])
                                                        cy = int(M["m01"] / M["m00"])
                                                    else:
                                                        cx, cy = 0, 0
                                                    cv2.putText(annotated, "Permissive Zone", (cx - 100, cy), 
                                                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                                    
                                                    # Add stored detections from database
                                                    # Detections have been removed, so we skip this part
                                                    # but still draw the annotated frame with just the fence polygon
                                                    
                                                    # Use the annotated frame
                                                    img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                                                    cols[i % 3].image(img_rgb, 
                                                                   caption=f"Annotated original: {os.path.basename(file_path)}", 
                                                                   use_column_width=True)
                                                else:
                                                    # Display original without annotations
                                                    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                                                    cols[i % 3].image(img_rgb, 
                                                                   caption=f"Original file: {os.path.basename(file_path)}", 
                                                                   use_column_width=True)
                                                    
                                                successful_frames += 1
                                            else:
                                                debug_info.write(f"Failed to read image from file: {file_path}")
                                                cols[i % 3].error(f"Cannot read image: {os.path.basename(file_path)}")
                                        else:
                                            debug_info.write(f"File does not exist: {file_path}")
                                            cols[i % 3].error(f"File not found: {os.path.basename(file_path)}")
                                        
                                        # Close the database session
                                        session.close()
                                    except Exception as file_err:
                                        debug_info.write(f"Failed to load original file: {str(file_err)}")
                                        cols[i % 3].error(f"Error loading file: {str(file_err)}")
                            except Exception as e:
                                debug_info.write(f"❌ Error displaying frame {frame_id}: {e}")
                                cols[i % 3].error(f"Error: {str(e)}")
                        
                        # Update status message
                        if successful_frames > 0:
                            status_placeholder.success(f"Successfully displayed {successful_frames} frames")
                        else:
                            status_placeholder.error("Could not display any frames. See debug info for details.")
            else:
                st.info("No matching results found")
        except Exception as e:
            st.error(f"Error querying database: {str(e)}")
            st.exception(e)

# Live monitoring tab
with tab1:
    if start:
        # Initialize database
        db = DroneSecurityDB(db_path=db_path)
        st.success(f"Database initialized at {db_path}")
        
        # Load models once
        yolo = load_yolo_model()
        proc, cap_model = load_captioning_model()

        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg'))])

        # Placeholders
        image_placeholder = st.empty()
        info_placeholder = st.empty()

        track_history = {}
        detection_times = {}
        prev_hash = None
        for idx, fname in enumerate(files):
            img_path = os.path.join(folder, fname)
            frame = cv2.imread(img_path)
            if frame is None: continue

            current_alerts.clear()
            telemetry = simulate_telemetry_data()

            # Prepare annotated image
            annotated = frame.copy()
            pts = np.array(fence_polygon, np.int32).reshape(-1,1,2)
            cv2.polylines(annotated, [pts], True, (0,255,255), 3)
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0  # default to 0 if centroid can't be calculated

            # Overlay "No Entry Zone" text at the centroid
            # Ensure that cx, cy are being passed as a valid tuple
            cv2.putText(annotated, "Permissive Zone", (cx - 100, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Detect & track
            annotated_frame, dets, counts = detect_and_track(annotated, yolo)

            # Caption
            if idx % 20 == 0:
                caption = generate_caption(annotated_frame, proc, cap_model)
                last_caption = caption
            else:
                caption = last_caption  

            if counts:
                count_str = ", ".join(f"{k}:{v}" for k, v in counts.items())
                caption += f" | {count_str}"
            # Apply rules
            rule_perimeter_intrusion(dets)
            #rule_time_restricted_entry(dets)
            #rule_crowd_density(dets)
            #rule_abandoned_object(track_history, detection_times, idx)
            #rule_loitering(track_history, idx)

            # Update history
            for d in dets:
                tid = d['id']
                track_history.setdefault(tid, []).append(d['centroid'])
                detection_times.setdefault((tid, tuple(d['centroid'])), idx)

            # Alerts
            if current_alerts:
                alert_type, alert_msg = current_alerts[0]
                if isinstance(alert_msg, dict) and 'intruder_ids' in alert_msg:
                    intruders_str = ', '.join(map(str, alert_msg['intruder_ids']))
                    st.warning(f"{alert_type}: by {intruders_str}")
                else:
                    st.warning(f"{alert_type}: {alert_msg}")
            else:
                alert_type, alert_msg = 'NA', ''


            # Serialize for consistent hashing
            dets_serialized = serialize_detections(dets)
            #current_hash = hash_state(dets_serialized, current_alerts)
            current_hash = hash_state(counts)
            # Only log and update UI if something changed
            if current_hash != prev_hash:
                # Update display
                image_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                info_placeholder.json({
                    'telemetry': telemetry,
                    'class_counts': counts,
                    'caption': caption,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'alert_type': alert_type,
                    'alert_message': str(alert_msg)
                })

                # Save to database
                frame_id = db.store_frame(
                    frame=annotated_frame,
                    frame_number=idx,
                    folder_path=folder,
                    filename=fname,
                    caption=caption,
                    telemetry=telemetry,
                    detections=dets,
                    alerts=current_alerts
                )
                
                if frame_id:
                    st.sidebar.success(f"Frame #{idx} stored in database (ID: {frame_id})")

                # Optional: Save log to file
                with open('event_log.txt', 'a') as f:
                    f.write(json.dumps({
                        'timestamp': datetime.now().isoformat(),
                        'alerts': current_alerts,
                        'detections': dets_serialized
                    }, default=lambda o: int(o) if isinstance(o, (np.integer,)) else str(o)) + '\n')

                # Update state
                prev_hash = current_hash

            # control framerate
            time.sleep(0.1)
