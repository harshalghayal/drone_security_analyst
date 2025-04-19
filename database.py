import os
import sqlite3
import base64
from datetime import datetime
import json
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, LargeBinary, DateTime, ForeignKey, Text, JSON, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import pandas as pd
import cv2
import numpy as np
import traceback
from PIL import Image
import io

# Create base class for declarative class definitions
Base = declarative_base()

class Frame(Base):
    """Table to store video frames and their metadata"""
    __tablename__ = 'frames'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    frame_data = Column(LargeBinary)  # Stores the compressed image data
    frame_number = Column(Integer)
    folder_path = Column(String)
    filename = Column(String)
    caption = Column(Text)
    telemetry = Column(JSON)
    alert_type = Column(String, nullable=True)  # Alert type field
    alert_message = Column(Text, nullable=True)  # Added alert message field

class Detection(Base):
    """Table to store object detections"""
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True)
    frame_id = Column(Integer, ForeignKey('frames.id'))
    object_class = Column(String)
    object_id = Column(Integer)
    confidence = Column(Float)
    x1 = Column(Integer)
    y1 = Column(Integer)
    x2 = Column(Integer)
    y2 = Column(Integer)
    centroid_x = Column(Integer)
    centroid_y = Column(Integer)

class Alert(Base):
    """Table to store security alerts"""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    frame_id = Column(Integer, ForeignKey('frames.id'))
    alert_type = Column(String)
    alert_message = Column(Text)
    timestamp = Column(DateTime, default=datetime.now)
    
class DroneSecurityDB:
    """Handler class for database operations"""
    
    def __init__(self, db_path='drone_security.db'):
        """Initialize the database connection"""
        self.db_path = db_path
        
        # Check if database file exists
        db_exists = os.path.exists(db_path)
        print(f"[DEBUG] Database file '{db_path}' exists: {db_exists}")
        
        # Create database connection
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Check for existing frames
        session = self.Session()
        try:
            frame_count = session.query(Frame).count()
            print(f"[DEBUG] Found {frame_count} existing frames in database")
            
            if frame_count > 0:
                # Get info about first and last frame
                first_frame = session.query(Frame).order_by(Frame.id).first()
                last_frame = session.query(Frame).order_by(Frame.id.desc()).first()
                print(f"[DEBUG] First frame ID: {first_frame.id}, Last frame ID: {last_frame.id}")
                print(f"[DEBUG] Sample frame attributes: filename={first_frame.filename}, " +
                      f"has_data={'Yes' if first_frame.frame_data else 'No'}")
        except Exception as e:
            print(f"[DEBUG] Error checking database: {e}")
        finally:
            session.close()
    
    def store_frame(self, frame, frame_number, folder_path, filename, caption, telemetry, detections, alerts):
        """Store a frame and all its associated data"""
        session = self.Session()
        try:
            # Validate input frame
            if frame is None or not isinstance(frame, np.ndarray):
                print(f"Error: Invalid frame data, cannot store frame #{frame_number}")
                return None
                
            # Ensure frame is in correct format (BGR)
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                print(f"Warning: Frame #{frame_number} has unusual shape: {frame.shape}")
                
            try:
                # Store the frame with JPEG compression to reduce size
                print(f"Storing frame #{frame_number} with shape {frame.shape}")
                
                # Convert frame to bytes with JPEG compression (smaller file size)
                success, encoded_img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not success:
                    print(f"Error encoding frame #{frame_number} with cv2.imencode")
                    return None
                    
                frame_data = encoded_img.tobytes()
                print(f"[DEBUG] Frame #{frame_number} encoded to {len(frame_data)} bytes")
                print(f"[DEBUG] First 20 bytes of frame data: {frame_data[:20]}")
                frame_shape = frame.shape
                
                # Store the shape and encoding info in telemetry
                if telemetry is None:
                    telemetry = {}
                telemetry['frame_shape'] = frame_shape
                telemetry['encoding'] = 'jpeg'
                
                print(f"Successfully prepared frame #{frame_number} for storage")
            except Exception as e:
                print(f"Error processing frame #{frame_number}: {e}")
                traceback.print_exc()
                return None
                
            # Get alert type and message if available
            alert_type = None
            alert_message = None
            if alerts and len(alerts) > 0:
                alert_type = alerts[0][0]  # Get the first alert type
                if len(alerts[0]) > 1:
                    # Convert alert message to string, handling both dict and primitive types
                    alert_msg = alerts[0][1]
                    if alert_msg is not None:
                        if isinstance(alert_msg, dict):
                            alert_message = json.dumps(alert_msg)
                        else:
                            alert_message = str(alert_msg)
                
            # Create Frame record with alert_type and alert_message
            frame_record = Frame(
                frame_data=frame_data,
                frame_number=frame_number,
                folder_path=folder_path,
                filename=filename,
                caption=caption,
                telemetry=telemetry,
                alert_type=alert_type,
                alert_message=alert_message
            )
            session.add(frame_record)
            session.flush()  # Flush to check if the data is stored before commit
            
            # Verify that frame_data was stored correctly
            print(f"[DEBUG] Verifying frame_data storage for frame #{frame_number}")
            stored_frame = session.query(Frame).get(frame_record.id)
            if stored_frame and stored_frame.frame_data:
                print(f"[DEBUG] Successfully stored frame_data in database: {len(stored_frame.frame_data)} bytes")
                if len(stored_frame.frame_data) != len(frame_data):
                    print(f"[WARNING] Stored frame_data size ({len(stored_frame.frame_data)}) doesn't match original ({len(frame_data)})")
            else:
                print(f"[ERROR] Frame_data not stored in database for frame #{frame_number}")
            
            session.commit()
            print(f"Frame #{frame_number} successfully stored with ID: {frame_record.id}")
            return frame_record.id
        except Exception as e:
            session.rollback()
            print(f"Error storing frame #{frame_number}: {e}")
            traceback.print_exc()
            return None
        finally:
            session.close()
    
    def get_frame_image(self, frame_id):
        """Retrieve a frame image by ID"""
        session = self.Session()
        try:
            print(f"[DEBUG] Attempting to retrieve frame ID: {frame_id}")
            frame_record = session.query(Frame).get(frame_id)
            
            if not frame_record:
                print(f"[ERROR] No record found for frame ID {frame_id}")
                return None
                
            if not frame_record.frame_data:
                print(f"[ERROR] Frame ID {frame_id} exists but frame_data is None or empty")
                if hasattr(frame_record, 'frame_data'):
                    print(f"[DEBUG] frame_data attribute exists but value is: {type(frame_record.frame_data)}")
                    if frame_record.frame_data:
                        print(f"[DEBUG] frame_data length: {len(frame_record.frame_data)} bytes")
                        print(f"[DEBUG] First 20 bytes: {frame_record.frame_data[:20]}")
                    else:
                        print(f"[DEBUG] frame_data is empty or None: {frame_record.frame_data}")
                else:
                    print(f"[DEBUG] frame_data attribute does not exist on record")
                
                # Print other record attributes to verify data is there
                print(f"[DEBUG] Other record attributes: id={frame_record.id}, frame_number={frame_record.frame_number}, filename={frame_record.filename}")
                print(f"[DEBUG] Record telemetry: {frame_record.telemetry}")
                print(f"[DEBUG] Caption: {frame_record.caption}")
                return None
            
            # If we got here, frame_data exists
            data = frame_record.frame_data
            print(f"[DEBUG] Retrieved frame_data of type {type(data)} with size {len(data)} bytes")
            print(f"[DEBUG] First 20 bytes of frame_data: {data[:20]}")

            # Check if telemetry indicates the encoding format
            encoding_format = None
            if frame_record.telemetry and 'encoding' in frame_record.telemetry:
                encoding_format = frame_record.telemetry['encoding']
                print(f"[DEBUG] Frame encoding format from telemetry: {encoding_format}")

            # 1) Try OpenCV imdecode (most reliable for our JPEG encoding)
            try:
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    print(f"[DEBUG] Successfully loaded via cv2.imdecode, shape: {img.shape}")
                    return img
                else:
                    print("[ERROR] cv2.imdecode returned None")
            except Exception as e:
                print(f"[ERROR] cv2.imdecode failed: {e}")

            # 2) Try PIL decode as fallback
            try:
                pil_img = Image.open(io.BytesIO(data))
                img = np.array(pil_img)
                # Convert RGB→BGR for OpenCV consistency if needed
                if img.ndim == 3 and img.shape[2] >= 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                print(f"[DEBUG] Loaded via PIL, shape: {img.shape}")
                return img
            except Exception as e:
                print(f"[ERROR] PIL decode failed: {e}")

            # 3) Fallback: raw‐bytes + telemetry shape
            if frame_record.telemetry and 'frame_shape' in frame_record.telemetry:
                shape = frame_record.telemetry['frame_shape']
                if isinstance(shape, list):
                    shape = tuple(shape)
                print(f"[DEBUG] Attempting raw reconstruction with shape {shape}")
                try:
                    img = np.frombuffer(data, dtype=np.uint8).reshape(shape)
                    print(f"[DEBUG] Reconstructed raw array, shape: {img.shape}")
                    return img
                except Exception as e:
                    print(f"[ERROR] Raw reconstruction failed: {e}")

            print(f"[ERROR] All decoding methods failed for frame ID {frame_id}")
            return None

        except Exception as e:
            print(f"[ERROR] Error retrieving frame ID {frame_id}: {e}")
            traceback.print_exc()
            return None
        finally:
            session.close()

    
    def query_frames_by_date(self, start_date, end_date):
        """Query frames by date range"""
        session = self.Session()
        try:
            frames = session.query(Frame).filter(
                Frame.timestamp >= start_date,
                Frame.timestamp <= end_date
            ).all()
            return frames
        finally:
            session.close()
    
    def query_frames_by_objects(self, object_classes):
        """Query frames containing specific object classes based on caption"""
        session = self.Session()
        try:
            # Now search in caption instead of detections
            query = session.query(Frame)
            filters = []
            for obj_class in object_classes:
                filters.append(Frame.caption.ilike(f'%{obj_class}%'))
            
            if filters:
                query = query.filter(or_(*filters))  # Using imported or_ directly
            
            frames = query.all()
            return frames
        finally:
            session.close()
    
    def query_frames_by_alerts(self, alert_types):
        """Query frames with specific alert types"""
        session = self.Session()
        try:
            # Now use alert_type column directly from Frame table
            frames = session.query(Frame).filter(
                Frame.alert_type.in_(alert_types)
            ).all()
            return frames
        finally:
            session.close()
            
    def get_all_detections_df(self):
        """Get all frames as a pandas DataFrame for easier querying"""
        session = self.Session()
        try:
            query = """
            SELECT 
                f.id as frame_id, 
                f.timestamp, 
                f.frame_number,
                f.filename, 
                f.caption,
                f.alert_type,
                f.alert_message
            FROM frames f
            """
            return pd.read_sql(query, self.engine)
        finally:
            session.close()
            
    def natural_language_query(self, query_text):
        """Simple natural language query implementation
        This can be extended with proper NLP later
        """
        # Very simple keyword matching for demonstration
        df = self.get_all_detections_df()
        
        # Process keywords based only on alert_type and caption fields
        keywords = {
            'alert': df['alert_type'].notna(),
            'intrusion': df['alert_type'].str.contains('Intrusion', na=False) if 'alert_type' in df.columns else False,
            'loitering': df['alert_type'].str.contains('Loitering', na=False) if 'alert_type' in df.columns else False,
            'abandoned': df['alert_type'].str.contains('Abandoned', na=False) if 'alert_type' in df.columns else False,
            # Search for objects in the caption instead
            'person': df['caption'].str.contains('person', case=False, na=False),
            'car': df['caption'].str.contains('car', case=False, na=False),
            'people': df['caption'].str.contains('people', case=False, na=False),
            'motor': df['caption'].str.contains('motor', case=False, na=False),
            'pedestrian': df['caption'].str.contains('pedestrian', case=False, na=False),
        }
        
        # Apply filters based on query text
        filtered_df = df.copy()
        for keyword, condition in keywords.items():
            if keyword.lower() in query_text.lower():
                filtered_df = filtered_df[condition]
        
        return filtered_df