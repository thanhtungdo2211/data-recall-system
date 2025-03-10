from datetime import datetime

import sqlalchemy
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base

Base =  declarative_base()

class DetectionEvents(Base):
    __tablename__ = 'detection_events'  # Removed extra space
    id = Column(Integer, primary_key=True)
    camera_id = Column(String)
    timestamp = Column(DateTime)
    detection_count = Column(Integer)
    bucket_name = Column(String)
    image_url = Column(String)
    validated = Column(sqlalchemy.Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Define relationship to Detections
    detections = sqlalchemy.orm.relationship("Detections", back_populates="event")
    
class Detections(Base):
    __tablename__ = 'detections'
    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, sqlalchemy.ForeignKey('detection_events.id'))
    class_id = Column(Integer)
    class_name = Column(String)
    confidence = Column(sqlalchemy.Float)  # Changed to Float
    box_x1 = Column(sqlalchemy.Float)      # Changed to Float
    box_y1 = Column(sqlalchemy.Float)      # Changed to Float
    box_x2 = Column(sqlalchemy.Float)      # Changed to Float
    box_y2 = Column(sqlalchemy.Float)      # Changed to Float
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Define relationship to DetectionEvents
    event = sqlalchemy.orm.relationship("DetectionEvents", back_populates="detections")
    
    
    