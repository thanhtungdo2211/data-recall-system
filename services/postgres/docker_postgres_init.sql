-- Create schemas if not exists
CREATE SCHEMA IF NOT EXISTS public;

-- Extension uuid-ossp to generate UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop tables if exists
DROP TABLE IF EXISTS detections CASCADE;
DROP TABLE IF EXISTS detection_events CASCADE;

-- detection_events table
CREATE TABLE detection_events (
    id SERIAL PRIMARY KEY,
    camera_id VARCHAR(255),
    timestamp TIMESTAMP WITHOUT TIME ZONE,
    detection_count INTEGER,
    bucket_name VARCHAR(255),
    image_url VARCHAR(255),
    validated BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);


-- Create index for camera_id and timestamp
CREATE INDEX idx_detection_events_camera_id ON detection_events(camera_id);
CREATE INDEX idx_detection_events_timestamp ON detection_events(timestamp);

-- detections table
CREATE TABLE detections (
    id SERIAL PRIMARY KEY,
    event_id INTEGER REFERENCES detection_events(id) ON DELETE CASCADE,
    class_id INTEGER,
    class_name VARCHAR(255),
    confidence FLOAT,
    box_x1 FLOAT,
    box_y1 FLOAT,
    box_x2 FLOAT,
    box_y2 FLOAT,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for event_id and class_name
CREATE INDEX idx_detections_event_id ON detections(event_id);
CREATE INDEX idx_detections_class_name ON detections(class_name);

-- Function update_updated_at_column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger update_updated_at_column
CREATE TRIGGER update_detection_events_updated_at
BEFORE UPDATE ON detection_events
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_detections_updated_at
BEFORE UPDATE ON detections
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Add sample data
-- INSERT INTO detection_events (camera_id, timestamp, detection_count, bucket_name, image_url)
-- VALUES ('camera-demo', CURRENT_TIMESTAMP, 1, 'detection-frames', 'frames/demo/original/frame_001.jpg');

-- Configurations for user postgres
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO postgres;