import logging
import os
import sys
sys.insert(1, '.')
sys.insert(1, 'services/airflow/dags')  

import sqlalchemy
from sqlalchemy.orm import Session
from sqlalchemy import text, query
import load_env
from datetime import datetime
import json
from dotenv import load_dotenv

from tasks.db_utils import DetectionEvents, Detections, Base

load_dotenv()

POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'postgres')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')


engine = sqlalchemy.create_engine(
            f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        )
session = Session(engine)
# Create DetectionEvents record


