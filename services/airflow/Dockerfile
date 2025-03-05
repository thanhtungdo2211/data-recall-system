# base Airflow image
FROM apache/airflow:2.8.3-python3.9

ARG AIRFLOW_HOME=${AIRFLOW_HOME}

USER root

# Install jre for spark plugins and dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y \
    default-jre-headless \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Spark dependencies
COPY ./requirements.txt ${AIRFLOW_HOME}/requirements.txt
RUN pip install --no-cache-dir -r ${AIRFLOW_HOME}/requirements.txt

COPY ./dags/ ${AIRFLOW_HOME}/dags/