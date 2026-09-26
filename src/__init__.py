"""Inference components: force local-only Ultralytics initialization."""
import os

os.environ['YOLO_OFFLINE'] = 'true'
os.environ['YOLO_AUTOINSTALL'] = 'false'
