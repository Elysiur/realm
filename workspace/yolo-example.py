import sys
sys.path.append('')

from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.predict(save=True)