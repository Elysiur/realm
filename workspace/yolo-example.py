import sys
sys.path.append('')

from ultralytics import YOLO
model = YOLO('yolo11n.yaml',task='detect')
model.train(data='coco128.yaml', epochs=50)
model.val()