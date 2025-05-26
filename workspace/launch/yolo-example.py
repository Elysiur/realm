import sys
sys.path.append('')

from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt",task='pose')

# model.train(data="coco.yaml",epochs=50,cfg='yolo-example.yaml')

model.predict('bus.jpg', save=True,visualize=True)