import sys
sys.path.append('')

from ultralytics import YOLO

model = YOLO("yolo12n.yaml",task='detect')

model.train(data="coco.yaml",epochs=50,cfg='yolo-example.yaml')