import sys
sys.path.append('')

from ultralytics import YOLO
from ultralytics import SETTINGS
model = YOLO('hrnet.yaml')
SETTINGS['tensorboard'] = True
model.train(data='coco.yaml', cfg="hrnet-example.yaml", epochs=60)
SETTINGS['tensorboard'] = False