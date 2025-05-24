import sys
sys.path.append('')

from ultralytics import YOLO
from ultralytics import SETTINGS
model = YOLO('hrnet.yaml', task='pose')
SETTINGS['tensorboard'] = True
model.train(data='hand-keypoints.yaml', cfg="hrnet-example.yaml", epochs=120, plots=True)
SETTINGS['tensorboard'] = False