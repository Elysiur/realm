import sys
sys.path.append('')

from ultralytics import HRNet
from ultralytics import SETTINGS

model = HRNet('hrnet.engine', task='pose')

# SETTINGS['tensorboard'] = True
# model.train(data='hand-keypoints.yaml', cfg="hrnet-example.yaml", plots=True)
# SETTINGS['tensorboard'] = False

# model.predict(source="workspace/assets/hand.mp4",save=True)
# model.val()

# model.export(format='onnx')
# model.export(format='engine', device="cuda")
# model.export(format='openvino')