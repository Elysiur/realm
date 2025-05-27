import sys
sys.path.append('')

from ultralytics import YOLO
from ultralytics import SETTINGS
model = YOLO('hrnet.yaml', task='pose')
SETTINGS['tensorboard'] = True
model.train(data='hand-keypoints.yaml', cfg="hrnet-example.yaml", plots=True)
SETTINGS['tensorboard'] = False
# model.predict('hello.jpg', save=True,visualize=True)
# model.val()
# model.export(format='onnx')
# model.export(format='engine', device="cuda")
# model.export(format='trt')
# model.export(format='openvino')