import sys
sys.path.append('')

from ultralytics.data.converter import convert_coco
convert_coco(labels_dir="your/coco/datasets/folder", save_dir="your/coco/datasets/folder/to/save")