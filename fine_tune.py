from ultralytics import YOLOWorld
from utils.class_mask import filter_dataset_category, filter_dataset_scenario
model = YOLOWorld("yolov8s-worldv2")
model = YOLOWorld("runs/detect/mini-train/weights/best.pt")
# class_map = filter_dataset_scenario(keep_scenario=[0])
# model.train(data='dataset/EgoObjects/mini_scenario/0/EgoObjects.yaml', epochs=50, name="mini-train", batch=8)
model.val(data='dataset/EgoObjects/mini_scenario/0/EgoObjects.yaml')
