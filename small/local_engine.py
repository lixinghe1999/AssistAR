from ultralytics import YOLO
from small.adaptive_yolo import Adaptive_YOLO

def init_yolo(model_name="yolov8n"):
    # Load a model
    model = YOLO(f"{model_name}.yaml")  # build a new model from scratch
    model = YOLO(f"{model_name}.pt")  # load a pretrained model (recommended for training)
    return model

def init_adaptive_yolo(model_name="yolov8n"):
    # Load a model
    model = Adaptive_YOLO(f"{model_name}.yaml")  # build a new model from scratch
    model = Adaptive_YOLO(f"{model_name}.pt")  # load a pretrained model (recommended for training)
    return model