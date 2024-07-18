from ultralytics import YOLO

class Adaptive_YOLO(YOLO):
    '''
    Inherit from YOLOv8
    '''
    def __init__(self, model_name="yolov8n.pt", task=None, verbose=False):
        super().__init__(model_name, task, verbose)
        self.head_zoo = []

    def save_head(self, head_name):
        new_head = {}
        new_head['head_name'] = head_name
        # split the backbone and head before layer 10
        head = self.ckpt['model'].model[22:]
        new_head['head'] = head
        # log the storage size of total parameters in the head
        new_head['head_size'] = sum([p.numel() for p in head.parameters()]) * 4 / 1024 / 1024
        print(f"Head size: {new_head['head_size']} MB")
        self.head_zoo.append(new_head)