from ultralytics import YOLOWorld
def init_yoloworld(name="yolov8x-world-ego"):
    # Initialize a YOLO-World model
    model = YOLOWorld(name)  # or select yolov8m/l-world.pt for different sizes
    return model

def inference_yoloworld(model, image_file, prompt=None):
    if prompt is not None:
        # model.set_classes(["person", "bus"])
        # pass
        if type(prompt) == str:
            prompt = [prompt]
        model.set_classes(prompt)
    # Execute inference with the YOLOv8s-world model on the specified image
    results = model.predict(image_file)
    return results[0]

def parser_yoloworld(result):
    '''
    Unified format [x1, y1, x2, y2] in [0, 1]
    '''
    result = result.cpu()
    names = result.names
    orig_shape = result.orig_shape
    box = (result.boxes.xyxy.numpy().tolist())
    box = [[x[0]/orig_shape[1], x[1]/orig_shape[0], x[2]/orig_shape[1], x[3]/orig_shape[0]] for x in box]

    ref = (result.boxes.cls.numpy().tolist())
    ref = [names[int(x)] for x in ref]
    # result.save('output.png')
    return ref, box

