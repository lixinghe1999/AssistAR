from .yolo_world import init_yoloworld, inference_yoloworld, parser_yoloworld
from .owl_vit import init_owl, inference_owl, parser_owl

def init_builder(model_name):
    if model_name.startswith('yolo'):
        return init_yoloworld
    elif model_name == 'owl':
        return init_owl
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
def inference_builder(model_name):
    if model_name.startswith('yolo'):
        return inference_yoloworld
    elif model_name == 'owl':
        return inference_owl
    else:
        raise ValueError(f"Unknown model: {model_name}")

def parser_builder(model_name):
    if model_name.startswith('yolo'):
        return parser_yoloworld
    elif model_name == 'owl':
        return parser_owl
    else:
        raise ValueError(f"Unknown model: {model_name}")