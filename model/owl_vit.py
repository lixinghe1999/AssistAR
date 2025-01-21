import requests
from PIL import Image
import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection

def init_owl(name):
    assert name == "owl"
    # model_name = "owlv2-large-patch14" # not possible on 3090
    model_name = "owlv2-base-patch16-ensemble"
    processor = Owlv2Processor.from_pretrained("google/{}".format(model_name))
    model = Owlv2ForObjectDetection.from_pretrained("google/{}".format(model_name)).to('cuda')
    return processor, model
def inference_owl(model, image_file, prompt=None):
    processor, model = model

    image = Image.open(image_file)
    texts = [[f"a photo of a {p}" for p in prompt]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.3)
   
    return target_sizes[0], results[0], prompt

def parser_owl(output):
    target_size, results, prompt = output
    boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
    ref = []; norm_boxes = []
    for box, score, label in zip(boxes, scores, labels):
        box = box.tolist()
        box = [box[0]/target_size[1], box[1]/target_size[0], box[2]/target_size[1], box[3]/target_size[0]]
        norm_boxes.append([b.item() for b in box])
        ref += [prompt[label]]
    return ref, norm_boxes
if __name__ == "__main__":
    model = init_owl('owl')
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_file = requests.get(url, stream=True).raw
    output = inference_owl(model, image_file, ['cat', 'dog'])
    ref, norm_boxes = parser_owl(output)
    print(ref, norm_boxes)