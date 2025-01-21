import matplotlib.pyplot as plt
from model.yolo_world import init_yoloworld, inference_yoloworld, parser_yoloworld
from model import init_builder, inference_builder, parser_builder
from model.api import request_detection, parser_api
import os
from tqdm import tqdm
from utils.class_mask import filter_dataset_scenario
from model.evaluate import evaluate
import time
def plot_bbox_image(image_path, output_image_path='output.png', refs=None, boxs=None):
    plt.figure()
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    img_height, img_width = img.shape[:2]
    for ref, box in zip(refs, boxs):
        box = [box[0] * img_width, box[1] * img_height, box[2] * img_width, box[3] * img_height]  # [x1, y1, x2, y2]
        plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor='red', linewidth=2))
        plt.text(box[0], box[1]+50, ref, fontsize=12, color='red', verticalalignment='top')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return img_width, img_height
def bbox_to_yolo(output_label_path, ref, bbox, class_map):
    class_map = {v: k for k, v in class_map.items()}
    with open(output_label_path, 'w') as f:
        for r, box in zip(ref, bbox):
            r = class_map[r]
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            f.write(f'{r} {x_center} {y_center} {width} {height}\n')
    

def inference_scenario(keep_scenario, model_name):
    init_func = init_builder(model_name)
    inference_func = inference_builder(model_name)
    parser_func = parser_builder(model_name)

    class_map = filter_dataset_scenario(keep_scenario)
    dataset_dir = 'dataset/EgoObjects/mini_scenario/{}'.format('_'.join([str(i) for i in keep_scenario]))
    image_dir = os.path.join(dataset_dir, 'images')
    output_dir = os.path.join(dataset_dir, f'{model_name}')
    os.makedirs(output_dir, exist_ok=True)
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    if os.path.exists(output_image_dir):
        os.system(f'rm -rf {output_image_dir}')
    if os.path.exists(output_label_dir):
        os.system(f'rm -rf {output_label_dir}')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    prompt = list(class_map.values())
    images = os.listdir(image_dir)
    images.sort()
    model = init_func(model_name)
    for i, image in enumerate(tqdm(images)):
        image_path = os.path.join(image_dir, image)

        response = inference_func(model, image_path, prompt)
        ref, box = parser_func(response)

        output_image_path = os.path.join(output_image_dir, image)
        output_label_path = os.path.join(output_label_dir, image.replace('.jpg', '.txt'))
        if len(box) > 0:
            plot_bbox_image(image_path, output_image_path, ref, box)
            bbox_to_yolo(output_label_path, ref, box, class_map)
        else:
            # save the original image without any bounding box
            os.system(f'cp {image_path} {output_image_path}')
            with open(output_label_path, 'w') as f:
                f.write('')
        # if i == 10:
        #     break
    evaluate(dataset_dir, output_dir)

def inference_api(keep_scenario):
    class_map = filter_dataset_scenario(keep_scenario)
    dataset_dir = 'dataset/EgoObjects/mini_scenario/{}'.format('_'.join([str(i) for i in keep_scenario]))
    image_dir = os.path.join(dataset_dir, 'images')
    output_dir = os.path.join(dataset_dir,  'api')
    os.makedirs(output_dir, exist_ok=True)
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    if os.path.exists(output_image_dir):
        os.system(f'rm -rf {output_image_dir}')
    if os.path.exists(output_label_dir):
        os.system(f'rm -rf {output_label_dir}')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    prompt = list(class_map.values())
    images = os.listdir(image_dir)
    images.sort()
    for i, image in enumerate(tqdm(images)):
        image_path = os.path.join(image_dir, image)
        response = request_detection(image_path, '.'.join(prompt))
        ref, box = parser_api(response, image_path)
        print(ref, box)
        output_image_path = os.path.join(output_image_dir, image)
        output_label_path = os.path.join(output_label_dir, image.replace('.jpg', '.txt'))
        if len(box) > 0:
            plot_bbox_image(image_path, output_image_path, ref, box)
            bbox_to_yolo(output_label_path, ref, box, class_map)
        else:
            # save the original image without any bounding box
            os.system(f'cp {image_path} {output_image_path}')
            with open(output_label_path, 'w') as f:
                f.write('')
        # if i == 10:
        #     break
        time.sleep(10)
    evaluate(dataset_dir, output_dir)


if __name__ == '__main__':
    keep_scenario = [1]
    # model_names = ['yolov8x-worldv2', 'yolov8s-worldv2']
    model_names = ['yolov8x-worldv2']
    # model_names = ['owl']
    for model_name in model_names:
        inference_scenario(keep_scenario, model_name)

    # inference_api(keep_scenario)