import matplotlib.pyplot as plt
from large.qwen_vl import init_qwenvl, inference_qwenvl
import os
from tqdm import tqdm
def plot_bbox_image(image_path, output_image_path='output.png', ref=None, box=None):
    plt.figure()
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    box = [x/1000 for x in box]
    img_height, img_width = img.shape[:2]
    box = [box[0] * img_width, box[1] * img_height, box[2] * img_width, box[3] * img_height]  # [x1, y1, x2, y2]

    plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor='red', linewidth=2))
    plt.text(box[0], box[1]-50, ref, fontsize=6, color='red', verticalalignment='top')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return box, img_width, img_height
def bbox_to_yolo(output_label_path, bbox, img_width, img_height):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    x_center, y_center, width, height = x_center / img_width, y_center / img_height, width / img_width, height / img_height
    with open(output_label_path, 'w') as f:
        f.write(f'0 {x_center} {y_center} {width} {height}')
    


dataset_dir = 'dataset/EgoObjects_mini'
image_dir = os.path.join(dataset_dir, 'images')
output_dir = os.path.join(dataset_dir, 'qwenvl_output')
query_object = 'adhesive tape'
images = os.listdir(image_dir)
qwen_model, qwen_tokenizer = init_qwenvl('cuda:1')

for i, image in enumerate(tqdm(images)):
    image_path = os.path.join(image_dir, image)
    response = inference_qwenvl(qwen_model, qwen_tokenizer, image_path, detection_prompt='Can you find {}'.format(query_object), )
    output_image_path = os.path.join(output_dir, 'images', image)
    output_label_path = os.path.join(output_dir, 'labels', image.replace('.jpg', '.txt'))
    if '<ref>' not in response:
        with open(output_label_path, 'w') as f:
            f.write('')
        print('No object detected')
        continue
    else:
        print(response)
        ref = response.split('<ref>')[1].split('</ref>')[0]
        box = response.split('<box>')[1].split('</box>')[0]
        box = box.replace('(', '').replace(')', '').split(',')
        box = [int(x) for x in box]
        box, img_width, img_height = plot_bbox_image(image_path, output_image_path, ref=query_object, box=box)
        print(box)
        bbox_to_yolo(output_label_path, box ,img_width, img_height)
    break
# image_path = 'dataset/example/0001.jpg'
# response = inference_qwenvl(qwen_model, qwen_tokenizer, image_path, detection_prompt='get the location of the {}'.format(query_object), )
# # response = '<ref> the handclap</ref><box>(537,508),(581,593)</box>'
# # parse the response by <ref> and <box>
# ref = response.split('<ref>')[1].split('</ref>')[0]
# box = response.split('<box>')[1].split('</box>')[0]
# print(ref, box)
# plot_bbox_image(image_path, output_image_path='dataset/example/output.png', ref=ref, box=box)
# # draw the bounding box and savefig
