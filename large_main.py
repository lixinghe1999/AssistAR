import matplotlib.pyplot as plt
from large.qwen_vl import init_qwenvl, inference_qwenvl
def plot_bbox_image(image_path, output_image_path='output.png', ref=None, box=None):
    plt.figure()
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    box = box.replace('(', '').replace(')', '').split(',')
    box = [int(x)/1000 for x in box]
    img_height, img_width = img.shape[:2]
    box = [box[0] * img_width, box[1] * img_height, box[2] * img_width, box[3] * img_height]  # [x1, y1, x2, y2]
    print(box)

    plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor='red', linewidth=2))
    plt.text(box[0], box[1]-50, ref, fontsize=6, color='red', verticalalignment='top')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)


qwen_model, qwen_tokenizer = init_qwenvl()

image_path = 'dataset/example/bottles.jfif'
response = inference_qwenvl(qwen_model, qwen_tokenizer, image_path, detection_prompt='get the location of the red bottle', )
# response = '<ref> the handclap</ref><box>(537,508),(581,593)</box>'
# parse the response by <ref> and <box>
ref = response.split('<ref>')[1].split('</ref>')[0]
box = response.split('<box>')[1].split('</box>')[0]
print(ref, box)
plot_bbox_image(image_path, output_image_path='output.png', ref=ref, box=box)
# draw the bounding box and savefig
