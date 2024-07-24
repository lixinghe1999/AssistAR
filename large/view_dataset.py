import numpy as np
def plot_bbox(image_path, label_path, auto_label_path, name='output.png'):
    import matplotlib.pyplot as plt
    img = plt.imread(image_path)
    width, height = img.shape[1], img.shape[0]
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            x1, y1, w, h = map(float, line[1:])
            x1, y1 = x1 - w/2, y1 - h/2
            x1, y1, w, h = x1 * width, y1 * height, w * width, h * height
            plt.gca().add_patch(plt.Rectangle((x1, y1), w, h, fill=False, edgecolor='r', linewidth=4))

    with open(auto_label_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            x1, y1, w, h = map(float, line[1:])
            x1, y1 = x1 - w/2, y1 - h/2
            x1, y1, w, h = x1 * width, y1 * height, w * width, h * height
            plt.gca().add_patch(plt.Rectangle((x1, y1), w, h, fill=False, edgecolor='g', linewidth=2))

    plt.imshow(img)
    plt.axis('off')
    plt.savefig(name, transparent=True, bbox_inches='tight', pad_inches=0)

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    box1 (numpy.ndarray): An array of 4 elements (x_center, y_center, width, height) describing the first bounding box.
    box2 (numpy.ndarray): An array of 4 elements (x_center, y_center, width, height) describing the second bounding box.
    
    Returns:
    float: The Intersection over Union of the two boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate the coordinates of the intersection rectangle
    xA = max(x1 - w1 / 2, x2 - w2 / 2)
    yA = max(y1 - h1 / 2, y2 - h2 / 2)
    xB = min(x1 + w1 / 2, x2 + w2 / 2)
    yB = min(y1 + h1 / 2, y2 + h2 / 2)
    
    # Calculate the area of intersection rectangle
    intersection = max(0, xB - xA) * max(0, yB - yA)
    
    # Calculate the area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # Calculate the Intersection over Union by taking the intersection
    # area and dividing it by the sum of prediction and ground-truth
    # areas - the intersection area
    iou = intersection / float(box1_area + box2_area - intersection)
    
    return iou

def calculate_map50(gt_file, pred_file):
    """
    Calculate the mean Average Precision (mAP) given ground truth and prediction bounding box files.
    
    Parameters:
    gt_file (str): Path to the ground truth bounding box file.
    pred_file (str): Path to the prediction bounding box file.
    
    Returns:
    float: The mean Average Precision (mAP).
    """
    # Load ground truth and prediction data
    with open(gt_file, 'r') as f:
        gt_data = np.array([list(map(float, line.strip().split())) for line in f])
    with open(pred_file, 'r') as f: 
        pred_data = np.array([list(map(float, line.strip().split())) for line in f])
    if len(gt_data) == 0 or len(pred_data) == 0:
        return 0
    # Sort the predictions by confidence score in descending order
    # pred_data = pred_data[np.argsort(-pred_data[:, 0])]
    
    true_positives = np.zeros(len(pred_data))
    false_positives = np.zeros(len(pred_data))
    
    # Iterate through the predictions
    for i, pred_box in enumerate(pred_data):
        # Find the best matching ground truth box
        max_iou = 0
        best_gt_box = None
        for gt_box in gt_data:
            iou = calculate_iou(pred_box[1:], gt_box[1:])
            if iou > max_iou:
                max_iou = iou
                best_gt_box = gt_box
        # If the best matching ground truth box has an IoU > 0.5, it's a true positive
        if max_iou > 0.2:
            true_positives[i] = 1
        else:
            false_positives[i] = 1
    # Calculate precision-recall curve
    precision = true_positives.sum() / (true_positives.sum() + false_positives.sum())
    return precision

if __name__ == '__main__':
    import os
    dataset_dir = 'dataset/EgoObjects_mini'

    auto_label_list = []

    # auto_label_dir = os.path.join(dataset_dir, 'groundingdino_output')
    # auto_label_list += [(os.path.join(auto_label_dir, 'train/labels'), os.listdir(os.path.join(auto_label_dir, 'train/labels')))]
    # auto_label_list += [(os.path.join(auto_label_dir, 'valid/labels'), os.listdir(os.path.join(auto_label_dir, 'valid/labels')))]

    auto_label_dir = os.path.join(dataset_dir, 'qwenvl_output')
    auto_label_list += [(os.path.join(auto_label_dir, 'labels'), os.listdir(os.path.join(auto_label_dir, 'labels')))]

    image_dir = os.path.join(dataset_dir, 'images')
    label_dir = os.path.join(dataset_dir, 'labels')
    mAPs = []
    for i, label in enumerate(os.listdir(label_dir)):
        image = label.replace('txt', 'jpg')
        image_path = os.path.join(image_dir, image)
        label_path = os.path.join(label_dir, label)

        for data in auto_label_list:
            tmp_dir, tmp_list = data
            if label in tmp_list:
                auto_label_path = os.path.join(tmp_dir, label)

        if i == 45:
            plot_bbox(image_path, label_path, auto_label_path, name='output.png')

        mAP = calculate_map50(label_path, auto_label_path)
        mAPs.append(mAP)
    print('mAP:', np.mean(mAPs))
        #break