import numpy as np
from collections import defaultdict

'''
Given the ground truth and prediction bounding box files, calculate the mean Average Precision (mAP) at IoU=0.5.
'''
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
    Calculate the Mean Average Precision at 50% Intersection over Union (mAP50) given ground truth and predicted bounding box files.
    
    Parameters:
    gt_file (str): Path to the ground truth bounding box file.
    pred_file (str): Path to the predicted bounding box file.
    
    Returns:
    float: The mAP50 score.
    """
    # Load ground truth and predicted bounding boxes
    gt_boxes = np.loadtxt(gt_file, delimiter=' ')
    pred_boxes = np.loadtxt(pred_file, delimiter=' ')
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0
    if len(gt_boxes.shape) == 1:
        gt_boxes = np.expand_dims(gt_boxes, axis=0)
    if len(pred_boxes.shape) == 1:
        pred_boxes = np.expand_dims(pred_boxes, axis=0)

   # Group ground truth boxes by class
    gt_by_class = defaultdict(list)
    for box in gt_boxes:
        class_idx, x, y, w, h = box
        gt_by_class[int(class_idx)].append(np.array([x, y, w, h]))
    
    # Calculate precision-recall curve for each class
    aps = []
    for class_idx in gt_by_class:
        # Get all predicted boxes for the current class
        class_pred_boxes = pred_boxes[pred_boxes[:, 0] == class_idx]
        
        # Initialize precision and recall
        precision = []
        recall = []
        tp, fp = 0, 0
        
        # Iterate through predicted boxes
        for box in class_pred_boxes:
            # Calculate IoU with ground truth boxes
            max_iou = 0
            for gt_box in gt_by_class[class_idx]:
                iou_val = calculate_iou(box[1:], gt_box)
                max_iou = max(max_iou, iou_val)
            
            # Update precision and recall
            if max_iou >= 0.5:
                tp += 1
            else:
                fp += 1
            precision.append(tp / (tp + fp))
            recall.append(tp / len(gt_by_class[class_idx]))
        
        # Calculate average precision
        ap = 0
        for i in range(len(precision)):
            if i == 0 or precision[i] != precision[i-1]:
                ap += precision[i] * (recall[i] - (0 if i == 0 else recall[i-1]))
        aps.append(ap)
    
    # Calculate mAP50
    mAP50 = np.mean(aps)
    return mAP50
def evaluate(dataset_dir, auto_label_dir):
    import os

    gt_label = os.path.join(dataset_dir, 'labels')
    auto_label = os.path.join(auto_label_dir, 'labels')
    mAPs = []
    for i, label in enumerate(os.listdir(auto_label)):
        auto_label_path = os.path.join(auto_label, label)
        gt_label_path = os.path.join(gt_label, label)
        mAP = calculate_map50(gt_label_path, auto_label_path)
        mAPs.append(mAP)
    mAP = np.mean(mAPs)
    print('mAP:', mAP)
    return mAP
if __name__ == '__main__':
    evaluate('dataset/EgoObjects/mini_scenario/0/', 'dataset/EgoObjects/mini_scenario/0/yolov8x-world-ego/')