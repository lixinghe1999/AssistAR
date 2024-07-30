from small.class_mask import filter_dataset_category
from ultralytics import YOLO, YOLOWorld
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='yolov8x-worldv2.pt')
    parser.add_argument('--dataset', type=str, default='dataset/EgoObjects/EgoObjects.yaml') # yaml file for training, folder or jpg for test
    parser.add_argument('--model', type=str, default='yolov8-world')
    parser.add_argument('--mode', type=str, default='full-train')
    args = parser.parse_args()

    if args.model == 'yolov8':
        model = YOLO(args.ckpt)
    elif args.model == 'yolov8-world':
        model = YOLOWorld(args.ckpt)

    if args.mode == 'full-train':
        model.train(data=args.dataset, epochs=2, name=args.model)
        # metrics = model.val()  # no arguments needed, dataset and settings remembered
    else:
        metrics = model.val(data=args.dataset)