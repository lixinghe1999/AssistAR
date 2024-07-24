from small.class_mask import filter_dataset
from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8x')
    parser.add_argument('--mode', type=str, default='full-train')
    args = parser.parse_args()

    if args.mode == 'full-train':
        # model = YOLO(args.model)
        # model.train(data="dataset/EgoObjects/EgoObjects.yaml", epochs=20, freeze=10, name='full-train')


        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        model = YOLO('runs/detect/full-train/weights/best.pt')
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # a list contains map50-95 of each category
    elif args.mode == 'mini-train':
        model = YOLO('yolov8n.pt')
        for i in range(2):
            keep_num = filter_dataset([i])
            if keep_num[0] == 0 or keep_num[1] == 0: # either train or val = 0
                print('No data for training')
            else:
                model.train(data="dataset/EgoObjects_mini/EgoObjects.yaml", epochs=100, device=1, name=i, freeze=10)  # train the model


# model = init_adaptive_yolo("yolov8m")
# model.save_head("head1")
# metrics = model.val()
# model.export(format="onnx", int8=True, optimize=True)
