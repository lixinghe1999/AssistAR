from small.local_engine import init_yolo, init_adaptive_yolo
from small.class_mask import filter_dataset
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8x')
    parser.add_argument('--mode', type=str, default='full-train')
    args = parser.parse_args()

    if args.mode == 'full-train':
        model = init_yolo(args.model)
        model.train(data="dataset/EgoObjects/EgoObjects.yaml", epochs=20, batch=32, freeze=10)
    elif args.mode == 'mini-train':
        model = init_yolo("yolov8n")
        for i in range(10):
            keep_num = filter_dataset([i])
            if keep_num[0] == 0 or keep_num[1] == 0: # either train or val = 0
                print('No data for training')
            else:
                model.train(data="EgoObjects/EgoObjects_tmp.yaml", epochs=10, batch=32, device=1, name=i, freeze=20)  # train the model


# model = init_adaptive_yolo("yolov8m")
# model.save_head("head1")
# metrics = model.val()
# model.export(format="onnx", int8=True, optimize=True)
