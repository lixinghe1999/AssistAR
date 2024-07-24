'''
Given a yolo-dataset, this script will generate a temporary dataset with only the selected classes.
'''

import os
from tqdm import tqdm
import json

def filter_class(txt, tmp_txt, keep_class):
    with open(txt, 'r') as file:
        lines = file.readlines()
    exist = False
    new_lines = []
    for line in lines:
        line = line.strip().split()
        if int(line[0]) in keep_class:
            exist = True
            line[0] = str(keep_class.index(int(line[0])))
            new_lines.append(line)
    if exist:
        with open(tmp_txt, 'w') as file:
            for line in new_lines:
                file.write(' '.join(line) + '\n')
    return exist

def filter_dataset(keep_class=[0]):
    dataset_name = 'EgoObjects_mini'
    dataset_dir = os.path.join('dataset', dataset_name)

    labels_dir = os.path.join(dataset_dir, 'labels')
    images_dir = os.path.join(dataset_dir, 'images')
    # remove and create new labels and images dir
    os.system(f'rm -rf {labels_dir}')
    os.system(f'rm -rf {images_dir}')
    os.makedirs(labels_dir)
    os.makedirs(images_dir)

    raw_yaml = os.path.join(dataset_dir, 'raw_EgoObjects.yaml')
    import yaml
    with open(raw_yaml, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    # only keep the class
    data['nc'] = len(keep_class)
    data['names'] = {i:data['names'][keep_class[i]] for i in range(len(keep_class))}
    # save the new yaml
    tmp_yaml = raw_yaml.replace('raw_', '')
    with open(tmp_yaml, 'w') as file:
        yaml.dump(data, file)

    train_txt = os.path.join(dataset_dir, 'raw_train.txt')
    val_txt = os.path.join(dataset_dir, 'raw_val.txt')

    keep_num = []
    for split in [train_txt, val_txt]:
        tmp_txt = []
        with open(split, 'r') as file:
            lines = file.readlines()
        for line in lines:
            label = line.strip().replace('images', 'raw_labels').replace('jpg', 'txt')
            label_path = os.path.join(dataset_dir, label)
            target_label_path = label_path.replace('raw_labels', 'labels')
            exist = filter_class(label_path, target_label_path, keep_class)
            if exist:
                tmp_txt.append(line)
                raw_image_path = line.strip().replace('images', 'raw_images')
                image_path = line.strip()
                os.system(f'cp {dataset_dir}/{raw_image_path} {dataset_dir}/{image_path}')
        with open(split.replace('raw_', ''), 'w') as file:
            for line in tmp_txt:
                file.write(line)
                # copy 
        print('Done filtering', split, f'keep files {len(tmp_txt)}/{len(lines)}')
        keep_num.append(len(tmp_txt))
    return keep_num
 
if __name__ == '__main__':
    filter_dataset([1])


    

    # foplit in [train_txt, val_txt]:
    #     data_dir = os.path.join(dataset_dir, 'labels', split)
    #     class_num = 80
    #     class_files = {int(i): [] for i in range(class_num)}
    #     for f in tqdm(os.listdir(data_dir)):
    #         class_file = {}
    #         with open(os.path.join(data_dir, f), 'r') as file:
    #             lines = file.readlines()
    #         for line in lines:
    #             line = line.strip().split()
    #             class_file[int(line[0])] = f
    #         for k in class_file.keys():
    #             class_files[k].append(class_file[k])
    #     json.dump(class_files, open(r sdataset_dir + 'class_{}.json'.format(split), 'w'), indent=4)
    
    # generate_temp_list([0], out_dir='./coco')