'''
Given a yolo-dataset, this script will generate a temporary dataset with only the selected classes.
'''

import os
from tqdm import tqdm
import yaml
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

def filter_dataset_category(keep_class=[0]):
    dataset_dir = 'dataset/EgoObjects/mini_category/' + '_'.join([str(i) for i in keep_class])
    orig_dataset_dir = 'dataset/EgoObjects'

    labels_dir = os.path.join(dataset_dir, 'labels')
    images_dir = os.path.join(dataset_dir, 'images')
    # remove and create new labels and images dir
    os.system(f'rm -rf {labels_dir}')
    os.system(f'rm -rf {images_dir}')
    os.makedirs(labels_dir)
    os.makedirs(images_dir)

    orig_yaml = os.path.join(orig_dataset_dir, 'EgoObjects.yaml')
    import yaml
    with open(orig_yaml, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    # only keep the class
    data['nc'] = len(keep_class)
    data['names'] = {i:data['names'][keep_class[i]] for i in range(len(keep_class))}
    print('Keep classes:', data['names'])
    # save the new yaml
    with open(os.path.join(dataset_dir, 'EgoObjects.yaml'), 'w') as file:
        yaml.dump(data, file)
    keep_num = []
    for split in ['train.txt', 'val.txt']:
        split_file = os.path.join(orig_dataset_dir, split)
        tmp_txt = []
        with open(split_file, 'r') as file:
            lines = file.readlines()
        for line in lines:
            label = line.strip().replace('images', 'labels').replace('jpg', 'txt')
            label_path = os.path.join(orig_dataset_dir, label)
            target_label_path = os.path.join(dataset_dir, label)
            exist = filter_class(label_path, target_label_path, keep_class)
            if exist:
                tmp_txt.append(line)
                os.system(f'cp {orig_dataset_dir}/{line.strip()} {dataset_dir}/{line.strip()}')
        with open(os.path.join(dataset_dir, split), 'w') as file:
            for line in tmp_txt:
                file.write(line)
        print('Done filtering', split, f'keep files {len(tmp_txt)}/{len(lines)}')
        keep_num.append(len(tmp_txt))
    return keep_num, data['names']

def filter_dataset_scenario(keep_scenario=[0], synthetic_error=0):
    '''
    EgoObjects contains multiple scenarios: a short video, within the scenario, the instance and object are roughly keep the same
    keep_scenario: a list of scenario index to keep, e.g., 0 refers to a cat scenario
    
    '''
    dataset_dir = 'dataset/EgoObjects/mini_scenario/' + '_'.join([str(i) for i in keep_scenario])
    orig_dataset_dir = 'dataset/EgoObjects'

    labels_dir = os.path.join(dataset_dir, 'labels')
    images_dir = os.path.join(dataset_dir, 'images')
    # remove and create new labels and images dir
    os.system(f'rm -rf {labels_dir}')
    os.system(f'rm -rf {images_dir}')
    os.makedirs(labels_dir)
    os.makedirs(images_dir)
    lines = []
    for split in ['val.txt']:
        split_file = os.path.join(orig_dataset_dir, split)
        with open(split_file, 'r') as file:
            lines += file.readlines()
    scenario_idx = {}
    for line in lines:
        line = line.strip()
        scenario = line[9:].split('_')[0]
        if scenario not in scenario_idx:
            scenario_idx[scenario] = [line]
        else:
            scenario_idx[scenario] += [line]
    idxs = []
    scenario_key = list(scenario_idx.keys())
    train_idx = []; val_idx = []
    for _scenario in keep_scenario:
        idx = scenario_idx[scenario_key[_scenario]]
        idxs += idx
        train_idx += idx[:int(len(idx)*0.8)]
        val_idx += idx[int(len(idx)*0.8):]
    with open(os.path.join(dataset_dir, 'train.txt'), 'w') as file:
        for line in train_idx:
            file.write(line + '\n')
    with open(os.path.join(dataset_dir, 'val.txt'), 'w') as file:
        for line in val_idx:
            file.write(line + '\n')

    for line in idxs:
        label = line.replace('images', 'labels').replace('jpg', 'txt')
        label_path = os.path.join(orig_dataset_dir, label)
        target_label_path = os.path.join(dataset_dir, label)
        os.system(f'cp {label_path} {target_label_path}')
        os.system(f'cp {orig_dataset_dir}/{line.strip()} {dataset_dir}/{line.strip()}')
    print('Done filtering', f'keep files {len(idxs)}/{len(lines)}')

    # keep the same way of naming
    import yaml
    with open('{}/EgoObjects.yaml'.format(orig_dataset_dir), 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    data['path'] = dataset_dir
    with open(os.path.join(dataset_dir, 'EgoObjects.yaml'), 'w') as file:
        yaml.dump(data, file)

    possible_class = []
    for line in idxs:
        label_path = line.replace('images', 'labels').replace('jpg', 'txt')
        label_path = os.path.join(dataset_dir, label_path)
        with open(label_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            if int(line[0]) not in possible_class:
                possible_class.append(int(line[0]))
    possible_class_name = {i:data['names'][i] for i in possible_class}
    print('Possible classes:', possible_class_name)
    return possible_class_name

if __name__ == '__main__':
    # filter_dataset_category([150])
    filter_dataset_scenario([0])