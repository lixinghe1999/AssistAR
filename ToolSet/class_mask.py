import os
from tqdm import tqdm
import json
import random
def scenario_candidate():
    all_class = list(range(80))
    select_class = random.sample(all_class, 3)
    save_dir = './temp/{}'.format('_'.join([str(i) for i in select_class]))
    os.makedirs(save_dir, exist_ok=True)
    generate_temp_list(select_class, out_dir=save_dir)

def generate_temp_list(select_class, out_dir='./temp'):
    class_map = {c:i for i, c in enumerate(select_class)}
    for split in ['train2017', 'val2017']:
        class_files = json.load(open('class_{}.json'.format(split), 'r'))
        class_files = [v for k, v in class_files.items() if int(k) in select_class]
        select_files = []
        for v in class_files:
            select_files.extend(v)
        with open(os.path.join(out_dir, 'class_{}_temp.txt'.format(split)), 'w') as file:
            for f in select_files:
                file.write(f+'\n')

    import yaml
    cocoyaml = yaml.load(open('./datasets/coco/coco.yaml', 'r'), Loader=yaml.FullLoader)
    names = cocoyaml['names']
    names = {class_map[int(i)]:names[int(i)] for i in select_class}
    cocoyaml['names'] = names
    yaml.dump(cocoyaml, open(os.path.join(out_dir, 'coco_temp.yaml'), 'w'), default_flow_style=False)
    return class_map
if __name__ == '__main__':
    # dataset_dir = './datasets/coco/labels/' 
    # for split in ['train2017', 'val2017']:
    #     data_dir = os.path.join(dataset_dir, split)
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
    #     json.dump(class_files, open('class_{}.json'.format(split), 'w'), indent=4)

    scenario_candidate()