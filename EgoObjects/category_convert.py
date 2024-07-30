#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import unittest
from copy import deepcopy
import os
import numpy as np
from detectron2.utils.logger import create_small_table
from detectron2.data import MetadataCatalog
from egoobjects_api.eval import EgoObjectsEval
from egoobjects_api.results import EgoObjectsResults
from egoobjects_api.egoobjects import EgoObjects, FILTER_OPTS
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataset = '../dataset/EgoObjects/'
image_dir = dataset + 'images/'




def get_egoobjects_meta(metadata_path: str,):
    """
    return metadata dictionary with 4 keys:
        cat_det_cats
        inst_det_cats
        cat_det_cat_id_2_cont_id
        cat_det_cat_names
    """
    with open(metadata_path, "r") as fp:
        metadata = json.load(fp)

    cat_det_cat_id_2_name = {cat["id"]: cat["name"] for cat in metadata["cat_det_cats"]}
    cat_det_cat_ids = sorted([cat["id"] for cat in metadata["cat_det_cats"]])
    cat_det_cat_id_2_cont_id = {cat_id: i for i, cat_id in enumerate(cat_det_cat_ids)}
    cat_det_cat_names = [cat_det_cat_id_2_name[cat_id] for cat_id in cat_det_cat_ids]

    metadata["cat_det_cat_id_2_cont_id"] = cat_det_cat_id_2_cont_id
    metadata["cat_det_cat_names"] = cat_det_cat_names
    return metadata

def visualize(image, anns, img_ann_map, cats, i):
    image_path = os.path.join(image_dir, image['url'])
    img = plt.imread(image_path)
    plt.figure(figsize=(10, 10))
    ann_ids = img_ann_map[image['id']]
    for ann_id in ann_ids:
        ann = anns[ann_id]
        bbox = ann['bbox']
        x, y, w, h = bbox
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='r', linewidth=2))
        plt.text(x, y, cats[ann['category_id']]['name'], fontsize=12, color='r')
    plt.imshow(img)
    plt.savefig(f'figs/{i}.png')
def yolo_convert(gt, split, cat_id_map=None):
    file_list = []
    for i in tqdm(range(len(gt.dataset['images']))):
        image = gt.dataset['images'][i]
        if i < 0:
            visualize(image, gt.anns['cat_det'], gt.img_ann_map['cat_det'], gt.cats['cat_det'], i)

        height, width = image['height'], image['width']
        label_path = os.path.join(dataset + 'labels/', image['url'].replace('.jpg', '.txt'))
        ann_ids = gt.img_ann_map['cat_det'][image['id']]
        coco_list = []
        for ann_id in ann_ids:
            ann = gt.anns['cat_det'][ann_id]

            if cat_id_map is not None:
                cat_id = ann['category_id']
                if cat_id not in cat_id_map: # not frequent, remove
                    continue
                else: # remap the category id
                    ann['category_id'] = cat_id_map[cat_id]
            bbox = ann['bbox']
            x, y, w, h = bbox
            x, y, w, h = (x+ w/2)/width, (y + h/2)/height, w/width, h/height
            # we get weird out-of-bound
            if x + w/2 >= 1: w = (1 - x) * 2
            if y + h/2 >= 1: h = (1 - y) * 2
            if x - w/2 <= 0: w = x * 2
            if y - h/2 <= 0: h = y * 2
            coco_list.append(" ".join([str(ann['category_id']), str(x), str(y), str(w), str(h)]))
        with open(label_path, 'w') as f:
            f.write("\n".join(coco_list))
        file_list.append(os.path.join('./images', image['url']))
    with open(f'{dataset}{split}.txt', 'w') as f:
        f.write("\n".join(file_list))
def category_distribution(gt, name):
    cat_distribution = {cat: 0 for cat in gt.cats['cat_det']}
    for image_id in gt.imgs:
        ann_ids = gt.img_ann_map['cat_det'][image_id]
        for ann_id in ann_ids:
            ann = gt.anns['cat_det'][ann_id]
            cat_id = ann['category_id']
            cat_distribution[cat_id] += 1

    # sort the categories by frequency
    cat_distribution_ordered = {k: v for k, v in sorted(cat_distribution.items(), key=lambda item: item[1], reverse=True)}
    # only keep the 20 frequent categories
    cat_distribution_filtered = {k: v for k, v in cat_distribution_ordered.items() if v > 100}
    # mapping from the orginal category id to the new category id
    cat_id_map = {cat_id: i for i, cat_id in enumerate(cat_distribution_filtered.keys())}
    print("Keep {} categories".format(len(cat_distribution_filtered)), 
          "the average number of instances per category is", np.mean(list(cat_distribution_filtered.values())))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].bar(range(len(cat_distribution)), cat_distribution.values())
    axs[0].set_yscale('log')
    axs[1].bar(range(len(cat_distribution_filtered)), cat_distribution_filtered.values())
    axs[1].set_yscale('log')
    
    plt.savefig('figs/{}_distribution.png'.format(name))
    return cat_id_map
def main():
    metadata_json_file = dataset + "EgoObjectsV1_unified_metadata.json"
    dataset_name = "EgoObjects"
    metadata = get_egoobjects_meta(metadata_json_file)
    MetadataCatalog.get(dataset_name).set(**metadata)
    metadata = MetadataCatalog.get(dataset_name)

    gt_train = EgoObjects(dataset + "EgoObjectsV1_unified_train.json", metadata, filter_opts=FILTER_OPTS[f"egoobjects_cat_det_train"])
    cat_id_map_train = category_distribution(gt_train, 'train')
    gt_val = EgoObjects(dataset + "EgoObjectsV1_unified_eval.json", metadata, filter_opts=FILTER_OPTS[f"egoobjects_cat_det_train"])
    cat_id_map_val = category_distribution(gt_val, 'val') # don't remap the category id to keep train and val consistent
    
    cat_id_map = {}
    for cat_id in cat_id_map_val:
        if cat_id not in cat_id_map_train: # not present in train, remove
            continue
        else:
            cat_id_map[cat_id] = len(cat_id_map)  
    print("After filtering, keep {} categories".format(len(cat_id_map)))
    yolo_convert(gt_train, 'train', cat_id_map)
    yolo_convert(gt_val, 'val', cat_id_map)

    

    yaml_file = dataset + '/EgoObjects.yaml'
    with open(yaml_file, 'w') as f:
        cat_names = gt_val.cats['cat_det']
        cat_names = {cat: cat_names[cat_id_ori] for cat_id_ori, cat in cat_id_map.items()}
        cat_map = {key: value['name'] for key, value in cat_names.items()}
        yaml.dump({'names': cat_map, 'nc': len(cat_names), 'path': '../dataset/EgoObjects/', 'train': 'train.txt', 
                                               'val': 'val.txt'}, f, default_flow_style=False)

if __name__ == "__main__":
    main()