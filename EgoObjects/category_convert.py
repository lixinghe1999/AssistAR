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




def get_egoobjects_meta(metadata_path: str):
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
def yolo_convert(gt, split):
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
            bbox = ann['bbox']
            x, y, w, h = bbox
            x, y, w, h = (x+ w/2)/width, (y + h/2)/height, w/width, h/height
            # we get weird out-of-bound
            if x + w/2 >= 1: w = (1 - x) * 2
            if y + h/2 >= 1: h = (1 - y) * 2
            if x - w/2 <= 0: w = x * 2
            if y - h/2 <= 0: h = y * 2
            coco_list.append(" ".join([str(ann['category_id']-1), str(x), str(y), str(w), str(h)]))
        with open(label_path, 'w') as f:
            f.write("\n".join(coco_list))
        file_list.append(os.path.join('./images', image['url']))
    with open(f'{split}.txt', 'w') as f:
        f.write("\n".join(file_list))
def category_distribution(gt, name):
    cat_distribution = {cat: 0 for cat in gt.cats['cat_det']}
    for image_id in gt.imgs:
        ann_ids = gt.img_ann_map['cat_det'][image_id]
        for ann_id in ann_ids:
            ann = gt.anns['cat_det'][ann_id]
            cat_id = ann['category_id']
            cat_distribution[cat_id] += 1
    # bar plot in log
    plt.figure(figsize=(10, 10))
    plt.yscale('log')
    plt.bar(cat_distribution.keys(), cat_distribution.values())
    plt.savefig('figs/{}_distribution.png'.format(name))

def main():
    metadata_json_file = dataset + "EgoObjectsV1_unified_metadata.json"
    dataset_name = "EgoObjects"
    metadata = get_egoobjects_meta(metadata_json_file)
    MetadataCatalog.get(dataset_name).set(**metadata)
    metadata = MetadataCatalog.get(dataset_name)

    split = 'train'
    gt = EgoObjects(dataset + "EgoObjectsV1_unified_train.json", metadata, filter_opts=FILTER_OPTS[f"egoobjects_cat_det_train"])
    category_distribution(gt, split)
    yolo_convert(gt, split)

    # split = 'val'
    # gt = EgoObjects(dataset + "EgoObjectsV1_unified_eval.json", metadata, filter_opts=FILTER_OPTS[f"egoobjects_cat_det_train"])
    # category_distribution(gt, split)
    # yolo_convert(gt, split)

    
   

    # yaml_file = 'EgoObjects.yaml'
    # with open(yaml_file, 'w') as f:
    #     cat_map = {key-1: value['name'] for key, value in gt.cats['cat_det'].items()}
    #     yaml.dump({'names': cat_map, 'nc': len(gt.cats['cat_det']), 'path': '../dataset/EgoObjects/', 'train': 'train.txt', 
    #                                            'val': 'val.txt'}, f, default_flow_style=False)

if __name__ == "__main__":
    main()