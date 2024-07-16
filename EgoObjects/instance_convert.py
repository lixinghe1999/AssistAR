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
gt_json_file = dataset + "EgoObjectsV1_unified_eval.json"
metadata_json_file = dataset + "EgoObjectsV1_unified_metadata.json"



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

def visualize(z, i):
    # plt.figure(figsize=(10, 10))
    fig, axs = plt.subplots(1, len(z), figsize=(len(z)*3, 3))
    for idx in range(len(z)):
        image, ann = z[idx]
        image_path = os.path.join(image_dir, image['url'])
        axs[idx].imshow(plt.imread(image_path))

        bbox = ann['bbox']
        x, y, w, h = bbox
        axs[idx].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='r', linewidth=2))
        axs[idx].axis('off')
        axs[idx].set_title(ann['category_freeform'])
    fig.savefig(f'figs/instance_{i}.png')

def instance_tracking(gt, split):
    instance_ids = list(gt.instance_ids)
    for i in range(10):
        instance_id = instance_ids[i]
        anno_ids = gt.dataset['instance_id_2_anno_ids'][str(instance_id)]
        output_list = []
        for anno_id in anno_ids:
            if anno_id in gt.anns['inst_det']:
                ann = gt.anns['inst_det'][anno_id]
                image = gt.imgs[ann['image_id']]
                output_list.append((image, ann))
        visualize(output_list, i)
        #     print(anno)
        #     instance_id = gt.dataset['instance_id_2_anno_ids'][str(ann['instance_id'])]
        #     print(instance_id)
    return
def main():
    dataset_name = "EgoObjects"
    metadata = get_egoobjects_meta(metadata_json_file)
    MetadataCatalog.get(dataset_name).set(**metadata)
    metadata = MetadataCatalog.get(dataset_name)

    # split = 'train'
    # gt = EgoObjects(gt_json_file, metadata, filter_opts=FILTER_OPTS[f"egoobjects_cat_det_{split}"])
    split = 'val'
    gt = EgoObjects(gt_json_file, metadata, filter_opts=FILTER_OPTS[f"egoobjects_cat_det_{split}"])
    instance_tracking(gt, split)


    # yaml_file = 'egoobjects.yaml'
    # with open(yaml_file, 'w') as f:
    #     cat_map = {key-1: value['name'] for key, value in gt.cats['cat_det'].items()}
    #     yaml.dump({'names': cat_map, 'nc': len(gt.cats['cat_det']), 'path': '../dataset/EgoObjects/', 'train': 'train.txt', 
    #                                            'val': 'val.txt'}, f, default_flow_style=False)

if __name__ == "__main__":
    main()