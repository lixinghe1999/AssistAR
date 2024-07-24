"""
Extract frames from clip
"""

import csv
import functools
import json
import multiprocessing
import os
import time
from collections import defaultdict
from typing import List, NamedTuple
from downsample import downsample_clip, split_frames

# import av

from PIL import Image
from tqdm import tqdm


class ExtractFramesWorkflowParams(NamedTuple):
    num_process: int = 8
    clip_dir: str = "../dataset/v2/clips"
    annotation_path: str = (
        "../dataset/v2/egotracks/egotracks_val.json"
    )
    output_dir: str = "../dataset/v2/clips_frames"

def run_single_process(clip_uid: str, params: ExtractFramesWorkflowParams):
    info_save_path = os.path.join(params.output_dir, "clip_info", f"{clip_uid}.csv")

    print(f"Start processing {clip_uid}!")
    s = time.time()
    # downsample_clip(data_folder=params.clip_dir, output_folder=os.path.join(params.output_dir, "frames"), clip=f"{clip_uid}.mp4", split_frames=True)
    split_frames(data_folder=params.clip_dir, output_folder=os.path.join(params.output_dir, "frames"), clip=f"{clip_uid}.mp4", split_frames=True)
    # clip_info csv
    frame_list = os.listdir(os.path.join(params.output_dir, "frames", clip_uid))
    # order it
    frame_list.sort(key=lambda x: int(x.split(".")[0]))

    with open(info_save_path, "w") as f:
         write = csv.writer(f, delimiter="\n")
         write.writerow(frame_list)
         
    print(f"Finished {clip_uid} in {time.time() - s}!")


def extract_clip_ids(file_path: str):
    with open(file_path, "r") as f:
        annotations = json.load(f)
    clip_uids = []
    for v in annotations["videos"]:
        for c in v["clips"]:
            # print(c.keys())
            if "exported_clip_uid" in c:
                clip_uids.append(c["exported_clip_uid"])
    return clip_uids


def remove_finished_clip_uids(clip_uids: List, params: ExtractFramesWorkflowParams):
    res = []
    info_save_dir = os.path.join(params.output_dir, "clip_info")

    for clip_uid in clip_uids:
        if not os.path.exists(os.path.join(info_save_dir, f"{clip_uid}.csv")):
            res.append(clip_uid)
        else:
            print(f"{clip_uid} was already extracted!")

    return res


def read_csv(path: str):
    if not os.path.exists(path):
        raise RuntimeError
    with open(path) as f:
        frame_numbers = [line.strip() for line in f.readlines()]
    return frame_numbers


def combine_clip_info(params):
    combined_save_path = os.path.join(params.output_dir, "clip_info.json")
    clip_info_dir = os.path.join(params.output_dir, "clip_info")
    clip_info_files = os.listdir(clip_info_dir)
    clip_info_dict = defaultdict(dict)
    for clip_info_file in tqdm(clip_info_files, total=len(clip_info_files)):
        clip_uid = clip_info_file.split(".csv")[0]

        info_path = os.path.join(clip_info_dir, f"{clip_uid}.csv")
        frame_numbers = read_csv(info_path)
        clip_info_dict[clip_uid]["frames"] = frame_numbers

    with open(combined_save_path, "w") as f:
        json.dump(clip_info_dict, f)


def main():
    params = ExtractFramesWorkflowParams()
    clip_uids = extract_clip_ids(params.annotation_path)
    # clip_uids = remove_finished_clip_uids(clip_uids, params)
    print(f"Total {len(clip_uids)} to be processed ...")
    clip_uids = clip_uids[:1]
    # # run_single_process(clip_uids[0], params=params)
    pool = multiprocessing.Pool(params.num_process)
    pool.map(functools.partial(run_single_process, params=params), clip_uids)

    pool.close()
    pool.join()
    # Combine info for each clip into one file
    combine_clip_info(params)


if __name__ == "__main__":
    main()