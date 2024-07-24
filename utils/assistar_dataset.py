import torch
from .base_video_dataset import BaseVideoDataset
from .ego4d_lt_tracking_dataset import EGO4DLTTrackingDataset
from .sequence_dataset import EGO4DLTT
import random
class AssistDataset(EGO4DLTT):
    '''
    Work similar to sequence_dataset.py, but can access blank frames (full-video)
    '''
    def __init__(
        self,
        data_dir: str,
        annotation_path: str,
        data_fraction=None,
    ):
        """
        args: all the same with EGO4DLTT
            data_dir - directory that contains image files
            annotation_path - which annotation file to read
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        super().__init__(data_dir, annotation_path, data_fraction)
    def __len__(self):
        return len(self.ego4d_lt_tracking.sequences)  
    def get_item(self, idx):
        '''
        idx = sequence id
        '''
        anno = self.get_sequence_info(idx)
        frame_numbers = anno["frame_numbers"]
        max_frame_ids = len(frame_numbers)

        
        frame_list, anno_frames, obj_meta = self.get_frames(idx, [0, 1, 2, 3, 4, 5,], anno)
        return frame_list, anno_frames, obj_meta

    def get_frames(self, seq_id, frame_ids, anno=None):
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        seq = self.ego4d_lt_tracking[seq_id]
        # obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        frame_numbers = anno["frame_numbers"]
        valid_frame_ids = [frame_numbers[f_id] for f_id in frame_ids]
        obj_meta = {'object_ids': anno['object_ids'], 
                    'valid_frame_ids': [frame_number - valid_frame_ids[0] for frame_number in valid_frame_ids]}

        frame_numbers = range(valid_frame_ids[0], valid_frame_ids[-1]+1)
        # fill with many no-object frames
        frame_list = [
            self._get_frame(seq.frames[f_id]) for f_id in frame_numbers
        ]
        print(len(frame_list))

        anno_frames = {}
        for key, value in anno.items():
            if type(value) == torch.Tensor:
                anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        return frame_list, anno_frames, obj_meta,