from utils.sequence_dataset import EGO4DLTT
from utils.tracking_dataset import TrackingDataset
from utils.assistar_dataset import AssistDataset
import matplotlib.pyplot as plt
import torch

# dataset = EGO4DLTT(data_dir='dataset/v2/clips_frames', annotation_path='dataset/v2/egotracks/egotracks_val.json')


# frame_list, anno_frames, obj_meta = dataset.get_frames(0, [10, 20,], None)
# print(anno_frames, obj_meta)
# for i, (frame, anno) in enumerate(zip(frame_list, anno_frames['bbox'])):
#     print(f"frame: {frame.shape}, anno: {anno}")

#     plt.imshow(frame)
#     plt.gca().add_patch(plt.Rectangle((anno[0], anno[1]), anno[2], anno[3], fill=False, edgecolor='red', linewidth=2))
#     plt.axis('off')
#     plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

#     plt.savefig('{}.png'.format(i))
#     plt.close()

dataset = AssistDataset(data_dir='dataset/v2/clips_frames', annotation_path='dataset/v2/egotracks/egotracks_val.json')

frame_list, anno_frames, obj_meta = dataset.get_frames(0, [0, 1, 2, 3, 4, 5,], None)
print(anno_frames, obj_meta)
for i, frame in enumerate(frame_list):
    print(f"frame: {frame.shape}")
    if i in obj_meta['valid_frame_ids']:
        anno = anno_frames['bbox'][obj_meta['valid_frame_ids'].index(i)]
        plt.imshow(frame)
        plt.gca().add_patch(plt.Rectangle((anno[0], anno[1]), anno[2], anno[3], fill=False, edgecolor='red', linewidth=2))
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        plt.savefig('figs/{}.png'.format(i))
        plt.close()
    else: 
        anno = [0, 0, 0, 0]
        print("no object")

    

