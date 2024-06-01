'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-05-31 20:36:24
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-05-31 20:38:32
 # @ Description: This file is distributed under the MIT license.
'''
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from rinarak.utils.os import load_json
import matplotlib.pyplot as plt
import os

def normal_img(img):
    if len(img.shape) == 4:
        if not img.shape[1] in [1,3,4]: return img.permute(0,3,1,2)
    if len(img.shape) == 3:
        if not img.shape[0] in [1,3,4]: return img.permute(2,0,1)

class PlagueWorksDataset(Dataset):
    def __init__(self,name="Plagueworks", split = "train", resolution = (128,128), dataset_dir = "datasets", motion_only = True):
        super().__init__()
        self.split = split
        self.root_dir = dataset_dir + f"/{name}"

        img_data_path = dataset_dir + f"/{name}"+ f"/{split}/img"
        self.files = os.listdir(img_data_path)

        """ add a working resolution to adapt different scenes and parameters"""
        self.transform = transforms.Resize(resolution)
        self.motion_only = motion_only
    
    def __len__(self):
        return len(self.files) // 4
    
    def __len__(self): return 50

    def __getitem__(self, idx):
        root_dir = self.root_dir
        split = self.split
        img_data_path = root_dir + f"/{self.split}/img"

        #scene_data_path = root_dir + f"/{self.split}/scene/{idx}.json"
        #scene_setup = load_json(scene_data_path)
        avatar_id = "a"

        data = {}
        img1 = torch.tensor(plt.imread(img_data_path + f"/img_{idx}_1_{avatar_id}.png"))
        img2 = torch.tensor(plt.imread(img_data_path + f"/img_{idx}_2_{avatar_id}.png"))
        albedo = torch.tensor(plt.imread(img_data_path + f"/albedo_{idx}_1_{avatar_id}.png"))
        id_map = torch.tensor(plt.imread(img_data_path + f"/id_{idx}_1_{avatar_id}.png"))
        masks = np.load(img_data_path + f"/mask_{idx}_1.npy")
        ids_seq = np.load(root_dir + f"/{self.split}/scene/ids_{idx}_1.npy")

        #for i,id in enumerate(ids_seq):
        #    #print( str(int(id)), scene_setup[str(int(id))]["model"], scene_setup[str(int(id))]["movable"])
        #    if self.motion_only and not scene_setup[str(int(id))]["movable"]:
        #        masks[torch.tensor(masks).int() == i] = 0
        masks = self.process_segmentation_color(id_map.permute(2,0,1) )


        data["img1"] = self.transform(normal_img(img1))
        data["img2"] = self.transform(normal_img(img2))
        data["albedo"] = self.transform(normal_img(albedo))
        data["masks"] = self.transform(masks.unsqueeze(0)).squeeze(0)
        data["ids"] = id_map
        data["ids_sequence"] = ids_seq
        #data["scene"] = scene_setup
        return data

    @staticmethod
    def _object_id_hash(objects, val=256, dtype=torch.long):
        C = objects.shape[0]
        objects = objects.to(dtype)
        out = torch.zeros_like(objects[0:1, ...])
        for c in range(C):
            scale = val ** (C - 1 - c)
            out += scale * objects[c:c + 1, ...]
        return out

    def process_segmentation_color(self, seg_color):
        # convert segmentation color to integer segment id
        seg_color = (seg_color * 256 ).long()
        raw_segment_map = self._object_id_hash(seg_color, val=256, dtype=torch.long)
        raw_segment_map = raw_segment_map.squeeze(0)

        # remove zone id from the raw_segment_map
        #meta_key = 'playroom_large_v3_images/' + file_name.split('/images/')[-1] + '.hdf5'
        #zone_id = int(self.meta[meta_key]['zone'])
        zone_id = 0
        #raw_segment_map[raw_segment_map == zone_id] = 0

    
        # convert raw segment ids to a range in [0, n]
        _, segment_map = torch.unique(raw_segment_map, return_inverse=True)
        segment_map -= segment_map.min()

        return segment_map