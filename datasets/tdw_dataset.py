'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-04-16 06:37:16
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-04-16 06:37:19
 # @ Description: This file is distributed under the MIT license.
'''
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from rinarak.utils.os import load_json

local = True
dataset_dir = "/Users/melkor/Documents/datasets" if local else "datasets"

def normal_img(img):
    if len(img.shape) == 4:
        if not img.shape[1] in [1,3,4]: return img.permute(0,3,1,2)
    if len(img.shape) == 3:
        if not img.shape[0] in [1,3,4]: return img.permute(2,0,1)

def identiy_masks(img):
    return

class TDWRoomDataset(Dataset):
    def __init__(self,name="TDWRoom", split = "train", resolution = (128,128), root_dir = "datasets", motion_only = True):
        super().__init__()
        self.split = split
        self.root_dir = root_dir + f"/{name}"

        img_data_path = root_dir + f"/{name}"+ f"/{split}/img"
        self.files = os.listdir(img_data_path)

        """ add a working resolution to adapt different scenes and parameters"""
        self.transform = transforms.Resize(resolution)
        self.motion_only = motion_only
    
    def __len__(self):
        return len(self.files) // 4
    
    def __getitem__(self, idx):
        root_dir = self.root_dir
        split = self.split
        img_data_path = root_dir + f"/{self.split}/img"

        scene_data_path = root_dir + f"/{self.split}/scene/{idx}.json"
        scene_setup = load_json(scene_data_path)

        data = {}
        img = torch.tensor(plt.imread(img_data_path + f"/img_{idx}.png"))
        albedo = torch.tensor(plt.imread(img_data_path + f"/albedo_{idx}.png"))
        id_map = torch.tensor(plt.imread(img_data_path + f"/id_{idx}.png"))
        masks = np.load(img_data_path + f"/mask_{idx}.npy")
        ids_seq = np.load(root_dir + f"/{self.split}/scene/ids_{idx}.npy")
        #masks = torch.tensor(plt.imread(img_data_path + f"/id_{split}_{idx}.png"))
        
        for i,id in enumerate(ids_seq):
            #print( str(int(id)), scene_setup[str(int(id))]["model"], scene_setup[str(int(id))]["movable"])
            if self.motion_only and not scene_setup[str(int(id))]["movable"]:
                masks[torch.tensor(masks).int() == i] = 0


        data["img"] = self.transform(normal_img(img))
        data["albedo"] = self.transform(normal_img(albedo))
        data["masks"] =self.transform(torch.tensor(masks).unsqueeze(0)).squeeze(0)
        data["ids"] = id_map
        data["ids_sequence"] = ids_seq
        data["scene"] = scene_setup
        return data