'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-02-13 00:47:32
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-02-13 00:47:34
 # @ Description: This file is distributed under the MIT license.
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ClevrVQADataset(Dataset):
    def __init__(self, split = "train"):
        super().__init__()
        self.split = split
    
    def __getitem__(self, idx):
        return idx
    
    def __len__(self):
        return 0