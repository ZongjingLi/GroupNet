'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-02-16 16:48:11
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-02-16 16:48:14
 # @ Description: This file is distributed under the MIT license.
'''
import torch
import torch.nn as nn

class LineDemo(nn.Module):
    def __init__(self, num_objs = 3, min_parts_num = 3, max_parts_num = 5):
        super().__init__()
        self.sample_nums = 32
        self.data = self.generate()
    
    def generate(self):
        data = []
        for i in range(self.sample_nums):
            scene_bind = {}
            data.append(scene_bind)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self): return len(self.data)