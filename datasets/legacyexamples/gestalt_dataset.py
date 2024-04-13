'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-02-25 02:44:44
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-02-25 02:44:57
 # @ Description: This file is distributed under the MIT license.
'''

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

class GestaltGridPointDataset(Dataset):
    def __init__(self, split = "train"):
        super().__init__()
    
    def __len__(self): return 1

    def __getitem__(self, idx):
        return idx

def grid_layout(resolution, dx, dy):
    """generate a grid layout for the point
    Args:
        resolution: the input points grid size.
        dx, dy: the difference between point pairs.
    """
    W, H = resolution[0], resolution[1]
    x = torch.linspace(0, W - 1, W) * dx
    y = torch.linspace(0, H - 1, H) * dy
    x = x / x.max()
    y = y / y.max()
    grid_x, grid_y = torch.meshgrid([x, y])
    grid = torch.cat([grid_x[..., None], grid_y[..., None]], dim = -1)
    return grid

if __name__ == "__main__":
    # generate Gesltatl dataset
    import os

    grid = grid_layout([8, 8], 0.5, 1.)
    plt.figure("figure", figsize = (5,4))
    plt.scatter(grid[...,0], grid[..., 1])
    plt.show()