'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-05-27 15:12:55
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-05-27 15:12:56
 # @ Description: This file is distributed under the MIT license.
'''

import torch
import torch.nn as nn
import numpy as np

class Patchify(nn.Module):
    def __init__(self, patch_size = (16,16)):
        self.patch_size = patch_size