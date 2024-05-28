'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-02-26 13:41:17
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-02-26 13:41:19
 # @ Description: This file is distributed under the MIT license.
'''

import torch
import torch.nn as nn

from torch.autograd import Variable

class CounterFactualWorldModel(nn.Module):
    def __init__(self, resolution = (128,128)):
        super().__init__()
        """
        a masked autoencoder as the counter factual world model, forced to capture short term actions.
        this is essentially a vmae that use xt and masked xt+d to predict the ground truth xt+d input, then
        several interfaces are provided to read specific visual features from the encoder
        """
        self.resolution = resolution
        # a masked autoencoder as the counter factual world model, forced to capture short term actions.
        self.encoder = None
        self.decoder = None

        """a patchify mask generator, mask the 1-p percent of the whole second frame"""
        self.mask_generator = None
        self.patch_generator = None # transform an image into a sequence of vectors of size of the patch
    
    def foward(self, x_t, x_tp):
        return 

    def sample_mask(self, size = (64,64), p = 0.1):
        mask = torch.randn(size)
        return mask