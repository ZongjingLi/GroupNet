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
    def __init__(self):
        super().__init__()
        """ the pre-linguistic foundation model for solving distinct tasks.
        """
        # a masked autoencoder as the counter factual world model, forced to capture short term actions.
        self.encoder = None
        self.decoder = None
    
    def foward(self, x_t, x_tp):
        return 

    def sample_mask(self, size = (64,64), p = 0.1):
        mask = torch.randn(size)
        return mask