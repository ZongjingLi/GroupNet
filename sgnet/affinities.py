'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-05-31 20:33:34
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-05-31 20:33:39
 # @ Description: This file is distributed under the MIT license.
'''
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from torch_sparse  import SparseTensor
from torch_scatter import scatter_max
from torch_scatter import scatter_mean


from rinarak.utils.tensor import logit

class AffinityConditionedAggregation(torch.nn.Module, ABC):

    # Takes in tensor of node pairs and returns an affinity tensor and a 
    # threshold tensor to filter the affinites with. Also returns any loss
    # items to pass back to the training layer as a dict.
    # x is the list of graph nodes and row, col are the tensors of the adj. list
    @abstractmethod
    def affinities_and_thresholds(self, x, row, col):
        pass

    def forward(self, x, edge_index, batch, device):
        """
        Args:
            x: feature vector in the shape of MxD
            edge_index: edge_index in the shape of Bx2xM
            
        """

        row, col = edge_index

        ### Collect affinities/thresholds to filter edges 

        affinities, threshold, losses = self.affinities_and_thresholds(x,row,col)
        return affinities, threshold, losses

class DotProductAffinity(AffinityConditionedAggregation):
    def affinities_and_thresholds(self, nodes, row, col):
        device = nodes.device
        n, d = nodes.shape
        # Norm of difference for every node pair on grid
        row_features = nodes[row]
        col_features = nodes[col]
        
        edge_affinities = (row_features * col_features).sum(dim = -1) * (d ** -0.5)


        affinity_thresh = torch.zeros_like(edge_affinities, device = device)
        return edge_affinities.to(device), affinity_thresh.to(device), 0.0

class InverseNormAffinity(AffinityConditionedAggregation):
    
    def __init__(self):
        super().__init__()
        self.threholds = nn.Parameter(torch.tensor(3.0))

    def affinities_and_thresholds(self, nodes, row, col):
        device = nodes.device
        n, d = nodes.shape
        eps = 0.01
        # Norm of difference for every node pair on grid
        edge_affinities = torch.linalg.norm(nodes[row] - nodes[col],dim = 1) # this is for the difference version
        edge_affinities = 1 / (edge_affinities + eps)
        edge_affinities = logit(edge_affinities)

        # Inverse mean affinities for each node to threshold each edge with
        inv_mean_affinity = scatter_mean(edge_affinities, row.to(nodes.device))
        affinity_thresh   = torch.min(inv_mean_affinity[row],
                                      inv_mean_affinity[col])
        #affinity_thresh = torch.nn.functional.softplus(self.threholds).to(device)
        return edge_affinities.to(device), .1 * affinity_thresh , 0.0
    
class EcllipticalBoundary(nn.Module):
    def __init__(self):
        super().__init__()

    
class LinearBoundary(nn.Module):
    def __init__(self):
        super().__init__()
        