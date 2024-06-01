'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-05-31 20:35:50
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-05-31 20:35:55
 # @ Description: This file is distributed under the MIT license.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse  import SparseTensor
from torch_geometric.nn    import max_pool_x, GraphConv
from torch_geometric.data  import Data,Batch
from torch_geometric.utils import grid, to_dense_batch

from skimage import measure
import numpy as np

from .backbones.resnet import ResidualDenseNetwork
from .backbones.propagation import GraphPropagation
from .backbones.competition import Competition
from .affinities import DotProductAffinity, InverseNormAffinity
from rinarak.utils.tensor import logit


def to_cc_masks(masks, threshold = 12):
    all_masks = []
    for i in range(masks.size(-1)):
        #all_labels = measure.label(masks[:,:,i])
        blobs_labels = measure.label(masks[:,:,i]>0.5, background=0)
        for cc_label in range(1,blobs_labels.max()):
            cc_mask = blobs_labels == cc_label
            if cc_mask.sum() > threshold:
                all_masks.append(cc_mask[...,None])
    all_masks = np.concatenate(all_masks, axis = -1)
    return all_masks

def weighted_softmax(x, weight):
    maxes = torch.max(x, -1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    x_exp_sum = (torch.sum(x_exp * weight, -1, keepdim=True) + 1e-12)
    return (x_exp / x_exp_sum) * weight

def generate_local_indices(img_size, K, padding = 'constant'):
    H, W = img_size
    indice_maps = torch.arange(H * W).reshape([1, 1, H, W]).float()

    # symetric_padding
    assert K % 2 == 1 # assert K is odd
    half_K = int((K - 1) / 2)

    assert padding in ["reflection", "constant"]
    if padding == "constant":
        pad_fn = torch.nn.ReflectionPad2d(half_K)
    else:
        pad_fn = torch.nn.ConstantPad2d(half_K)
    
    indice_maps = pad_fn(indice_maps)

    local_inds = F.unfold(indice_maps, kernel_size = K, stride = 1)

    local_inds = local_inds.permute(0,2,1)
    return local_inds

def local_to_sparse_global_affinity(local_adj, sample_inds, activated=None, sparse_transpose=False):
    """
    Convert local adjacency matrix of shape [B, N, K] to [B, N, N]
    :param local_adj: [B, N, K]
    :param size: [H, W], with H * W = N
    :return: global_adj [B, N, N]
    """

    B, N, K = list(local_adj.shape)

    if sample_inds is None:
        return local_adj
    device = local_adj.device

    assert sample_inds.shape[0] == 3
    local_node_inds = sample_inds[2] # [B, N, K]

    batch_inds = torch.arange(B).reshape([B, 1]).to(local_node_inds)
    node_inds = torch.arange(N).reshape([1, N]).to(local_node_inds)
    row_inds = (batch_inds * N + node_inds).reshape(B * N, 1).expand(-1, K).flatten().to(device)  # [BNK]

    col_inds = local_node_inds.flatten()  # [BNK]
    valid = col_inds < N

    col_offset = (batch_inds * N).reshape(B, 1, 1).expand(-1, N, -1).expand(-1, -1, K).flatten() # [BNK]
    col_inds += col_offset
    value = local_adj.flatten()

    if activated is not None:
        activated = activated.reshape(B, N, 1).expand(-1, -1, K).bool()
        valid = torch.logical_and(valid, activated.flatten())

    if sparse_transpose:
        global_adj = SparseTensor(row=col_inds[valid], col=row_inds[valid],
                                  value=value[valid], sparse_sizes=[B*N, B*N])
    else:
        raise ValueError('Current KP implementation assumes tranposed affinities')

    return global_adj

class SymbolicGrouper(nn.Module):
    def __init__(self, resolution = (128,128), K = 7, long_range_ratio = 0.2):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        W, H = resolution
        self.W, self.H = W, H

        """create the convolutional backbone for low level feature extraction"""
        base_feature_dim = 128
        self.convolution_net = ResidualDenseNetwork(grow0 = base_feature_dim, n_colors = 3)

        """local indices plus long range indices"""
        _ ,self.spatial_coords = grid(W, H,device = device)
        self.spatial_coords = self.spatial_coords / W

        """generate local indices to build a complete graph"""
        self.K = K
        local_indices = generate_local_indices([W,H], K)
        self.register_buffer(f"indices_{W}x{H}", local_indices)
        self.num_long_range = int(K * K * long_range_ratio)

        """visual cue affinities for comprehensive understanding of the scene"""
        self.affinity_modules = nn.ModuleDict()
        self.add_movable_affinity()

        """extract segments from the affinity graph using the graph propagation and competition"""
        self.propagation = GraphPropagation(num_iters = 232)
        self.competition = Competition(num_masks = 30)
    
    def add_movable_affinity(self): self.affinity_modules["movable"] = DotProductAffinity()

    def preprocess(self, img):
        if img.shape[1] > 4: return img.permute(0,2,3,1)
        return img

    def get_indices(self,resolution = (128,128), B = 1, stride = 1, num_long_range = None):
        W, H = resolution
        device = self.device
        if num_long_range is None: num_long_range = self.num_long_range
        indices = getattr(self,f"indices_{W//stride}x{H//stride}").repeat(B,1,1).long().to(device)
        v_indices = torch.cat([
                indices, torch.randint(H * W, [B, H*W, num_long_range]).to(device)
            ], dim = -1).unsqueeze(0).to(device)

        _, B, N, K = v_indices.shape # K as the number of local indices at each grid location

        """Gather batch-wise indices and the u,v local features connections"""
        u_indices = torch.arange(W * H).reshape([1,1,W*H,1]).repeat(1,B,1,K).to(device)
        batch_inds = torch.arange(B).reshape([1,B,1,1]).repeat(1,1,H*W,K).to(device)

        indices = torch.cat([
                batch_inds, u_indices, v_indices
            ], dim = 0)
        return indices

    def extract_segments(self, indices, logits, prop_dim = 64):
        W, H = self.W, self.H
        B, N, K = logits.shape
        D = prop_dim
        """Initalize the latent space vector for the propagation"""
        h0 = torch.FloatTensor(B,N,D).normal_().to(logits.device).softmax(-1)

        """Construct the connection matrix"""
        adj = torch.softmax(logits, dim = -1)
        adj = adj / torch.max(adj, dim = -1, keepdim = True)[0].clamp(min=1e-12)

        # tansform the indices into the sparse global indices
        adj = local_to_sparse_global_affinity(adj, indices, sparse_transpose = True)
        
        """propagate random normal latent features by the connection graph"""
        prop_maps = self.propagation(h0.detach(), adj.detach())
        prop_map = prop_maps[-1].reshape([B,W,H,D])
        masks, agents, alive, phenotypes, _ = self.competition(prop_map)
        return masks, agents, alive, prop_maps

    def calculate_inference_loss(self, logits, edges, target_masks, size = None):
        B, W, H = target_masks.shape
        _, M = edges.shape
        row, col = edges

        samples = target_masks.reshape([B * W * H])



        targets = ( (samples[row] == samples[col]) & (samples[row] != 0) ).float()
        null_mask = (samples[row] == 0) # & (samples == 0)  only mask the rows
        mask = 1 - null_mask.float()


        # [compute log softmax on the logits] (F.kl_div requires log prob for pred)
        #mask = torch.ones_like(mask)
        y_pred = weighted_softmax(logits, mask)
        #y_pred = torch.softmax(logits, dim = -1)


        y_pred = torch.log(y_pred.clamp(min=1e-8))  # log softmax

        # [compute the target probabilities] (F.kl_div requires prob for target)
        y_true = targets / (torch.sum(targets, -1, keepdim=True) + 1e-9)
        
        # [compute kl divergence]
        kl_div = F.kl_div(y_pred, y_true, reduction='none') * mask
        loss = kl_div.sum(-1)
        return loss, y_true

    def forward(self, img, cues, verbose = False):
        """
        Args:
            img: a batch of image in the shape BxCxWxH
        Returns:
            outputs: a diction containing the info of output
        """
        ### extract visual features using a convolutional network and
        img = self.preprocess(img)
        B, W, H, C = img.shape
        
        im_feats = self.convolution_net(img)
        coords_added_im_feats = torch.cat([
                  self.spatial_coords.unsqueeze(0).repeat(im_feats.size(0),1,1),
                  im_feats.flatten(2,3).permute(0,2,1)
                                          ],dim=2)
        coords_added_im_feats = im_feats
        coords_added_im_feats = torch.nn.functional.normalize(coords_added_im_feats, dim = -1)
        
        coords_added_im_feats = im_feats.flatten(2,3).permute(0,2,1)
        if verbose: print(coords_added_im_feats.shape)

        ### build a quais-local graph using stored indices and long range indices
        indices = self.get_indices(resolution = (self.W, self.H),B = B)
        _, _, N, K = indices.shape
        from_indices = indices[1,:,:,:].reshape([B, 1, N * K])
        to_indices = indices[2,:,:,:].reshape([B, 1, N * K])
        spatial_edges = torch.cat([from_indices, to_indices], dim = 1) # Bx2xM

        ### Run image feature graph through affinity modules
        graph_in = Batch.from_data_list([Data(x,edge)
                                                for x,edge in zip(coords_added_im_feats, spatial_edges)])
        x, edge_index, batch = graph_in.x, graph_in.edge_index, graph_in.batch
        if verbose: print("graph_in", x.shape, edge_index.shape, batch.shape)
        
        losses = {}
        all_affinity = {}
        masks = {}
        for affinity_key in self.affinity_modules:
            affinity_aggregator = self.affinity_modules[affinity_key]
            affinity, threshold, loss = affinity_aggregator(x, edge_index, batch, x.device)
            losses[affinity_key] = loss
            
            affinity = affinity - threshold
            if verbose: print(affinity.shape, affinity.max(), affinity.min())
            
            """inference the affinity using the ground truth cue to perform segmentation"""
            if cues is not None and affinity_key in cues:
                loss, aff_true = self.calculate_inference_loss(
                    affinity, edge_index, cues[affinity_key])
                losses[affinity_key] = losses[affinity_key] + loss
            
            aff_masks, agents, alive, prop_maps = self.extract_segments(indices, affinity.reshape([B, N, K]))
            all_affinity[affinity_key] = affinity
            masks[affinity_key] = aff_masks

        outputs = {
            "loss": losses,
            "masks": masks,
            "indices": indices,
            "affinity": all_affinity,
            "prop_maps": prop_maps
        }

        return outputs