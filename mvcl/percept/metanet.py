'''
 # @ Author: Zongjing Li
 # @ Create Time: 2023-12-14 03:34:00
 # @ Modified by: Zongjing Li
 # @ Modified time: 2023-12-14 03:34:17
 # @ Description: This file is distributed under the MIT license.
'''
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F

from .propagation import GraphPropagation
from .competition import Competition

from .backbones import ResidualDenseNetwork, FeatureMapEncoder
from rinarak.dklearn.cv.unet import UNet

from rinarak.utils.tensor import gather_tensor, stats_summary, weighted_softmax, logit
from torch_sparse import SparseTensor

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

def downsample_tensor(x, stride):
    # x should have shape [B, C, H, W]
    if stride == 1:
        return x
    B, C, H, W = x.shape
    x = F.unfold(x, kernel_size=1, stride=stride)  # [B, C, H / stride * W / stride]
    return x.reshape([B, C, int(H / stride), int(W / stride)])

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

    assert sample_inds.shape[0] == 3
    local_node_inds = sample_inds[2] # [B, N, K]

    batch_inds = torch.arange(B).reshape([B, 1]).to(local_node_inds)
    node_inds = torch.arange(N).reshape([1, N]).to(local_node_inds)
    row_inds = (batch_inds * N + node_inds).reshape(B * N, 1).expand(-1, K).flatten()  # [BNK]

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

class MetaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        device = config.device
        self.device = device
        num_prop_itrs = 72
        num_masks = config.max_num_masks
        W, H = config.resolution
        self.W = W
        self.H = H

        """general visual feature backbone, perfrom grid size convolution"""
        latent_dim = config.backbone_feature_dim
        rdn_args = SimpleNamespace(g0=latent_dim  ,RDNkSize=3,n_colors=config.channel_dim,
                               RDNconfig=(4,3,16),scale=[2],no_upsampling=True)
        self.backbone = ResidualDenseNetwork(latent_dim)
        #UNet(n_channels = config.channel_dim, n_classes = latent_dim, bilinear = True)
        #ResidualDenseNetwork(latent_dim)
        #FeatureMapEncoder(config.channel_dim, z_dim = latent_dim)
        #RDN(rdn_args)

        """local indices plus long range indices"""
        supervision_level = 1
        K = 7
        self.K = K
         # the local window size ( window size of [n x n])
        self.supervision_level = supervision_level
        for stride in range(1, supervision_level + 1):
            locals = generate_local_indices([W,H], K)
            self.register_buffer(f"indices_{W//stride}x{H//stride}", locals)

        self.u_indices = torch.arange(H * W).to(device)

        """graph propagation on the grid and masks extraction"""
        self.propagator = GraphPropagation(num_iters = num_prop_itrs)
        self.competition = Competition(num_masks = num_masks)

        kq_dim = 132
        self.ks_map = nn.Linear(latent_dim, kq_dim)
        self.qs_map = nn.Linear(latent_dim, kq_dim)
        self.num_long_range = int(7 * 7 * 0.2)

    def forward(self, ims, affinity_calculator, key = None, target_masks = None, lazy = True):
        """
        Args:
            ims: the image batch send to calculate the
        Returns:
            a diction that contains the output with keys 
            masks:
            connections:
        """
        if not lazy:assert len(ims.shape) == 4,"need to process with batch"
        elif len(ims.shape) == 3: ims = ims.unsqueeze(0)
        outputs = {}
        B, C, W, H = ims.shape

        all_logits = []
        all_sample_inds = []
        loss = 0.0

        for stride in range(1, self.supervision_level+1):
            indices = self.get_indices([W,H], B, stride)
            _, B, N, K = indices.shape
            # [3, B, N, K]
            affinity_features = affinity_calculator.calculate_affinity_feature(indices, ims)
            logits = affinity_calculator.calculate_entailment_logits(affinity_features, key)

            all_sample_inds.append(indices)

            if target_masks is None:
                stride_loss = 0.0
                util_logits = logits
            else:
                stride_loss, util_logits = self.compute_loss(logits,indices,target_masks,[W,H])
            loss += stride_loss
            all_logits.append(util_logits)
        
        """Compute segments by extracting the connected components"""
        masks, agents, alive, propmaps = self.compute_masks(all_logits[0],all_sample_inds[0])
        
        outputs["loss"] = loss
        outputs["masks"] = masks
        outputs["alive"] = alive
        outputs["all_logits"] = all_logits
        outputs["prop_maps"] = propmaps

        del all_sample_inds
        return outputs
    
    def get_indices(self,resolution = (128,128), B = 1, stride = 1, num_long_range = None):
        W, H = resolution
        device = self.device
        if num_long_range is None: num_long_range = self.num_long_range
        indices = getattr(self,f"indices_{W//stride}x{H//stride}").repeat(B,1,1).long()
        v_indices = torch.cat([
                indices, torch.randint(H * W, [B, H*W, num_long_range])
            ], dim = -1).unsqueeze(0)

        _, B, N, K = v_indices.shape # K as the number of local indices at each grid location

        """Gather batch-wise indices and the u,v local features connections"""
        u_indices = torch.arange(W * H).reshape([1,1,W*H,1]).repeat(1,B,1,K)
        batch_inds = torch.arange(B).reshape([1,B,1,1]).repeat(1,1,H*W,K).to(device)

        indices = torch.cat([
                batch_inds, u_indices, v_indices
            ], dim = 0)
        return indices
    
    def compute_loss(self,logits, sample_inds, target_masks, size = None):
        if len(target_masks.shape) == 3: target_masks = target_masks.unsqueeze(1)
        if size is None: size = [self.W, self.H]
        B, N, K = logits.shape

        segment_targets = F.interpolate(target_masks.float(), size, mode='nearest')

        segment_targets = segment_targets.reshape([B,N]).unsqueeze(-1).long().repeat(1,1,K)
        if sample_inds is not None:
            samples = torch.gather(segment_targets,1, sample_inds[2,...]).squeeze(-1)
        else:
            samples = segment_targets.permute(0, 2, 1)

        targets = segment_targets == samples
        null_mask = (segment_targets == 0) # & (samples == 0)  only mask the rows
        mask = 1 - null_mask.float()


        # [compute log softmax on the logits] (F.kl_div requires log prob for pred)
        y_pred = weighted_softmax(logits, mask)


        y_pred = torch.log(y_pred.clamp(min=1e-8))  # log softmax

        # [compute the target probabilities] (F.kl_div requires prob for target)
        y_true = targets / (torch.sum(targets, -1, keepdim=True) + 1e-9)

        # [compute kl divergence]
        kl_div = F.kl_div(y_pred, y_true, reduction='none') * mask
        kl_div = kl_div.sum(-1)

        # [average kl divergence aross rows with non-empty positive / negative labels]
        agg_mask = (mask.sum(-1) > 0).float()
        loss = kl_div.sum() / (agg_mask.sum() + 1e-9)



        return loss, y_pred
    

    
    def compute_masks(self, logits, indices, prop_dim = 64):
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
        prop_maps = self.propagator(h0.detach(), adj.detach())
        prop_map = prop_maps[-1].reshape([B,W,H,D])
        masks, agents, alive, phenotypes, _ = self.competition(prop_map)
        return masks, agents, alive, prop_maps
