import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sgnet.grouping import GraphPropagation, Competition, local_to_sparse_global_affinity, generate_local_indices
from sgnet.utils import to_cc_masks

w = 2
boundary = torch.zeros([1, 128, 128])
boundary[:, 20: 100, 40:40+w] = 1
boundary[:, 20: 100, 80:80+w] = 1
boundary[:, 62: 62+w, 40:80+w] = 1
boundary[:, 20: 20+w, 40:80+w] = 1

import cv2 as cv
img = cv.imread('/Users/melkor/Documents/datasets/Plagueworks/train/img/img_51_1_a.png',0)
img = cv.resize(img, (128,128))
edges = cv.Canny(img,100,200)

boundary = torch.tensor(edges).unsqueeze(0) /255.

import numpy as np
import math
num_edges = 100

import matplotlib.pyplot as plt
import torchvision
def visualize_image_grid(images, row, save_name = "image_grid"):
    plt.figure(save_name, frameon = False, figsize = (10,10));plt.cla()
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    comps_grid = torchvision.utils.make_grid(images,normalize=True,nrow=row).permute([1,2,0])

    plt.imshow(comps_grid.cpu().detach().numpy())
    plt.savefig("outputs/{}.png".format(save_name), bbox_inches='tight', pad_inches=0)

def visualize_affinities(indices, affinity, im = None, num_edges = 10):
    _, B, N, K = indices.shape
    W = int(math.sqrt(N))
    indices = indices.reshape([3, B, N * K])
    rand_idx = np.random.randint(0,N * K, [num_edges])
    rand_idx = torch.tensor(rand_idx).int()
    aff = affinity.reshape(B, N * K)
    if im is None: im = torch.zeros([B,3, N])
    plt.subplot(121)
    plt.imshow(im.reshape([B,3,W,W])[0].permute(1,2,0))
    for idx in rand_idx:
        u_idx = indices[1,0,idx]
        v_idx = indices[2,0,idx]
        ux, uy = u_idx // W, u_idx % W
        vx, vy = v_idx // W, v_idx % W
        color = "green" if aff[0,idx] > 0.0 else "red"
        plt.plot([uy,vy],[ux,vx], c = color)
    plt.subplot(122)
    plt.imshow(im.reshape([B,3,W,W])[0].permute(1,2,0))

class BoundaryGrouper(nn.Module):
    def __init__(self, K = 7, resolution = (128,128)):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        W, H = resolution
        self.W, self.H = W, H
        long_range_ratio = 0.2

        """generate local indices to build a complete graph"""
        self.K = K
        local_indices = generate_local_indices([W,H], K)
        self.register_buffer(f"indices_{W}x{H}", local_indices)
        self.num_long_range = int(K**2 * long_range_ratio)

        """extract segments from the affinity graph using the graph propagation and competition"""
        self.propagation = GraphPropagation(num_iters = 232)
        self.competition = Competition(num_masks = 30, mask_thresh=0.5, compete_thresh=0.2)

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
    
    def forward(self, boundary):
        B = 1
        indices = self.get_indices(resolution = (self.W, self.H),B = B)
        _, _, N, K = indices.shape
        from_indices = indices[1,:,:,:].reshape([B, N * K])
        to_indices = indices[2,:,:,:].reshape([B, N * K])
        
        affinity = torch.ones([B, N, K]) * 10

        affinity, boundary_inds = self.restrict(from_indices, to_indices, affinity, boundary)
        print("object affinity:",affinity.max(), affinity.min())
        aff_masks, agents, alive, prop_maps = self.extract_segments(indices, affinity.reshape([B, N, K])) 
        aff_masks = torch.einsum("bwhn,bnd->bwhn", aff_masks, alive)
        return aff_masks, prop_maps, affinity, indices, boundary_inds

    def indice_to_coords(self, indices, resolution = None):
        B, N = indices.shape
        if resolution is None: resolution = (self.W, self.H)
        W, H = resolution
        indices = indices.int()
        x_val = (indices // H).unsqueeze(-1).float()
        
        y_val = (indices % W).unsqueeze(-1).float()

        return torch.cat([x_val, y_val], dim = -1)

    def coords_to_indice(self, coords, resolution = None):
        B, N, _, _ = coords.shape
        if resolution is None: resolution = (self.W, self.H)
        W, H = resolution
        return coords[:,:,0,:] * W + coords[:,:,1,:]

    def restrict(self, from_indices, to_indices, affinity, boundary, L = 60):
        """
        Args:
            indices:  BxK locations of the the affinity indices
            affinity: BxK affinity indices both local and global
            boundary: BxWxH boundary density at each location
        Returns:
            restricted affinity that does not cross boundary
        """
        # [sample_inds on the line]
        W, H = self.W, self.H
        B, K = from_indices.shape
        from_indices = from_indices.reshape([B, K])
        to_indices = to_indices.reshape([B, K])
        from_coords = self.indice_to_coords(from_indices)
        to_coords = self.indice_to_coords(to_indices)
        offsets = to_coords - from_coords
        offsets = offsets[..., None].repeat(1,1,1,L).float()
        t_params = torch.linspace(0, 1, L).unsqueeze(0).unsqueeze(0).repeat(B, K, 1).to(from_indices.device)
        #print(W * H* self.K * self.K, K)
        #t_params[:,W*H*self.K*self.K:,:] = 0

        sample_inds = from_coords[..., None].repeat(1,1,1,L).float() + torch.einsum("bnds,bns->bnds", offsets, t_params)
        sample_inds = self.coords_to_indice(sample_inds.long())

        # [sum boundary density on these sample inds location ont he boundary density map]
        boundary_density = boundary.reshape([B * W * H])
        boundary_density = boundary_density[sample_inds]
        reduction_density = boundary_density.sum(dim = -1)

        # [reduce the affinity strength by the same amount]
        B, N, K = affinity.shape
        affinity = affinity  - reduction_density.reshape([B, N, K])
        return affinity, sample_inds

sgnet = BoundaryGrouper(K = 7)
masks, prop_maps, aff, inds, binds = sgnet(boundary)


visualize_affinities(inds, aff, num_edges = 5000)
if 1:
    plt.figure("vis masks")
    comps = to_cc_masks(masks[0])
    mask = torch.zeros([128,128])
    for i in range(comps.shape[-1]):
        mask[comps[:,:,i]] = i+1
    plt.imshow(mask)
#visualize_image_grid((masks.permute(0,3,1,2).unsqueeze(2)).repeat(1,1,3,1,1)[0], 6)


plt.figure("visualize boundary propagation", figsize = (10, 6))
plt.subplot(121)
plt.imshow(boundary[0].detach(), cmap = "bone")
plt.subplot(122)
plt.imshow(boundary[0].detach(), cmap = "bone")

for i,map in enumerate(prop_maps):
    plt.subplot(122)
    plt.cla()
    plt.imshow(map.reshape([128,128,-1])[:,:,:3])
    plt.text(0, 0, i)
    plt.pause(0.0001)

plt.show()