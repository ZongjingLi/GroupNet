from sqlite3 import connect
from mvcl.percept import propagation
from mvcl.percept.propagation import *
from mvcl.percept.competition import *
from mvcl.percept.metanet import generate_local_indices, MetaNet, local_to_sparse_global_affinity, logit
from mvcl.config import config

from datasets.playroom_dataset import PlayroomDataset, DataLoader
from torchvision import transforms

dataset = PlayroomDataset(True)
loader = DataLoader(dataset, batch_size = 1, shuffle = True)
for sample in loader:
    sample;break;


dx, dy = (33,43)
B = 1
W, H ,C  = 128,128,3
resolution = (W, H, C)

GPM = GraphPropagation(num_iters = 132, inhibit=1, excite=1, project=0, adj_thresh = 0.5)
extractor = Competition(num_masks = 32, mask_thresh=0.5, mask_beta = 10, num_competition_rounds=3)

K = 11; D = 32
num_long_range =  int(K * K * 0.15) #1024 - K**2
N = W * H
locals = generate_local_indices([W,H], K).long()

if False:
    masks = torch.zeros([1,W, H, 1])
    masks[:,20:20+dx,23:23+dy] = 1
    masks[:,65:65+dx,70:70+dx] = 2
else:
    img = sample["img1"]
    masks = sample["gt_segment"]
    Wi, Hi = (128, 128)
    img = transforms.Resize([Wi, Hi])(img)
    masks = transforms.Resize([Wi, Hi])(masks).unsqueeze(-1)
    print(img.shape, masks.shape)

def kalescope_propgation(logits, indices, resolution = (W,H)):
    W, H = resolution
    B, N, K = logits.shape
    h0 = torch.FloatTensor(B, N, D).normal_()
    
    adj = torch.softmax(logits, dim = -1)
    adj = adj / torch.max(adj, dim = -1, keepdim = True)[0].clamp(min = 1e-12)

    adj = local_to_sparse_global_affinity(adj, indices.long(), sparse_transpose= True)

    prop_maps = GPM.forward(h0.detach(), adj.detach())
    prop_map = prop_maps[-1]
    prop_map = prop_map.reshape([B, W, H, D])

    masks, agents, alive, phenotypes, _ = extractor(prop_map)
    return masks, alive, prop_maps

def get_indices():
    B = 1
    indices = locals
    v_indices = torch.cat([
                indices, torch.randint(H * W, [B, H*W, num_long_range])
            ], dim = -1).unsqueeze(0)

    _, B, N, K = v_indices.shape # K as the number of local indices at each grid location

    """Gather batch-wise indices and the u,v local features connections"""
    u_indices = torch.arange(W * H).reshape([1,1,W*H,1]).repeat(1,B,1,K)
    batch_inds = torch.arange(B).reshape([1,B,1,1]).repeat(1,1,H*W,K)

    indices = torch.cat([
                batch_inds, u_indices, v_indices
            ], dim = 0)
            # [3, B, N, K]
    return indices

def inference(sample_inds, segment_targets):
    e = 0.1
    lK = sample_inds.shape[-1]
    
    segment_targets = segment_targets.reshape([B,N]).unsqueeze(-1).long().repeat(1,1,lK)

    # spatial proximity restriction

    if sample_inds is not None:
        samples = torch.gather(segment_targets,1, sample_inds[2,...]).squeeze(-1)
    else:
        samples = segment_targets.permute(0, 2, 1)
    targets = segment_targets == samples

    #targets = targets / (torch.sum(targets, dim = -1, keepdim = True) + 1e-9)

    connectivity = logit(targets)

    return connectivity

def object_inference(region, connectivity):
    return 

indices = get_indices()
logits = inference(indices, masks.reshape(1,W*H,1))

masks, alive, prop_maps = kalescope_propgation(logits, indices)
alive = alive.reshape([-1])


plt.figure("kalescope")
for i,prop_map in enumerate(prop_maps):
    map = prop_map.reshape([W, H, D])
    plt.imshow(map.detach()[...,-3:])
    plt.text(0,0,i)
    plt.pause(0.001)
    plt.cla()

counter = 0
nums = int(alive.sum())
for i in range(alive.shape[0]):
    if alive[i]:
        counter += 1
        plt.figure("masks", figsize = (12, 3))
        plt.subplot(1,nums,counter)
        plt.imshow(masks[0,:,:,i]* alive[i])
        plt.axis(False)
        plt.figure("region", figsize = (12, 3))
        plt.subplot(1,nums,counter)
        plt.axis(False)
        comp_mask = masks[0,:,:,i]* alive[i]
        im = (comp_mask * img[0]).permute(1,2,0) / 256
        im = torch.cat([im, comp_mask.unsqueeze(-1)], dim = -1)
        plt.imshow(im)
plt.show()
