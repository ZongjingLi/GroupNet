import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from rinarak.utils.tensor import gather_loss

from mvcl.utils import gather_annotated_masks
from mvcl.model import MetaVisualLearner
from mvcl.config import config

from datasets.sprites_base_dataset import SpritesBaseDataset
from datasets.tdw_dataset import TDWRoomDataset

resolution = (64,64)
W, H = resolution
B = 1
#vocab = ["red", "blue", "green", "circle", "diamond", "square"]
#vocab = []
local = True
dataset_dir = "/Users/melkor/Documents/datasets" if local else "datasets"


"""load the checkpoint data for the demo domain"""
domain = None
metanet = MetaVisualLearner(domain, config)
#metanet.load_state_dict(torch.load("checkpoints/concept_expr.ckpt"))
metanet.load_state_dict(torch.load("checkpoints/concept_expr_prox.ckpt", map_location="cpu"))
#metanet.add_affinities(vocab)

#metanet.load_state_dict(torch.load("checkpoints/concept_expr.ckpt"))

"""gather a sample from the dataset"""
#dataset = TDWRoomDataset(resolution = (W, H))
dataset = TDWRoomDataset(resolution = resolution, root_dir = dataset_dir)
loader = DataLoader(dataset, batch_size = B, shuffle = True)

for sample in loader: break
ims = sample["img"]
targets = sample["masks"]
#annotated_masks = gather_annotated_masks(targets, sample["scene"])
#auguments = {"annotated_masks": annotated_masks} 
auguments = {}

"""show the images and masks, components and the gathered masks"""
b = 0
plt.figure("input-ims vs gt-masks")
plt.subplot(121)
plt.imshow(ims[0].permute(1,2,0))
plt.subplot(122)
plt.imshow(targets[0])

#plt.figure("visualize concept component masks", figsize = (8,4))
#for i,name in enumerate(vocab):
#    plt.subplot(1, len(vocab), 1 + i)
#    plt.title(name)
#    plt.imshow(annotated_masks[name][b], cmap = "bone")

"""calculate the actual objects segmented by the prediceted affinity"""
metanet.freeze_components()
outputs=metanet.calculate_object_affinity(
    ims,
    targets,
    working_resolution=(W,H),verbose=True, augument = auguments)

indices = outputs["indices"]
logits = outputs["affinity"]

masks, agents, alive, prop_maps = metanet.grouper.compute_masks(logits, indices)
alive = alive.reshape([B, -1])

plt.figure("kalescope")
for i,prop_map in enumerate(prop_maps):
    map = prop_map.reshape([W, H, -1])
    plt.imshow(map.detach()[...,-3:])
    plt.text(0,0,i)
    plt.pause(0.001)
    plt.cla()

plt.figure("visualize objects segment", figsize = (8,9))

num_rows = 6
for i in range(alive[b].shape[0]):
    plt.subplot(num_rows, alive[b].shape[0] // num_rows, i + 1)
    plt.imshow(masks[b,:,:,i] * alive[b,i])

plt.show()