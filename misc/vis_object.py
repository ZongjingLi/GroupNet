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

device = "cuda:0" if torch.cuda.is_available() else "cpu"
resolution = (128,128)
W, H = resolution
B = 1
#vocab = ["red", "blue", "green", "circle", "diamond", "square"]
#vocab = []

local = not torch.cuda.is_available()
syq_path = "/Users/melkor/Documents/datasets"
wys_path = "/data3/guofang/Meta/Benchmark/MultiPaperQA/wys_try/datasets"
dataset_dir = syq_path if local else wys_path

prefix = "" if local else "MetaVisualConceptLearner/"

"""load the checkpoint data for the demo domain"""
from rinarak.domain import load_domain_string, Domain
domain_parser = Domain("mvcl/base.grammar")

meta_domain_str = ""
with open(f"domains/demo_domain.txt","r") as domain:
        for line in domain: meta_domain_str += line
domain = load_domain_string(meta_domain_str, domain_parser)
config.resolution = resolution
metanet = MetaVisualLearner(domain, config)
#metanet.add_spatial_affinity()
#metanet.add_spelke_affinity()
flag = 0
if flag:
    metanet.add_affinities(["albedo"])
else:
    metanet.add_affinities(["spelke"])
    metanet.load_state_dict(torch.load("checkpoints/concept_expr_prox128.ckpt", map_location="cpu"))
#metanet.load_state_dict(torch.load("checkpoints/concept_expr.ckpt"))
#metanet.load_state_dict(torch.load(f"{prefix}checkpoints/concept_expr_prox128.ckpt", map_location="cpu"))
#metanet.add_affinities(vocab)
#torch.save(metanet.affinities["spelke"].state_dict(),"checkpoints/spelke_affinity.pth")
metanet = metanet.to(device)




"""gather a sample from the dataset"""
#dataset = SpritesBaseDataset(resolution = resolution)
dataset = TDWRoomDataset(name="TDWKitchen",split = "train",resolution = resolution, root_dir = dataset_dir)
loader = DataLoader(dataset, batch_size = B, shuffle = True)

for sample in loader: break
ims = sample["img"]
targets = sample["masks"]
if flag:
    albedo_map = sample["albedo"]

#annotated_masks = gather_annotated_masks(targets, sample["scene"])



if flag: auguments = {"annotated_masks":{"albedo": albedo_map}}
else: auguments = {}


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
    plt.pause(0.00001)
    plt.cla()

plt.figure("visualize objects segment", figsize = (8,9))

num_rows = 6

single_mask = torch.zeros([W,H])
for i in range(alive[b].shape[0]):
    single_mask[(masks[b,:,:,i] * alive[b,i]) > 0.1] = (i+1)
plt.imshow(single_mask.int())
plt.show()
