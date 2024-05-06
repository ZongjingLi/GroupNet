'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-04-23 07:52:32
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-04-23 07:52:34
 # @ Description: This file is distributed under the MIT license.
'''
import torch
from mvcl.config import config
from mvcl.custom import SpatialProximityAffinityCalculator,\
    SpelkeAffinityCalculator, GeneralAffinityCalculator
from mvcl.model import MetaVisualLearner

from datasets.sprites_base_dataset import SpritesBaseDataset
from datasets.tdw_dataset import TDWRoomDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from torchvision import transforms
import cv2

resolution = (128, 128)
B = 1
W, H = resolution

local = not torch.cuda.is_available()
syq_path = "/Users/melkor/Documents/datasets"
wys_path = "/data3/guofang/Meta/Benchmark/MultiPaperQA/wys_try/datasets"
dataset_dir = syq_path if local else wys_path

def group_affinity(metanet, ims = None):
    dataset = TDWRoomDataset(split = "train",resolution = resolution, root_dir = dataset_dir)
    loader = DataLoader(dataset, batch_size = B, shuffle = 1)
    for sample in loader: break
    if ims is None:
        ims = sample["img"]
        albedo = sample["albedo"]
    else:
        ims = ims
        albedo = ims
    targets = sample["masks"]
    
    #albedo = targets.unsqueeze(1)
    auguments = {"annotated_masks" : {"albedo": albedo} }

    plt.figure("input-ims vs gt-masks")
    plt.subplot(121)
    plt.imshow(ims[0].permute(1,2,0))
    plt.subplot(122)
    plt.imshow(targets[0])

    plt.figure("void illusion")
    gray = cv2.cvtColor(np.array(albedo[0].permute(1,2,0)) * 255, cv2.COLOR_RGB2GRAY) 
    blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.5)
    blurred = np.uint8(blurred)
    plt.subplot(121)
    plt.imshow(blurred)
    edges = cv2.Canny(blurred, 70, 135)
    plt.subplot(122)
    plt.imshow(np.array(edges))


    outputs=metanet.calculate_object_affinity(
        ims,
        working_resolution=(W,H),verbose=False, augument = auguments)

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
    b = 0
    for i in range(alive[b].shape[0]):
        single_mask[(masks[b,:,:,i] * alive[b,i]) > 0.1] = (i+1)
    plt.imshow(single_mask.int())
    plt.show()

if __name__ == "__main__":
    from datasets.sprites_base_dataset import normal_img
    from PIL import Image
    import numpy as np
    from rinarak.domain import load_domain_string, Domain
    domain_parser = Domain("mvcl/base.grammar")

    meta_domain_str = ""
    with open(f"domains/demo_domain.txt","r") as domain:
        for line in domain: meta_domain_str += line
    domain = load_domain_string(meta_domain_str, domain_parser)

    config.resolution = resolution
    
    metanet = MetaVisualLearner(domain, config)
    metanet.clear_components()
    metanet.add_spatial_affinity()
    
    metanet.add_affinities(["albedo"])
    #metanet.add_spelke_affinity()
    #metanet.affinities["spelke"] = torch.load("checkpoints/spelke_affinity.pth")
    #path = "outputs/illusion.png"
    #transform = transforms.Resize(resolution)
    #img = torch.tensor(np.array(Image.open(path).convert('RGB')))
    #img = transform(normal_img(img)).unsqueeze(0).float()
    img = None

    group_affinity(metanet, img)