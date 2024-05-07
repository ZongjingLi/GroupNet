'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-04-16 00:19:03
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-04-16 00:19:05
 # @ Description: This file is distributed under the MIT license.
 这是一条真实的信息
 '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
from datetime import date

from rinarak.utils.tensor import gather_loss

from mvcl.utils import gather_annotated_masks
from mvcl.model import MetaVisualLearner
from mvcl.config import config

from rinarak.logger import get_logger, set_logger_output_file
from datasets.sprites_base_dataset import SpritesBaseDataset
from datasets.tdw_dataset import TDWRoomDataset


from torch.utils.data import DataLoader, Dataset
from mvcl.utils import calculate_IoU_matrix, calculate_mIoU, expand_mask, to_onehot_mask

from rinarak.logger import set_output_file, get_logger
from rinarak.utils.os import save_json

from tqdm import tqdm
import argparse

local = not torch.cuda.is_available()
syq_path = "/Users/melkor/Documents/datasets"
wys_path = "/data3/guofang/Meta/Benchmark/MultiPaperQA/wys_try/datasets"
dataset_dir = syq_path if local else wys_path

mvcl_dir = ""

def ideal_grouper_experiment(model, dataset, idx = 0, epochs = 5000, lr = 2e-4, batch_size = 2, mechansim = "attention"):
    """
    experiment setup:
    ground truth grounding features are taken as input and ground truth object segmentation is taken (affinity). We use only different
    mechanisms to calculate the affinity adapter. calculate the metrics and output the mIoU.
    training is for the affinity adapter only.
    """
    set_output_file(f"logs/expr_concept_demo_train.txt")
    train_logger = get_logger("expr_concept_train")

    W, H = config.resolution # the resolution of the input images

    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        ious = []
        itrs = 0
        for sample in loader:
            ims = sample["img"]
            target_masks = sample["masks"]
            #annotated_masks = gather_annotated_masks(target_masks, sample["scene"])
            #auguments = {"annotated_masks": annotated_masks} 
            auguments = {}
            itrs += ims.shape[0] # add a batch num of items

            """calculate the ideal object affinity function using ground truth"""
            outputs=model.calculate_object_affinity(
            ims, # input image BxCxWxH
            target_masks, # ground truth object segment
            working_resolution=(W,H),verbose=False, augument = auguments)

            """(optional) extract the ground truth segments and calculate the iou"""
            obj_affinity = outputs["affinity"]
            indices = outputs["indices"]
            predict_masks, _ ,alive, prop_maps = model.extract_segments(obj_affinity, indices)
            predict_masks = torch.einsum("bwhn,bnd->bwhn", predict_masks, alive)

            miou = calculate_mIoU(target_masks.to("cpu"), predict_masks.to("cpu"))
            ious.append(miou)
            
            """gather loss and propagate"""
            loss = gather_loss(outputs["loss"])["adapter_loss"]
            epoch_loss += loss

            """backward propagate and optimize the target loss"""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            sys.stdout.write(f"\repoch:{epoch+1} [{itrs}/{len(dataset)}] loss:{loss} iou:{miou}")

        train_logger.critical(f"epoch:{epoch+1} loss:{epoch_loss} : mIoU:{sum(ious)/len(ious)}")
        torch.save(model.state_dict(), "MetaVisualConceptLearner/checkpoints/concept_expr_prox128.ckpt")
    return model

def autoencoder_grouper_experiment(model, dataset, idx = 0, epochs = 100, lr = 2e-4):
    """
    experiment setup:
    choose an autoencoder model to obtain the intermediate feature map. Use the intermediate results to calculate the perceptual
    grouping affinity (using dot product attention). calculate the metrics and output the mIoU.
    training is for the autoencoder only.
    """
    set_logger_output_file("logs/expr-autoencoder-grouper:{idx}")
    return

"""the true story lies after this line"""
def motion_affinity_training(model, dataset, batch_size = 2, epochs = 100, lr = 2e-4):
    """
    experient setup: the input dataset contains two connected frames and optical flow are precomputed
    choose a standard mvcl model and 
    """
    model.toggle_component_except("spelke") # freeze all the components except for motion
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    ious = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for sample in loader:
            ims = sample["img"]
            target_masks = sample["masks"]
            """strongly recommend to use precomputed optical flow to speed up training."""
            outputs=model.calculate_object_affinity(
            ims, # input image BxCxWxH
            target_masks, # ground truth object segment
            working_resolution=(W,H),verbose=False, augument = auguments)

            """(optional) extract the ground truth segments and calculate the iou"""
            obj_affinity = outputs["affinity"]
            indices = outputs["indices"]
            predict_masks, _ ,alive, prop_maps = model.extract_segments(obj_affinity, indices)
            predict_masks = torch.einsum("bwhn,bnd->bwhn", predict_masks, alive)

            miou = calculate_mIoU(target_masks.to("cpu"), predict_masks.to("cpu"))
            ious.append(miou)
            
            """gather loss and propagate"""
            loss = gather_loss(outputs["loss"])["adapter_loss"]
            epoch_loss += loss

            """backward propagate and optimize the target loss"""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def visual_feature_affinity_training(model, dataset, batch_size = 2, epochs = 100, lr = 2e-4):
    """
    experient setup: the input dataset contains two connected frames and optical flow are precomputed
    choose a standard mvcl model and 
    """
    model.toggle_component_except("albedo") # freeze all the components except for motion
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    for epoch in range(epochs):
        for sample in loader:
            pass
    return

def visual_feature_meta_training(model):
    pass

def evaluate_metrics(model, dataset, name = "expr"):
    model.eval()
    loader = DataLoader(dataset, batch_size = 1)
    ious = []
    itrs = 0
    split = dataset.split
    save_name = f"outputs/{name}/{split}/"
    accurates = [0]
    
    for sample in loader:
        
        imgs = sample["img"]
        target_masks = sample["masks"]
        albedo_map = imgs
        auguments = {"annotated_masks":{"albedo": albedo_map}}
        predict_masks = model.predict_masks(imgs, augument = auguments)["masks"]

        miou = calculate_mIoU(target_masks.to("cpu"), predict_masks.to("cpu"))
        ious.append(miou)
        sys.stdout.write(f"\r[{itrs}/{len(dataset)}]iou:{float(sum(ious)/ len(ious))}")

        evaluate_data_bind = {
            "iou":float(miou),
            "queries": ["in the scene?", "is there any green object in the scene"],
            "programs":["(scene $0)", "(exists (green $0))"],
            "answers":["null", "yes"],
            "gt_answers": ["null", "yes"], 
        }
        save_json(evaluate_data_bind, save_name + f"{itrs}_eval.json")

        #plt.imshow()
        plt.imsave(save_name + f"{itrs}_img.png", np.array(imgs[0].cpu().detach().permute(1,2,0)))
        plt.cla()
        plt.axis("off")
        plt.imshow(to_onehot_mask(predict_masks.cpu().detach())[0])
        plt.savefig(save_name + f"{itrs}_mask.png", bbox_inches = "tight")
        itrs += 1

    sys.stdout.write(f"\rmIoU:{float(sum(ious)/ len(ious))}")
    overall_data = {"miou":  float(sum(ious)/ len(ious)), "accuracy": sum(accurates)/len(accurates)}
    save_json(overall_data, save_name + "overall.json")
    return float(sum(ious)/ len(ious))

parser = argparse.ArgumentParser()
parser.add_argument("--expr_type",                        default = "demo")
parser.add_argument("--epochs",                           default = 100)
args = parser.parse_args()

# Example usage
if __name__ == "__main__":
    from rinarak.domain import load_domain_string, Domain
    domain_parser = Domain("mvcl/base.grammar")

    meta_domain_str = ""
    with open(f"domains/demo_domain.txt","r") as domain:
        for line in domain: meta_domain_str += line
    domain = load_domain_string(meta_domain_str, domain_parser)

    # Generate example masks
    from mvcl.utils import calculate_IoU_matrix, calculate_mIoU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.expr_type == "demo":
        resolution = (64,64)
        config.resolution = resolution
        model = MetaVisualLearner(domain, config)
        #model.load_state_dict(torch.load("MetaVisualConceptLearner/checkpoints/concept_expr.ckpt"))
        model.clear_components()
        model.add_spatial_affinity()
        model.add_affinities(["albedo"])
        model = model.to(device)
        dataset = TDWRoomDataset(resolution = resolution, root_dir = dataset_dir, split = "train")
        evaluate_metrics(model, dataset)

    if args.expr_type == "concept_demo":
        resolution = (64,64)
        config.resolution = resolution
        model = MetaVisualLearner(None, config)
        model.add_affinities(["albedo"])
        model = model.to(device)
        #model.load_state_dict(torch.load("MetaVisualConceptLearner/checkpoints/concept_expr.ckpt"))
        dataset = TDWRoomDataset(resolution = resolution, root_dir = dataset_dir)
        ideal_grouper_experiment(model, dataset)