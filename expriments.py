'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-04-16 00:19:03
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-04-16 00:19:05
 # @ Description: This file is distributed under the MIT license.
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
from torch.utils.data import DataLoader, Dataset
from mvcl.utils import calculate_IoU_matrix, calculate_mIoU

import argparse

def ideal_grouper_experiment(model, dataset, idx = 0, epochs = 100, lr = 2e-4, mechansim = "attention"):
    """
    experiment setup:
    ground truth grounding features are taken as input and ground truth object segmentation is taken (affinity). We use only different
    mechanisms to calculate the affinity adapter. calculate the metrics and output the mIoU.
    training is for the affinity adapter only.
    """
    from datasets.sprites_base_dataset import SpritesBaseDataset
    return

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
def motion_affinity_training(model, dataset, idx = 0, epochs = 100, lr = 2e-4):
    """
    experient setup: the input dataset contains two connected frames and optical flow are precomputed
    choose a standard mvcl model and 
    """
    return

def demo_experiment(epochs = 1000):
    resolution = (128,128)
    batch_size = 3
    dataset = SpritesBaseDataset(resolution = resolution)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    
    for epoch in range(epochs):
        ious = []
        for sample in loader:
            img = sample["img"] 
            masks = sample["masks"]
            predict_masks = torch.randn([masks.shape[0], resolution[0], resolution[1], 6]) > 0
            miou = calculate_mIoU(masks, predict_masks)
            ious.append(miou)
        sys.stdout.write(f"\nepoch:{epoch+1} mIoU:{float(sum(ious) / len(ious))}")

def evaluate_metrics(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size = 1)
    ious = []
    for sample in loader:
        imgs = sample["img"]
        target_masks = sample["masks"]

        predict_masks = model.predict_masks(imgs)["masks"]
        miou = calculate_mIoU(target_masks, predict_masks)
        ious.append(miou)
    sys.stdout.write(f"mIoU:{float(sum(ious)/ len(ious))}")
    return float(sum(ious)/ len(ious))

parser = argparse.ArgumentParser()
parser.add_argument("--expr_type",                        default = "demo")
parser.add_argument("--epochs",                           default = 100)
args = parser.parse_args()

# Example usage
if __name__ == "__main__":
    # Generate example masks
    from mvcl.utils import calculate_IoU_matrix, calculate_mIoU

    if args.expr_type == "demo":
        demo_experiment(args.epochs)