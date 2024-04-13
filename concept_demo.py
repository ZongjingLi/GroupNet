import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

"""Build the pure visual model"""
from mvcl.percept.metanet import MetaNet
from mvcl.model import MetaVisualLearner, config
from mvcl.custom import ObjectAffinityFeatures
from datasets.sprites_base_dataset import SpritesBaseDataset
from rinarak.utils.tensor import gather_loss
from rinarak.logger import get_logger, set_output_file
from mvcl.utils import gather_annotated_masks
from tqdm import tqdm
import sys

#affinity = ObjectAffinityFeatures(3, 100)

"""make the dataset"""
W, H = (64,64)
config.resolution = (W,H)
metanet = MetaVisualLearner(None, config)

metanet.add_affinities(["red", "blue", "green", "circle", "diamond", "square"])

epochs = 100
metanet.freeze_components()
dataset = SpritesBaseDataset(resolution = (W, H))
loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)
optimizer = torch.optim.Adam(metanet.parameters(), lr = 2e-4)


set_output_file(f"logs/expr_concept_demo_train.txt")
train_logger = get_logger("expr_concept_train")

for epoch in range(epochs):
    epoch_loss = 0.0
    for sample in tqdm(loader):
        ims = sample["img"]
        targets = sample["masks"]
        annotated_masks = gather_annotated_masks(targets, sample["scene"])
        auguments = {"annotated_masks": annotated_masks} 

        outputs=metanet.calculate_object_affinity(
        ims,
        targets,
        working_resolution=(W,H),verbose=False, augument = auguments)
        loss = gather_loss(outputs["loss"])["adapter_loss"]
        epoch_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_logger.critical(f"epoch:{epoch+1} loss:{epoch_loss}")
    torch.save(metanet.state_dict(), "checkpoints/concept_expr.ckpt")