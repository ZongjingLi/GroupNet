'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-05-08 10:38:32
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-05-08 10:38:33
 # @ Description: This file is distributed under the MIT license.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from mvcl.benchmarks.slot_attention import SlotAttentionParser
from mvcl.benchmarks.monet import Monet
from mvcl.utils import calculate_IoU_matrix, calculate_mIoU, expand_mask, to_onehot_mask

from rinarak.logger import set_output_file, get_logger
from rinarak.utils.os import save_json

import time
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

local = not torch.cuda.is_available()
syq_path = "/Users/melkor/Documents/datasets"
wys_path = "/data3/guofang/Meta/Benchmark/MultiPaperQA/wys_try/datasets"
dataset_dir = syq_path if local else wys_path

def train_warmup(model, train_set, epochs = 1000):
    decay_steps = 100000
    warmup_steps = 10000
    decay_rate = 0.5
    batch_size = 4
    learning_rate = 2e-4
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    criterion = nn.MSELoss()

    params = [{'params': model.parameters()}]

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                        shuffle=True, num_workers=1)

    optimizer = torch.optim.Adam(params, lr= learning_rate)

    start = time.time()
    i = 0
    for epoch in range(epochs):
        model.train()

        total_loss = 0

        for sample in tqdm(train_dataloader):
            i += 1

            if i < warmup_steps:
                learning_rate = learning_rate * (i / warmup_steps)
            else:
                learning_rate = learning_rate

            learning_rate = learning_rate * (decay_rate ** (
                i / decay_steps))

            optimizer.param_groups[0]['lr'] = learning_rate
        
            image = sample['img'].to(device)
            outputs = model(image)
            recon_combined = outputs["full_recons"]
            recons = outputs["recons"]
            masks = outputs["masks"]

            loss = criterion(recon_combined, image)
            total_loss += loss.item()

            del recons, masks

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), "checkpoints/slot.pth")
        total_loss /= len(train_dataloader)

def train_monet(model, dataset, lr = 2e-4):
    epochs = 1000
    batch_size = 4
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=1)
    params = [{'params': model.parameters()}]
    optimizer = torch.optim.Adam(params, lr= lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for sample in tqdm(train_dataloader):
            images = sample["img"]
            outputs = model(images)

            loss = outputs["loss"].mean()
            masks = outputs["masks"]
            recons = outputs["reconstructions"]
            total_loss += loss.cpu().detach()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        sys.stdout.write(f"epoch:{epoch} loss:{total_loss}")
        torch.save(model.state_dict(), "checkpoints/monet.pth")
    
def evaluate_benchmark(model, dataset, name):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
    ious = []
    itrs = 0
    split = dataset.split
    save_name = f"outputs/{name}/{split}/"
    accurates = [0]

    for sample in loader:
        imgs = sample["img"]
        target_masks = sample["masks"]

        outputs = model(imgs)
        predict_masks = outputs["masks"].squeeze(-1).permute(0,2,3,1)

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

if __name__ == "__main__":
    from datasets.tdw_dataset import TDWRoomDataset
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    name = "slot_eval"

    resolution = (128,128)
    if name == "slot_attention":
        model = SlotAttentionParser(8, 100, 7)
        model = model.to(device)

        dataset = TDWRoomDataset(resolution = resolution, root_dir = dataset_dir, split = "train")    
        train_warmup(model, dataset)
    if name == "monet":
        model = Monet(resolution[0], resolution[1], 3)
        model = model.to(device)
        dataset = TDWRoomDataset(resolution = resolution, root_dir = dataset_dir, split = "train")
        train_monet(model, dataset)
    
    if name == "monet_eval":
        model = Monet(resolution[0], resolution[1], 3)
        #model = SlotAttentionParser(8, 100, 7)
        model = model.to(device)
        model.load_state_dict(torch.load("checkpoints/monet.pth"))
        dataset = TDWRoomDataset(resolution = resolution, root_dir = dataset_dir, split = "train")
        evaluate_benchmark(model, dataset, name = "monet")
    
    if name == "slot_eval":
        #model = SlotAttentionParser(resolution[0], resolution[1], 3)
        model = SlotAttentionParser(8, 100, 7)
        model = model.to(device)
        model.load_state_dict(torch.load("checkpoints/slot.pth"))
        dataset = TDWRoomDataset(resolution = resolution, root_dir = dataset_dir, split = "train")
        evaluate_benchmark(model, dataset, name = "slot")