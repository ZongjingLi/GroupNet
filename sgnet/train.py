'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-05-31 21:52:01
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-05-31 21:52:07
 # @ Description: This file is distributed under the MIT license.
'''
import sys
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image
from .utils import calculate_IoU_matrix, calculate_mIoU, expand_mask, to_onehot_mask
from tqdm import tqdm

def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1],
        ]
    )
    batch = transforms(batch)
    return batch

def train(model, dataset, batch_size = 2, epochs = 100, lr = 2e-4, save_name = "states", device = None):
    """
    Args:
        model: the symbolic grouping networks model that use language and motion cues to learn affinity
    Returns:
        the trained model is returned, it is also saved in the checkpoints
    """
    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    model.to(device)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    """pure motion trainig requires optical flow as input, if the optical flow map is not precomputed it is esitmated"""
    flow_predicter = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    flow_predicter = flow_predicter.eval()

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_samples = 0
        miou = []
        for sample in loader:
            img1 = sample["img1"].to(device)
            img2 = sample["img2"].to(device)
            gt_masks = sample["masks"]
            num_samples += img1.shape[0] # add the batch number ot the total number of the samples iterated

            """predict the optical flow and motion direction as cues"""
            list_of_flows = flow_predicter(
                preprocess(img1).to(device),
                preprocess(img2).to(device))
            predicted_flows = list_of_flows[-1]
            motion_strength = torch.norm(predicted_flows.permute(0,2,3,1).float() , dim = -1) 

            """collect the affinity cues using various sources (motion, direction, albedo etc.)"""
            cues = {
            "movable": (motion_strength > 0.5).float()
            }
            
            outputs = model(img1, cues, verbose = False)
            predict_masks = outputs["masks"]["movable"]
            losses = outputs["loss"]
            
            """collect losses from the model output for object affinity learning."""
            working_loss = 0.0
            for loss_name in losses:
                working_loss += losses[loss_name]
            sample_miou = calculate_mIoU(gt_masks.to("cpu"), predict_masks.to("cpu"))
            miou.append(sample_miou.cpu().detach().numpy())
            
            """start the optimization of the overall loss of object adapter and affinity cues."""
            optimizer.zero_grad()
            working_loss.backward()
            optimizer.step()
            epoch_loss += working_loss.cpu().detach()
            sys.stdout.write(f"\repoch:{epoch+1} [{num_samples}/{len(dataset)}] loss:{working_loss.cpu().detach().numpy()} mIoU:{sum(miou)/len(miou)}")
        torch.save(model.state_dict(), f"checkpoints_/{save_name}.pth")
    sys.stdout.write(f"\nepoch {epoch+1} completed with loss {epoch_loss}")

def evaluate(model, dataset, device = None):
    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size = 1, shuffle = True)

    """pure motion trainig requires optical flow as input, if the optical flow map is not precomputed it is esitmated"""
    flow_predicter = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    flow_predicter = flow_predicter.eval()
    
    num_samples = 0
    miou = []
    for sample in tqdm(loader):
        img1 = sample["img1"].to(device)
        img2 = sample["img2"].to(device)
        gt_masks = sample["masks"]
        num_samples += img1.shape[0] # add the batch number ot the total number of the samples iterated

        """predict the optical flow and motion direction as cues"""
        list_of_flows = flow_predicter(
                preprocess(img1),
                preprocess(img2))
        predicted_flows = list_of_flows[-1]
        motion_strength = torch.norm(predicted_flows.permute(0,2,3,1).float() , dim = -1) 

        """collect the affinity cues using various sources (motion, direction, albedo etc.)"""
        cues = {
            "movable": (motion_strength > 0.5).float()
        }
            
        outputs = model(img1, cues, verbose = False)
        predict_masks = outputs["masks"]["movable"]

        sample_miou = calculate_mIoU(gt_masks.to("cpu"), predict_masks.to("cpu"))
        miou.append(sample_miou.cpu().detach().numpy())    

    sys.stdout.write(f"evaluation completed with miou : {sum(miou)/len(miou):.3f}")


def save_batch_result(gt_img, gt_segment, predict_segment):
    return