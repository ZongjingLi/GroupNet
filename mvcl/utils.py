'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-03-19 09:10:19
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-03-19 09:10:21
 # @ Description: This file is distributed under the MIT license.
 '''

import torch
import torch.nn as nn

class SegmentationMetric:
    def __init__(self, 
                    metrics,
                    background_value = 0,
                    min_pixels = 1, # the minimum pixels to be considered an true objecty
                    ):
        self.metrics = metrics
        self.background_value = background_value
    
    @property
    def background_value(self):
        return self.background_value

    @staticmethod
    def mIoU(pred_mask, gt_mask, min_gt_size = 1):
        """calculate iou over two boolean masks"""
        overlap = (pred_mask & gt_mask).sum().astype(float)
        return 

    def calculate_metrics(self):
        return
    
    def calculate_greedy_mIoU(self, target_masks, predict_masks):
        """
        input:
            target_masks: BxNxWxH
            predict_masks: BxMxWxH
        output:
            single mIoU average over the batch
        """
        pass

def calculate_IoU_matrix(pred_masks, target_masks):
    """
    Calculate Intersection over Union (IoU) for a batch of predicted masks and target masks.
    
    Args:
        pred_masks (torch.Tensor): Predicted masks of shape (B, W, H, N)
        target_masks (torch.Tensor): Target masks of shape (B, W, H, M)
    
    Returns:
        iou (torch.Tensor): Intersection over Union (IoU) of shape (B, N, M)
    """
    if len(pred_masks.shape) == 3: pred_masks = expand_mask(pred_masks)
    batch_size, _, _, num_pred_masks = pred_masks.size()
    _, _, _, num_target_masks = target_masks.size()
    
    # Flatten the masks to 2D tensors (B, W*H, N) and (B, W*H, M)
    pred_masks_flat = pred_masks.view(batch_size, -1, num_pred_masks)
    target_masks_flat = target_masks.view(batch_size, -1, num_target_masks)
    
    # Compute intersection
    intersection = torch.sum(torch.min(pred_masks_flat.unsqueeze(3), target_masks_flat.unsqueeze(2)), dim=1)
    # Compute union
    union = torch.sum(torch.max(pred_masks_flat.unsqueeze(3), target_masks_flat.unsqueeze(2)), dim=1)
    # Avoid division by zero
    epsilon = 1e-7
    
    # Compute IoU
    iou = (intersection + epsilon) / (union + epsilon)
    
    return iou  # Mean IoU across the batch

def calculate_mIoU(pred_masks, target_masks):
    """
    Calculate Intersection over Union (IoU) for a batch of predicted masks and target masks.
    
    Args:
        pred_masks (torch.Tensor): Predicted masks of shape (B, W, H, N)
        target_masks (torch.Tensor): Target masks of shape (B, W, H, M)
    
    Returns:
        iou (torch.Tensor): Mean Intersection over Union (IoU)
    """
    mIoU_matrix = calculate_IoU_matrix(pred_masks, target_masks)
    return mIoU_matrix.max(dim = -1).values.mean()

def expand_mask(mask, num_classes = None):
    """
    Expand a mask tensor of shape (B, W, H) to shape (B, W, H, N) where N is the number of classes.
    
    Args:
        mask (torch.Tensor): Mask tensor of shape (B, W, H) with class labels as integers
        num_classes (int): Number of classes
        
    Returns:
        expanded_mask (torch.Tensor): Expanded mask tensor of shape (B, W, H, N)
    """
    batch_size, width, height = mask.size()
    if num_classes is None: num_classes = int(mask.max())
    
    # Create an empty tensor of shape (B, W, H, N)
    expanded_mask = torch.zeros([batch_size, width, height, num_classes], dtype=torch.float32)
    
    # Iterate over each class and set corresponding elements in the mask tensor
    for i in range(num_classes):
        # Create a binary mask where class i is set to 1 and all other classes are set to 0
        class_mask = (mask == i).float()
        expanded_mask[:, :, :, i] = class_mask
    
    return expanded_mask


def gather_annotated_masks(part_masks, scene_dict, device = "cuda:0" if torch.cuda.is_available() else "cpu"):
    B, W, H = part_masks.shape
    K = len(scene_dict)
    masks = torch.zeros([B, W, H, K], device = device)
    masks_dict = {}

    for key in scene_dict:
        masks_dict[key] = torch.zeros([B,W,H], device = device)
        part_ids_binds = scene_dict[key]
        for b, part_ids in enumerate(part_ids_binds):
            ids = []
            for i in part_ids[1:-1].split(","):
                if len(i) > 0: ids.append(int(i))
            for id in ids:
                #print(masks_dict[key][b].shape)
                #print(part_masks[b,:,:].shape)
                masks_dict[key][b][part_masks[b,:,:]==id] = 1
    return masks_dict
