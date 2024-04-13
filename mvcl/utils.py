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