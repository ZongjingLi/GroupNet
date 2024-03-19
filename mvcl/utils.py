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
