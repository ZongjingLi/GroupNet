'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-03-19 08:57:21
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-03-19 08:57:24
 # @ Description: This file is distributed under the MIT license.
 '''

import torch
import torch.nn as nn
from rinarak.logger import get_logger, set_logger_output_file

def evaluate_metrics(model, loader, expr_name = "KFT"):
    set_logger_output_file(f"logs/{expr_name}_eval.txt")
    eval_logger = get_logger(f"{expr_name}_eval")
    miou = 0.
    loss = 0.
    
    eval_logger.critical("prepare to evaluate the miou and other metrics")

    for sample in loader:
        outputs = model(sample)
        curr_loss = outputs["loss"]
        loss += curr_loss

    eval_logger.critical(f"metrics calculated: miou:{miou} loss:{loss}")
    return miou, loss