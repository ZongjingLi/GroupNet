
import torch
import torch.nn as nn

import argparse

from rinarak.program import *
from rinarak.utils.tensor import logit
from rinarak.knowledge.executor import CentralExecutor

from .percept.metanet import MetaNet

parser = argparse.ArgumentParser()

parser.add_argument("--device",                 default = "cuda:0" if torch.cuda.is_available() else "cpu")

""" [Concept Model]"""
parser.add_argument("--concept_type",          default = "cone")
parser.add_argument("--object_dim",            default = 100)
parser.add_argument("--concept_dim",           default = 100)
parser.add_argument("--temperature",           default = 0.2)
parser.add_argument("--entries",               default = 100)
parser.add_argument("--method",                default = "uniform")
parser.add_argument("--center",                default = [-0.25,0.25])
parser.add_argument("--offset",                default = [-0.25,0.25])
parser.add_argument("--domain",                default = "demo")

"""[Perception Model]"""
parser.add_argument("--perception_model_name", default = "MetaNet")
parser.add_argument("--resolution",            default = (128,128))
parser.add_argument("--max_num_masks",         default = 10,       type = int)
parser.add_argument("--backbone_feature_dim",  default = 132)
parser.add_argument("--kq_dim",                default = 64)
parser.add_argument("--channel_dim",           default = 3)

config = parser.parse_args(args = [])