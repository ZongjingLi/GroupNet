
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
parser.add_argument("--perception_model_name", default = "SceneNet")
parser.add_argument("--resolution",            default = (128,128))
parser.add_argument("--max_num_masks",         default = 10,       type = int)
parser.add_argument("--backbone_feature_dim",  default = 132)
parser.add_argument("--kq_dim",                default = 64)
parser.add_argument("--channel_dim",           default = 3)

""" [Physics Model]"""
parser.add_argument("--physics_model_name",    default = "PropNet")
parser.add_argument("--state_dim",             default = 2 + 2,    type = int,     help = "the dynamic state dim, normally it's the x + v")
parser.add_argument("--attr_dim",              default = 5,        type = int,     help = "the number of attributes for each particle")
parser.add_argument("--relation_dim",          default = 2,        type = int,     help = "the number of relational features between particles")
parser.add_argument("--effect_dim",            default = 32,       type = int,     help = "the effect propagation dim")
parser.add_argument("--num_itrs",              default = 7,        type = int)

config = parser.parse_args(args = [])