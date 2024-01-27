'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-01-24 17:50:12
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-01-24 18:24:56
 # @ Description: This file is distributed under the MIT license.
'''
from rinarak.domain import Domain, load_domain_string
from rinarak.program import Primitive

from mvcl.model import MetaVisualLearner
from mvcl.train import train, evaluate
from mvcl.config import config, argparse
from mvcl.primitives import *
import torch

dataset_dir = "/Users/melkor/Documents/datasets"

argparser = argparse.ArgumentParser()
argparser.add_argument("--expr_name",                   default = "KFT")
argparser.add_argument("--dataset_dir",                 default = dataset_dir)
# [Experiment configuration]
argparser.add_argument("--domain_name",                 default = "demo")
argparser.add_argument("--mode",                        default = "train")
argparser.add_argument("--dataset_name",                default = "sprites_base")

# [Training detail configurations]
argparser.add_argument("--epochs",                      default = 100)
argparser.add_argument("--batch_size",                  default = 2)
argparser.add_argument("--optimizer",                   default = "Adam")
argparser.add_argument("--lr",                          default = 1e-3)

# [Elaborated training details]
argparser.add_argument("--freeze_perception",           default = False)
argparser.add_argument("--freeze_knowledge",            default = False)

# [Save checkpoints at dir...]
argparser.add_argument("--ckpt_itrs",                   default = 100)
argparser.add_argument("--ckpt_dir",                    default = "checkpoints")
argparser.add_argument("--load_ckpt",                   default = False)

args = argparser.parse_args()

# [Parse the domain configuration]
domain_parser = Domain("mvcl/base.grammar")
meta_domain_str = ""
with open(f"domains/{args.domain_name}_domain.txt","r") as domain:
    for line in domain: meta_domain_str += line
domain = load_domain_string(meta_domain_str, domain_parser)

# [Build the Model]
args.load_ckpt = "checkpoints/KFT_backbone.pth"
model = MetaVisualLearner(domain, config)
import torch.nn as nn
color_mapper = nn.Linear(132,100)
shape_mapper = nn.Linear(132,100)

model.implementations["color"] = color_mapper
model.implementations["shape"] = shape_mapper
if args.load_ckpt: model.load_state_dict(torch.load(args.load_ckpt))

model.load_state_dict(torch.load(args.load_ckpt))


# [Pre-define some concept mapper]
color = Primitive.GLOBALS["color"]
color.value = lambda x: {**x, "features": x["model"].get_mapper("color")(x["features"])}
shape = Primitive.GLOBALS["shape"]
shape.value = lambda x: {**x, "features": x["model"].get_mapper("shape")(x["features"])}


if args.mode == "train":
    train(model, config, args)
if args.mode == "eval":
    evaluate(model, config, args)