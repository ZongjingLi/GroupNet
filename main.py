'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-01-24 17:50:12
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-01-24 18:24:56
 # @ Description: This file is distributed under the MIT license.
'''
from distutils.command.build import build
from rinarak.domain import Domain, load_domain_string
from rinarak.program import Primitive
from rinarak.dklearn.nn.mlp import FCBlock

from mvcl.model import MetaVisualLearner
from mvcl.train import train, evaluate
from mvcl.config import config, argparse
from mvcl.primitives import *
from mvcl.custom import build_custom
import torch
import torch.nn as nn

local = True
dataset_dir = "/Users/melkor/Documents/datasets" if local else "datasets"

argparser = argparse.ArgumentParser()
argparser.add_argument("--expr_name",                   default = "PlayroomKL1")
argparser.add_argument("--dataset_dir",                 default = dataset_dir)
# [Experiment configuration]
argparser.add_argument("--domain_name",                 default = "demo")
argparser.add_argument("--mode",                        default = "train")
argparser.add_argument("--dataset_name",                default = "sprites_base")

# [Training detail configurations]
argparser.add_argument("--epochs",                      default = 1000)
argparser.add_argument("--batch_size",                  default = 8)
argparser.add_argument("--optimizer",                   default = "Adam")
argparser.add_argument("--lr",                          default = 5e-4)
argparser.add_argument("--shuffle",                     default = True)

# [Elaborated training details]
argparser.add_argument("--freeze_perception",           default = False)
argparser.add_argument("--freeze_knowledge",            default = True)

# [Save checkpoints at dir...]
argparser.add_argument("--ckpt_itrs",                   default = 32)
argparser.add_argument("--ckpt_dir",                    default = "checkpoints")
argparser.add_argument("--load_ckpt_knowledge",         default = False)
argparser.add_argument("--load_ckpt_percept",           default = False)

args = argparser.parse_args()

# [Parse the domain configuration]
domain_parser = Domain("mvcl/base.grammar")
meta_domain_str = ""
with open(f"domains/{args.domain_name}_domain.txt","r") as domain:
    for line in domain: meta_domain_str += line
domain = load_domain_string(meta_domain_str, domain_parser)

# [Build the Model]
#args.load_ckpt_percept = "checkpoints/KFT_percept_backup.pth"
#args.load_ckpt_knowledge = "checkpoints/KFT_knowledge_backup.pth"
if args.domain_name == "Playroom":
    config.domain_name = "Playroom"

if args.dataset_name == "Playroom":
    config.channel_dim = 3
    config.resolution = (64,64)

if args.dataset_name == "Sprites":
    config.channel_dim = 4
    config.resolution = (64,64)

model = MetaVisualLearner(domain, config)
model = build_custom(model, config, domain.domain_name)
#model.load_state_dict(torch.load("checkpoints/KFT_backup.pth"))

if args.load_ckpt_percept: model.perception.load_state_dict(torch.load(args.load_ckpt_percept))
if args.load_ckpt_knowledge: model.central_executor.load_state_dict(torch.load(args.load_ckpt_knowledge))


if args.mode == "train":
    train(model, config, args)
if args.mode == "eval":
    evaluate(model, config, args)

if args.mode == "meta":
    for train_mode in []:
        pass