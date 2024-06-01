'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-01-24 17:50:12
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-01-24 18:24:56
 # @ Description: This file is distributed under the MIT license.
'''
import argparse
from rinarak.domain import Domain, load_domain_string
from rinarak.program import Primitive
from rinarak.dklearn.nn.mlp import FCBlock

from sgnet.train  import train, evaluate
from sgnet.grouping import SymbolicGrouper

from datasets.plagueworks_dataset import PlagueWorksDataset

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

local = True
dataset_dir = "/Users/melkor/Documents/datasets" if local else "datasets"
project_dir = "/Users/melkor/Documents/GitHub/MetaVisualConceptLearner/"
device = "cuda" if torch.cuda.is_available() else "cpu"

argparser = argparse.ArgumentParser()
argparser.add_argument("--expr_type",                   default = "train")
argparser.add_argument("--dataset_dir",                 default = dataset_dir)
argparser.add_argument("--device",                      default = device)
argparser.add_argument("--save_name",                   default = "save_states")

# [Experiment configuration]
argparser.add_argument("--domain_name",                 default = "demo")
argparser.add_argument("--mode",                        default = "train")
argparser.add_argument("--dataset_name",                default = "PlagueWorksDataset")
argparser.add_argument("--resolution",                  default = (128,128))

# [Training detail configurations]
argparser.add_argument("--epochs",                      default = 100)
argparser.add_argument("--batch_size",                  default = 2)
argparser.add_argument("--optimizer",                   default = "Adam")
argparser.add_argument("--lr",                          default = 2e-4)
argparser.add_argument("--shuffle",                     default = True)

# [Elaborated training details]
argparser.add_argument("--freeze_perception",           default = False)
argparser.add_argument("--freeze_knowledge",            default = True)

# [Save checkpoints at dir...]
argparser.add_argument("--ckpt_itrs",                   default = 32)
argparser.add_argument("--ckpt_dir",                    default = "checkpoints_")
argparser.add_argument("--load_checkpoint",             default = False)

args = argparser.parse_args()

# [Parse the domain configuration]
domain_parser = Domain(project_dir + "sgnet/base.grammar")
meta_domain_str = ""
with open(project_dir + f"/domains/{args.domain_name}_domain.txt","r") as domain:
    for line in domain: meta_domain_str += line
domain = load_domain_string(meta_domain_str, domain_parser)


resolution = args.resolution
# [Make the Model]
model = SymbolicGrouper(resolution = resolution, K = 5)
if args.load_checkpoint:
    model.load_state_dict(torch.load(args.load_checkpoint, map_location = args.device))
    print(f"succesfully load checkpoint from {args.load_checkpoint}")
model.to(args.device)


# [Load the Dataset]
train_dataset = eval("{}(resolution = {},dataset_dir='{}')".
                         format(args.dataset_name, resolution, args.dataset_dir))

# [Train the Model]
if args.expr_type == "train":
    print(f"start the training process of SGNet on {args.dataset_name}")
    train(
        model = model, 
        dataset = train_dataset, 
        batch_size = args.batch_size, 
        epochs = args.epochs,
        lr = args.lr,
        save_name = args.save_name,
        device = args.device)
if args.expr_type == "eval":
    print(f"start the evaluation on {args.dataset_name}")
    evaluate(model = model, dataset = train_dataset, device = args.device)