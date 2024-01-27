'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-01-24 18:18:48
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-01-24 18:18:52
 # @ Description: This file is distributed under the MIT license.
 '''
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from rinarak.logger import get_logger,set_output_file
from rinarak.utils.tensor import gather_loss, logit
from rinarak.program import Program
from rinarak.utils.tensor import freeze

from mvcl.primitives import Primitive

from datasets.sprites_base_dataset import SpritesBaseDataset
from datasets.sprites_meta_dataset import SpritesMetaDataset


dataset_map = {
    "sprites_base": SpritesBaseDataset,
    "sprites_meta": SpritesMetaDataset,
}

def train(model, config, args):
    numbers = [str(i) for i in range(10)]
    set_output_file(f"logs/{args.domain_name}_expr_train.txt")
    train_logger = get_logger("expr_train")
    # [prepare to log the dataset]
    train_logger.critical(f"prepare the dataset {args.dataset_name}.")
    dataset_dir = args.dataset_dir
    train_dataset = dataset_map[args.dataset_name]("train", dataset_dir)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    # [log the model and mount on device]
    model = model.to(config.device)

    # [freeze parameters in the model] [optional]
    freeze(model.perception)
    freeze(model.central_executor)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    if args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr = args.lr)
    save_file = f"{args.ckpt_dir}/{args.expr_name}.pth"
    # [start the training processs]
    itrs = 0
    train_logger.critical("start training process")
    for epoch in range(args.epochs):
        epoch_loss = .0
        for sample in train_loader:
            itrs += 1
            # [calculate and gather the loss]
            ims = sample["img"]
            masks = sample["masks"]
            outputs = model.perception(ims, masks.long().unsqueeze(1))
            percept_loss = gather_loss(outputs)["loss"] # gather loss in the perception module
            all_masks = outputs["masks"]
            alives = outputs["alive"]

            language_loss = 0.0 # intialize the knowledge training
            questions = sample["questions"]
            programs = sample["programs"]
            answers = sample["answers"]
            backbone_features = outputs["features"]
    
            for b in range(len(programs[0])):
                context = {
                "end":logit(alives[b].squeeze(-1)),
                "masks": logit(all_masks[b].permute(2,0,1).flatten(start_dim = 1, end_dim = 1)),
                "features": backbone_features[b].flatten(start_dim = 0, end_dim = 1),
                "model": model
                }
                for program_idx in range(len(programs)):
                    question = questions[program_idx][b]
                    program = programs[program_idx][b]
                    answer = answers[program_idx][b]
                    p = Program.parse(program)
                    output = p.evaluate({0:context})
                    if answer in ["yes", "no"]:
                        if answer == "yes":
                            language_loss += -torch.log(output["end"].sigmoid())
                        if answer == "no":
                            language_loss += -torch.log(1 - output["end"].sigmoid())
                    if answer in numbers:
                            language_loss += (output["end"]-int(answer))**2

            language_loss /= len(programs[0]) * len(programs)

            # calculate the overall loss
            loss = percept_loss + language_loss

            # [start the optimization]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not (itrs % args.ckpt_itrs): 
                train_logger.critical(f"save the checkpoint at itrs:{itrs} h:{loss}")
                torch.save(model.state_dict(),f"{args.ckpt_dir}/{args.expr_name}_backup.pth")
            epoch_loss += float(loss) # add the current loss to the total loss
            sys.stdout.write(f"\repoch:{epoch+1} itrs:{itrs} loss:{loss}\n")
        train_logger.critical(f"epoch:{epoch+1} loss:{epoch_loss}")
    torch.save(model.state_dict(),save_file) # save the torch parameters
    train_logger.critical(f"model training completed, saved at {save_file}")

def evaluate(model, config, args):
    set_output_file(f"logs/{args.domain_name}_expr_eval.txt")
    eval_logger = get_logger("expr_eval")
    eval_logger.critical("start evaluation")
