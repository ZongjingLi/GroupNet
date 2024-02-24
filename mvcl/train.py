'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-01-24 18:18:48
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-01-24 18:18:52
 # @ Description: This file is distributed under the MIT license.
 '''
import sys
from turtle import back
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from rinarak.logger import get_logger,set_output_file
from rinarak.utils.tensor import gather_loss, logit
from rinarak.program import Program
from rinarak.utils.tensor import freeze
from rinarak.utils.tensor import stats_summary

from mvcl.primitives import Primitive

from datasets.sprites_base_dataset import SpritesBaseDataset
from datasets.sprites_meta_dataset import SpritesMetaDataset


dataset_map = {
    "sprites_base": SpritesBaseDataset,
    "sprites_meta": SpritesMetaDataset,
}

def unfreeze(module):
    for param in module.parameters():param.requires_grad = True

def to_instance_masks(one_hot_masks):
    ids = torch.unique(one_hot_masks)
    masks = []
    for i in ids:
            masks.append((one_hot_masks == i).unsqueeze(0))
    masks = torch.cat(masks, dim = 0).permute(1,0,2,3)
    return masks

def log_gt_inputs(ims, masks):
    plt.figure("input-img")
    plt.imshow(ims);plt.axis('off')
    plt.savefig("outputs/input_image.png", bbox_inches='tight');plt.cla()
    plt.imshow(masks);plt.axis('off') 
    plt.savefig("outputs/gt_masks.png", bbox_inches='tight')

def log_instance_masks(masks, alives):
    fig = plt.figure("masks"); b = 0
    for i in range(masks.shape[1]):
        ax = fig.add_subplot(2,5,1+i);plt.axis('off')
        ax.imshow(masks[b,i,:,:] * alives[b,i])
    plt.savefig("outputs/predict_masks.png", bbox_inches='tight')

def log_knowledge_grounding_info(questions, programs, answers, predict_answers, log_file = "outputs/logqa.txt"):
    with open(log_file,"w") as log_file:
        for i in range(len(programs)):
            log_file.write("q:"+questions[i][0]+"\n")
            log_file.write("p:"+programs[i][0]+"\n")
            log_file.write("gt-answer:"+answers[i][0]+" predict-answer: "+str(predict_answers[i])+"\n\n")

def to_logits(tensor):
    """check if input tensors are in the form of logits
    """
    if (tensor.max() > 1 or tensor.min() < 0): return tensor # return if it is already unbounded
    return logit(tensor)

def flatten(features, mode = "img"):
    """masks in the form of BxKxWxH"""
    if mode == "img": return features.permute(2,0,1).flatten(start_dim = 1, end_dim = 2)
    return features

def prepare_context(alives, masks, features, model, b = 0):
    """prepare the context for the execution
    inputs:
        end: BxKx1 tensor representing logits or probability (check bounded)
        masks: BxKxN tensor representing logits or probability of masks of primitives features (check bounded)
        features: BxNxD tensor representing features of the primitives
        model: the executor for concept embeddings and more
    return:
        the context as a diction
    """
    assert len(alives.shape) in [2,3], "only batch wise operations are allowed"
    if len(alives.shape) == 2:B, K = alives.shape
    elif len(alives.shape) == 3:B, K, _ = alives.shape
    ends = alives.reshape([B, K, 1])
    context = {
            "end": to_logits(ends)[b],
            "masks": to_logits(flatten(masks, mode = "img"))[b],
            "features": features[b].flatten(start_dim = 0, end_dim = 1),
            "model": model
            }
    return context

def ground_knowledge(vqa_sample, masks, alives, features, model):
    language_loss = 0.0
    numbers = [str(i) for i in range(10)]
    questions = vqa_sample["questions"]
    programs = vqa_sample["programs"]
    answers = vqa_sample["answers"]
    for b in range(len(programs[0])):
        context = {
                    "end":logit(alives[b]),
                    "masks": logit(masks[b].flatten(start_dim = 1, end_dim = 2)),
                    "features": features[b].flatten(start_dim = 0, end_dim = 1),
                    "model": model
            }
    predict_answers = []
    for program_idx in range(len(programs) ):
        program = programs[program_idx][b]
        answer = answers[program_idx][b]

        p = Program.parse(program)
        output = p.evaluate({0:context})
        if answer in ["yes", "no"]:
            if answer == "yes":
                language_loss += -torch.log(output["end"].sigmoid())
            if answer == "no":
                language_loss += -torch.log(1 - output["end"].sigmoid())
            pred_ans = "yes" if output["end"] > 0 else "no"
        if answer in numbers:
            language_loss += (output["end"]-int(answer))**2
            pred_ans = str(output["end"].detach().numpy())
        predict_answers.append(pred_ans)

    language_loss /= len(programs[0]) * len(programs)
    return {"loss": language_loss, "questions":questions, "answers":answers, "programs":programs, "predict_answers":predict_answers}

def visualize_concept_maps(concept_name, features, model):
    import math
    mapper = model.implementations[concept_name]

    concept_feature_map = mapper(features)

    values = model.central_executor.type_constraints[concept_name]

    W = H = int(math.sqrt(concept_feature_map.shape[1]))
    plt.figure(f"{concept_name}")
    for i,value in enumerate(values):
        value_map = model.entailment(concept_feature_map, value).reshape([W,H])
        plt.subplot(1, len(values), i+1)
        plt.cla()
        plt.imshow(value_map.sigmoid().detach() , cmap = "bone")
        plt.title(value)
    plt.savefig("outputs/{}_visualize.png".format(concept_name), bbox_inches='tight')

def train(model, config, args):
    model = model.to(config.device)
    set_output_file(f"logs/expr_{args.domain_name}_train.txt")
    train_logger = get_logger("expr_train")

    """prepare to log the dataset"""
    train_logger.critical(f"prepare the dataset {args.dataset_name}.")
    dataset_dir = args.dataset_dir
    train_dataset = dataset_map[args.dataset_name]("train", dataset_dir)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)

    """freeze parameters in the model] [optional] """
    if args.freeze_perception:
        freeze(model.perception);train_logger.critical("Percept module is freezed.")
    else: unfreeze(model.perception)
    for param in model.central_executor.parameters():
        param.requires_grad = False if args.freeze_knowledge else True
    if args.freeze_knowledge:train_logger.critical("Knowledge module is freezed")

    """setup hyperparameters for the Optimizer, Loss calculation"""
    alpha = 1.0
    beta = 1.0
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    if args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr = args.lr)
    save_file = f"{args.ckpt_dir}/{args.expr_name}_{{}}.pth"
   
    """start the training process"""
    itrs = 0
    train_logger.critical("start training process")
    for epoch in range(args.epochs):
        epoch_loss = .0
        for sample in train_loader:
            itrs += 1
            # [calculate and gather the loss]
            ims = sample["img"]
            masks = sample["masks"]

            """train the perception module, extract masks"""
            outputs = model.perception(ims, masks.long().unsqueeze(1))
            percept_loss = gather_loss(outputs)["loss"] / ims.shape[0] # gather loss in the perception module
            all_masks = outputs["masks"].permute(0,3,1,2)
            alives = outputs["alive"]

            # weird demo
            all_masks = to_instance_masks(masks)
            alives = torch.ones(all_masks.shape[:2])
            #print(all_masks.shape, alives.shape)

            """calculate loss for the knowledge grounding"""
            language_loss = 0.0 # intialize the knowledge training
            backbone_features = model.implementations["universal"](ims)
            if not args.freeze_knowledge:
                language_grounding_outputs = ground_knowledge(sample, all_masks, alives, backbone_features, model)
                language_loss += language_grounding_outputs["loss"]

            """calculate the overall loss"""
            loss = alpha * percept_loss + beta * language_loss

            """start the optimization"""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not (itrs % args.ckpt_itrs): 
                """Save model info and state dict at checkpoint"""
                train_logger.critical(f"save the checkpoint at itrs:{itrs} h:{loss}")
                torch.save(model.perception.state_dict(),f"{args.ckpt_dir}/{args.expr_name}_percept_backup.pth")
                torch.save(model.central_executor.state_dict(),f"{args.ckpt_dir}/{args.expr_name}_knowledge_backup.pth")

                """Save the image and masks for visualization"""
                log_gt_inputs(ims[0].permute(1,2,0), masks[0]) # only for the first batch
                """Save the output masks predicted"""
                log_instance_masks(all_masks, alives) # log the instance level masks for each object
                """Log Knowledge Gronding Visualizatoin"""
                if not args.freeze_knowledge:
                    questions = language_grounding_outputs["questions"]
                    programs = language_grounding_outputs["programs"]
                    answers = language_grounding_outputs["answers"]
                    predict_answers = language_grounding_outputs["predict_answers"]
                    log_knowledge_grounding_info(questions, programs, answers, predict_answers) # log the question answer grounding infos.

                visualize_concept_maps("color", backbone_features[0].flatten(start_dim = 0, end_dim = 1), model)
                visualize_concept_maps("shape", backbone_features[0].flatten(start_dim = 0, end_dim = 1), model)

            epoch_loss += float(loss) # add the current loss to the total loss
            sys.stdout.write(f"\repoch:{epoch+1} itrs:{itrs} loss:{loss} percept:{percept_loss} lang:{language_loss}\n")
        train_logger.critical(f"epoch:{epoch+1} loss:{epoch_loss}")

    """finall save the model checkpoint and state dict [seperately for perception and knowledge model]"""
    torch.save(model.perception.state_dict(),save_file.format("percept")) # save the torch parameters
    torch.save(model.central_executor.state_dict(),save_file.format("knowledge")) # save the torch parameters
    train_logger.critical(f"model training completed, saved at {save_file}")


def evaluate(model, config, args):
    set_output_file(f"logs/{args.domain_name}_expr_eval.txt")
    eval_logger = get_logger("expr_eval")
    eval_logger.critical("start evaluation")
