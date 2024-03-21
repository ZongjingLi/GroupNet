from operator import truediv
from datasets.sprites_base_dataset import SpritesBaseDataset
from mvcl.config import config
from mvcl.model import MetaVisualLearner
from mvcl.primitives import *
from rinarak.domain import Domain, load_domain_string

from datasets.playroom_dataset import *
import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte
from rinarak.utils.tensor import stats_summary
from mvcl.custom import *

shuffle = True
domain_parser = Domain("mvcl/base.grammar")

meta_domain_str = f"""
(domain meta_demo)
(:type
    object - vector[float,100]
    position - vector[float,2]
    color - vector[float, 64]
    category
)
(:predicate
    color ?x-object -> vector[float,64]
    is-red ?x-object -> boolean
    is-blue ?x-object -> boolean
    is-ship ?x-object -> boolean
    is-house ?x-object -> boolean
)
(:derived
    is-green ?x-color expr: (??f ?x)
)
(:constraint
    (color: is-red is-blue)
    (category: is-ship is-house)
)
"""
config.resolution = (64,64)
B = 1; D = 3
W, H = config.resolution
N = W * H
from torchvision import transforms
from rinarak.logger import get_logger, set_output_file
set_output_file(f"logs/demo_run.txt")
demo_logger = get_logger("DemoLogger")

"""Load the Domain and model Config """
domain = load_domain_string(meta_domain_str, domain_parser)
demo_logger.critical(f"domain string ({domain.domain_name}) loaded successfully.")


"""Load the MetaVisual-Learner"""
device = "cuda:0" if torch.cuda.is_available() else "cpu"
metapercept = MetaVisualLearner(domain, config)
metapercept = build_custom(metapercept, config, "MetaLearn")
metapercept.load_state_dict(torch.load("checkpoints/KL0_backup.pth", map_location = device))
#metapercept.central_executor.load_state_dict(torch.load("checkpoints/KFT-UNet_knowledge_backup.pth"))
demo_logger.critical("MetaVisualLearner created successfully.")
demo_logger.critical(config)

"""Load the Playroom Custom Dataset"""
dataset = PlayroomDataset(training = True)
dataset = SpritesBaseDataset()
loader = DataLoader(dataset, shuffle = shuffle)
for sample in loader:break

if 'img1' in sample:
    img = transforms.Resize([W, H])(sample["img1"])
else:
    img = transforms.Resize([W, H])(sample["img"])
#img = torch.cat([img], dim = 1)

if 'gt_segment' in sample:
    masks = sample["gt_segment"]
else:
    masks = sample["masks"]

masks = transforms.Resize([W, H])(masks).unsqueeze(-1)[0]


if "gt_segment" or "masks" in sample:
    plt.figure("input-img vs gt-segment")
    plt.subplot(121);plt.imshow(img[0].permute(1,2,0));plt.axis("off")
    plt.subplot(122);plt.imshow(masks.squeeze());plt.axis("off")
    plt.savefig("outputs/demo_inputs.png", bbox_inches='tight')
else:
    plt.figure("input-img")
    plt.imshow(img[9].permute(1,2,9));plt.axis("off"); plt.savefig("outputs/demo_inputs.png", bbox_inches='tight')

"""Test the Visual Grouping Module"""
propagator = metapercept.perception.propagator
extractor = metapercept.perception.competition

K = metapercept.perception.K
num_long_range = metapercept.perception.num_long_range
locals = getattr(metapercept.perception, f"indices_{W}x{H}").repeat(B,1,1).long()

# calculate the localized connections between local graphs.
segment_targets = masks


sample_inds = metapercept.perception.get_indices([W,H])


# calculate logits of for the connection logits
segment_targets = segment_targets.reshape([B,N]).unsqueeze(-1).long().repeat(1,1, sample_inds.shape[-1])

#u_indices = sample_inds[1,...].long()
#v_indices = sample_inds[2,...].long()
#u_seg = torch.gather(segment_targets, 1, u_indices)
#v_seg = torch.gather(segment_targets, 1, v_indices)
#connects = ((u_seg == v_seg).float())
#logits = logit( connects / (torch.sum(connects, dim = -1, keepdim = True) + 1e-9) )

samples = torch.gather(segment_targets,1, sample_inds[2,...]).squeeze(-1)
targets = segment_targets == samples
connection = targets / (torch.sum(targets, -1, keepdim=True) + 1e-9)
logits = logit(connection)

D = 64
# calculate the masks and prop plateaus maps
masks, agents, alive, prop_maps = metapercept.perception.compute_masks(logits, sample_inds, prop_dim = D)

def save_propmaps(prop_maps, name = "outputs/propmaps.gif"):
    images = [(img_as_ubyte(pmap.reshape(W,H,-1)[:,:,-3])) for pmap in prop_maps]
    imageio.mimsave(name, images, duration=0.1)


save_propmaps(prop_maps, 'outputs/props.gif')

# save the masks extracted on the plateau map
counter = 0
alive = alive.flatten()
nums = int(alive.sum())
for i in range(alive.shape[0]):
    if alive[i]:
        counter += 1
        plt.figure("masks", figsize = (12, 3))
        plt.subplot(1,nums,counter)
        plt.imshow(masks[0,:,:,i]* alive[i])
        plt.axis(False)
        plt.figure("region", figsize = (12, 3))
        plt.subplot(1,nums,counter)
        plt.axis(False)
        comp_mask = masks[0,:,:,i]* alive[i]
        im = (comp_mask * img[0]).permute(1,2,0)
        if im.shape[-1] == 3:
            im = torch.cat([im, comp_mask.unsqueeze(-1)], dim = -1)
        else:
            plt.imshow(im)
plt.savefig("outputs/predict_masks0.png", bbox_inches='tight')

demo_logger.critical("Visual Grouping Module Demonstration")


def from_onehot_mask(one_hot_mask):
    seg_masks = torch.zeros([B,W,H])
    for i in range(int(one_hot_mask.max())):
        seg_masks[one_hot_mask==i] = i;
    seg_masks = seg_masks
    return seg_masks


if img.max() > 1.1: img = img / 256.

seg_masks = from_onehot_mask(masks)

metapercept.perception.propagator.num_iters = 132
#metapercept.implementations.load_state_dict(torch.load("checkpoints/KL0_imps_backup.pth", map_location = device))
#metapercept = torch.load("checkpoints/KL0_backup.ckpt", map_location = "cpu")

outputs = metapercept.group_concepts(img, "object", target = None)
masks = outputs["masks"]
alive = outputs["alive"]
prop_maps = outputs["prop_maps"]


print(alive)

counter = 0
alive = alive.flatten()
nums = int(alive.sum())
for i in range(alive.shape[0]):
    if alive[i]:
        counter += 1
        plt.figure("masks", figsize = (12, 3))
        plt.subplot(1,nums,counter)
        plt.imshow(masks[0,:,:,i]* alive[i])
        plt.axis(False)
        plt.figure("region", figsize = (12, 3))
        plt.subplot(1,nums,counter)
        plt.axis(False)
        comp_mask = masks[0,:,:,i]* alive[i]
        im = (comp_mask * img[0]).permute(1,2,0)
        if im.shape[-1] == 3:
            im = torch.cat([im, comp_mask.unsqueeze(-1)], dim = -1)
        else:
            plt.imshow(im)
plt.savefig("outputs/predict_masks.png", bbox_inches='tight')


# transform the mask
save_propmaps(prop_maps, 'outputs/concept_props.gif')

demo_logger.critical("Concept Centric Affinity Calculator.")