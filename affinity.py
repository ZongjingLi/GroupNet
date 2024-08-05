import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sgnet.grouping import GraphPropagation, Competition, local_to_sparse_global_affinity, generate_local_indices
from sgnet.utils import to_cc_masks

from datasets.plagueworks_dataset import PlagueWorksDataset, DataLoader
import matplotlib.pyplot as plt
import torchvision


resolution = (128, 128)
dataset = PlagueWorksDataset(split = "train", resolution = resolution, dataset_dir = "/Users/melkor/Documents/datasets")
def resample():
    loader = DataLoader(dataset, batch_size = 1, shuffle = 0)
    for sample in loader:break
    return sample

def visualize_image_grid(images, row, save_name = "image_grid"):
    plt.figure(save_name, frameon = False, figsize = (10,10));plt.cla()
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    comps_grid = torchvision.utils.make_grid(images,normalize=True,nrow=row).permute([1,2,0])

    plt.imshow(comps_grid.cpu().detach().numpy())
    plt.savefig("outputs/{}.png".format(save_name), bbox_inches='tight', pad_inches=0)

sample = resample()
img1 = sample["img1"]
img2 = sample["img2"]
mask = sample["masks"].unsqueeze(1).repeat(1,3,1,1) * 25

batch_grid = torch.cat([img1 * 255., img2 * 255., mask], dim = 0)
visualize_image_grid(batch_grid, row = 3)

import cv2 as cv
import numpy as np
# boundary restriction
w = 3
f = 0.19921875
boundary = torch.zeros([1, 128, 128])
boundary[:, 20: 100, 40:40+w] = f
boundary[:, 20: 100, 80:80+w] = f
boundary[:, 62: 62+w, 40:80+w] = f
boundary[:, 20: 20+w, 40:80+w] = f


gray_scale = (img1[0] * 255).permute(1,2,0).int().detach().numpy()
gray_scale = np.uint8(gray_scale)

edges = cv.Canny(gray_scale,100,200)

boundary = None
boundary = torch.tensor(edges).unsqueeze(0) /255.


from sgnet.grouping import SymbolicGrouper
model = SymbolicGrouper(resolution = (128,128), K = 7, long_range_ratio = 0.2)
#model.load_state_dict(torch.load("checkpoints_/k7_save.pth"))
model = model.to("cpu")
model.activate_all_cues()
model.deactivate_cue("color")
#model.deactivate_cue("movable")
print(model.cues_activated)

outputs = model(img1, {}, boundary)

masks = outputs["masks"]
from sgnet.utils import to_cc_masks
plt.figure("vis masks")
comps = to_cc_masks(masks["objects"][0], 1)
mask = torch.zeros([128,128])
for i in range(comps.shape[-1]):
    mask[comps[:,:,i]] = i+1
plt.imshow(mask)
plt.axis("off")
plt.savefig("outputs/vis-mask.png", bbox_inches = "tight")
plt.show()