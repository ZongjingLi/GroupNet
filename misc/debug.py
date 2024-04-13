import torch
import torch.nn as nn
from mvcl.percept.resnet import ResNet_Deeplab

resnet = ResNet_Deeplab()

B, C, W, H = (3,3,512,512)
ims = torch.randn([B, C, W, H])

feats = resnet(ims)

print(ims.shape)
print(feats.shape)
