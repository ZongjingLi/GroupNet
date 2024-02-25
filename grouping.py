from mvcl.percept.propagation import *
from mvcl.percept.competition import *
from mvcl.percept.metanet import generate_local_indices, MetaNet
from mvcl.config import config

dx, dy = (3,3)
W, H ,C  = 64,64,3
resolution = (W, H, C)

propgator = GraphPropagation()
extractor = Competition()

class IdentityZJL(nn.Module):
    def __Init__(self):
        super().__init__()
    def forward(self, x): return x

config.backbone_feature_dim = C
config.max_num_masks = 10
net = MetaNet(config)
net.backbone = IdentityZJL()
net.ks_map = IdentityZJL()
net.qs_map = IdentityZJL()

ims = torch.zeros(resolution)

nW = 2
nH = 6

for i in range(nW):
    for j in range(nH):
        cx = int( (i+1) * (W / (nW + 1) ) )
        cy = int( (j+1) * (H / (nH + 1) ) )
        ims[cx-dx:cx+dx,cy-dy:cy+dy,:] = 1.0

pseduo_masks = torch.ones([1,1,64,64])
percept_outputs = net(ims[None,...].permute(0,3,1,2), pseduo_masks)
masks = percept_outputs["masks"]
ends = percept_outputs["alive"]

plt.figure("segments")
for i in range(masks.shape[-1]):
    plt.subplot(2,5,i + 1)
    plt.imshow(masks[0,:,:,i] * ends[0,i,0])
print(masks.shape)
print(ends)

plt.figure("gt input")
plt.imshow(ims)
plt.show()