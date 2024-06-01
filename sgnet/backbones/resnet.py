'''
 # @ Author: Zongjing Li
 # @ Create Time: 2023-12-22 11:01:46
 # @ Modified by: Zongjing Li
 # @ Modified time: 2023-12-22 11:06:21
 # @ Description: This file is distributed under the MIT license.
'''
import torch
import torch.nn as nn

class ResidualDenseConv(nn.Module):
    def __init__(self, channel_in, growRate, kernel_size=3):
        super(ResidualDenseConv, self).__init__()
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(channel_in, G, kernel_size, padding=(kernel_size-1)//2, stride=1),
            nn.Tanh(),
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class ResidualDenseBlock(nn.Module):
    def __init__(self, grow0, grow_rate, num_conv_layers):
        super(ResidualDenseBlock, self).__init__()
        convs = []
        for c in range(num_conv_layers):
            convs.append(ResidualDenseConv(grow0 + c*grow_rate, grow_rate))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.local_feature_fusion = nn.Conv2d(grow0 + num_conv_layers*grow_rate, grow0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.local_feature_fusion(self.convs(x)) + x

class ResidualDenseNetwork(nn.Module):
    """Residual Dense Network as the convolutional visual backbone
    """
    def __init__(self, grow0, n_colors = 4, kernel_size = 3, scale = [2],rdn_config = [4,3,16], no_upsample = True):
        super(ResidualDenseNetwork, self).__init__()
        self.no_upsample = no_upsample

        r = scale[0]

        # number of RDB blocks, conv layers, out channels
        self.block_num, conv_layer_num, out_channel_num = rdn_config
        """
        {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]
        """

        # Shallow feature extraction net
        self.shallow_feature_net1 = nn.Conv2d(n_colors, grow0, kernel_size, padding=(kernel_size-1)//2, stride=1)
        self.shallow_feature_net2 = nn.Conv2d(grow0, grow0, kernel_size, padding=(kernel_size-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.residual_dense_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.residual_dense_blocks.append(
                ResidualDenseBlock(grow0 = grow0, grow_rate = out_channel_num, num_conv_layers = conv_layer_num)
            )

        # Global Feature Fusion
        self.global_feature_fusion = nn.Sequential(*[
            nn.Conv2d(self.block_num * grow0, grow0, 1, padding=0, stride=1),
            nn.Conv2d(grow0, grow0, kernel_size, padding=(kernel_size-1)//2, stride=1)
        ])

        if no_upsample:
            self.out_dim = grow0
        else:
            self.out_dim = n_colors
            # Up-sampling net
            if r == 2 or r == 3:
                self.upsample_net = nn.Sequential(*[
                    nn.Conv2d(grow0, out_channel_num * r * r, kernel_size, padding=(kernel_size-1)//2, stride=1),
                    nn.PixelShuffle(r),
                    nn.Conv2d(out_channel_num, n_colors, kernel_size, padding=(kernel_size-1)//2, stride=1)
                ])
            elif r == 4:
                self.upsample_net = nn.Sequential(*[
                    nn.Conv2d(grow0, out_channel_num * 4, kernel_size, padding=(kernel_size-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(out_channel_num, out_channel_num * 4, kernel_size, padding=(kernel_size-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(out_channel_num, n_colors, kernel_size, padding=(kernel_size-1)//2, stride=1)
                ])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f1 = self.shallow_feature_net1(x)
        x = self.shallow_feature_net2(f1)

        residual_out = []
        for i in range(self.block_num):
            x = self.residual_dense_blocks[i](x)
            residual_out.append(x)

        x = self.global_feature_fusion(torch.cat(residual_out,1))
        x += f1

        if self.no_upsample:
            return x
        else:
            return self.upsample_net(x)