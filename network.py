import torch
import torch.nn as nn
from network_module import *
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

# # ----------------------------------------
# #         Initialize the networks
# # ----------------------------------------
# def weights_init(net, init_type = 'normal', init_gain = 0.02):
#     """Initialize network weights.
#     Parameters:
#         net (network)   -- network to be initialized
#         init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
#         init_gain (float)    -- scaling factor for normal, xavier and orthogonal
#     In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
#     """
#     def init_func(m):

#         classname = m.__class__.__name__

#         if hasattr(m, 'weight') and classname.find('Conv') != -1:
#             if init_type == 'normal':
#                 torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
#             elif init_type == 'xavier':
#                 torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
#             elif init_type == 'kaiming':
#                 torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
#             elif init_type == 'orthogonal':
#                 torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#         elif classname.find('BatchNorm2d') != -1:
#              torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
#              torch.nn.init.constant_(m.bias.data, 0.0)
#         elif classname.find('Linear') != -1:
#              torch.nn.init.normal_(m.weight, 0, init_gain)
#              # torch.nn.init.constant_(m.bias, 0)


#     # apply the initialization function <init_func>
#     print('initialize network with %s type' % init_type)
#     net.apply(init_func)



# class ConvBlock(torch.nn.Module):
#     def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='relu', norm='batch'):
#         super(ConvBlock, self).__init__()
#         self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

#         self.norm = norm
#         if self.norm =='batch':
#             self.bn = torch.nn.BatchNorm2d(output_size)
#         elif self.norm == 'instance':
#             self.bn = torch.nn.InstanceNorm2d(output_size)

#         self.activation = activation
#         if self.activation == 'relu':
#             self.act = torch.nn.ReLU(True)
#         elif self.activation == 'prelu':
#             self.act = torch.nn.PReLU()
#         elif self.activation == 'lrelu':
#             self.act = torch.nn.LeakyReLU(0.2, True)
#         elif self.activation == 'tanh':
#             self.act = torch.nn.Tanh()
#         elif self.activation == 'sigmoid':
#             self.act = torch.nn.Sigmoid()

#     def forward(self, x):
#         if self.norm is not None:
#             out = self.bn(self.conv(x))
#         else:
#             out = self.conv(x)

#         if self.activation is not None:
#             return self.act(out)
#         else:
#             return out
            
# class Net(torch.nn.Module):
#     def __init__(self, num_channels, scale_factor, d, s, m):
#         super(Net, self).__init__()

#         # Feature extraction
#         self.first_part = ConvBlock(num_channels, d, 5, 1, 5//2, activation='prelu', norm=None)

#         self.layers = []
#         # Shrinking
#         self.layers.append(ConvBlock(d, s, 1, 1, 0, activation='prelu', norm=None))
#         # Non-linear Mapping
#         for _ in range(m):
#             self.layers.append(ConvBlock(s, s, 3, 1, 1, activation='prelu', norm=None))
#         # self.layers.append(nn.PReLU())
#         # Expanding
#         self.layers.append(ConvBlock(s, d, 1, 1, 0, activation='prelu', norm=None))

#         self.mid_part = torch.nn.Sequential(*self.layers)

#         # Deconvolution
#         self.last_part = nn.ConvTranspose2d(d, num_channels, 9, scale_factor, 9//2, output_padding=scale_factor-1)
#         #self.last_part = nn.ConvTranspose2d(d, num_channels, 9, scale_factor, 3, output_padding=1)
        
#         #self.pslayers = []
#         #self.pslayers.append(ConvBlock(d, 4*num_channels, 9, 1, 9//2, norm=None))
#         #self.pslayers.append(nn.PixelShuffle(2))
#         #self.pslayers.append(ConvBlock(num_channels, 4*num_channels, 9, 1, 9//2, norm=None))
#         #self.pslayers.append(nn.PixelShuffle(2))
#         #self.last_part = torch.nn.Sequential(*self.pslayers)


#     def forward(self, x):
#         out = self.first_part(x)
#         out = self.mid_part(out)
#         out = self.last_part(out)
#         return out

# class Net_FullSizeOutput(torch.nn.Module):
#     def __init__(self, num_channels, scale_factor, d, s, m):
#         super(Net_FullSizeOutput, self).__init__()

#         # Feature extraction
#         self.first_part = ConvBlock(num_channels, d, 5, 1, 5//2, activation='prelu', norm=None)

#         self.layers = []
#         # Shrinking
#         self.layers.append(ConvBlock(d, s, 1, 1, 0, activation='prelu', norm=None))
#         # Non-linear Mapping
#         for _ in range(m):
#             self.layers.append(ConvBlock(s, s, 3, 1, 1, activation='prelu', norm=None))
#         # self.layers.append(nn.PReLU())
#         # Expanding
#         self.layers.append(ConvBlock(s, d, 1, 1, 0, activation='prelu', norm=None))

#         self.mid_part = torch.nn.Sequential(*self.layers)

#         # Deconvolution
#         self.last_part = nn.ConvTranspose2d(d, num_channels, 9, scale_factor, 3, output_padding=1)



#     def forward(self, x):
#         out = self.first_part(x)
#         out = self.mid_part(out)
#         out = self.last_part(out)
#         return out

# class FSRCNN(nn.Module):
#     def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
#         super(FSRCNN, self).__init__()
#         self.first_part = nn.Sequential(
#             nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
#             nn.PReLU(d)
#         )
#         self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
#         for _ in range(m):
#             self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
#         self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
#         self.mid_part = nn.Sequential(*self.mid_part)
#         self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
#                                             output_padding=scale_factor-1)

#     def forward(self, x):
#         x = self.first_part(x)
#         x = self.mid_part(x)
#         x = self.last_part(x)
#         return x


# class FSRCNN(nn.Module):
#     def __init__(self, Channel=4, k=3, scale_factor=4, num_channels=1, d=56, s=12, m=4):
#         super().__init__()
#         self.channel = Channel
#         self.pixel_shuffle = nn.PixelShuffle(scale_factor)
#         self.softmax = nn.Softmax(dim=1)
#         self.filter = nn.Conv2d(
#         in_channels=1,
#         out_channels=2*scale_factor*scale_factor*Channel,
#         kernel_size=k,
#         stride=1,
#         padding=(k-1)//2,
#         bias=False,
#     )

#     def forward(self, input):
#         filtered = self.pixel_shuffle(
#             self.filter(input)
#         )
#         B, n, H, W = filtered.shape

#         filtered = filtered.view(B, 2, self.channel, H, W)
#         upscaling = filtered[:, 0]
#         matching = filtered[:, 1]
#         return torch.sum(
#             upscaling * self.softmax(matching),
#             dim=1, keepdim=True
#         )      

# class FSRCNN(nn.Module):
#     def __init__(self, Chan=4, k=3, scale_factor=4, num_channels=1, d=56, s=12, m=4):
#         super().__init__()

#         self.pixel_shuffle = nn.PixelShuffle(scale_factor)
#         self.filter = nn.Conv2d(
#         in_channels=1,
#         out_channels=scale_factor*scale_factor*Chan,
#         kernel_size=k,
#         stride=1,
#         padding=(k-1)//2,
#         bias=False,
#         )

#     def forward(self, input):
#         return self.pixel_shuffle(self.filter(input) ).max(dim=1, keepdim=True)[0]

import torch
from torch import nn as nn

class X2SR(nn.Module):
    """X2SR network structure.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
        xt_flag: use res xt structure or not, xt structure is better
        three_flag: waiting verify
    """

    def __init__(self,
                 num_in_ch=1,
                 num_out_ch=1,
                 num_feat=16,
                 inside_feat = 64, 
                 num_block=3,
                 upscale=2,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 xt_flag=False,
                 three_flag=False,
                 deploy_flag=False):
        super(X2SR, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        
        layers = []
        for _ in range(3):
            layers.append(RBX2SR())
            layers.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*layers)


        self.conv_after_body = nn.Conv2d(num_feat, upscale**2, 3, 1, 1)
        self.upsample = nn.PixelShuffle(upscale)


    def forward(self, x):
        x = self.relu(self.conv_first(x))
        
        res = self.body(x)
        res = res+x
        
        x = self.conv_after_body(res)
        x = self.upsample(x)
        return x

class RBX2SR(nn.Module):
    def __init__(self, inp_planes=16, out_planes=16, depth_multiplier=2, deploy_flag = False):
        super(RBX2SR, self).__init__()  
        self.rbr_reparam = nn.Conv2d(inp_planes, out_planes, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        y = self.rbr_reparam(x)
        return y