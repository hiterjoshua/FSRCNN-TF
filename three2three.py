import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F


def reparameter_33(s_1, s_2):
    """
    Compute weight from 2 conv layers, whose kernel size larger than 3*3
    After derivation, F.conv_transpose2d can be used to compute weight of original conv layer
    :param s_1: 3*3 or larger conv layer
    :param s_2: 3*3 or larger conv layer
    :return: new weight and bias
    """
    if isinstance(s_1, nn.Conv2d):
        w_s_1 = s_1.weight  # output# * input# * kernel * kernel
        b_s_1 = s_1.bias
    else:
        w_s_1 = s_1['weight']
        b_s_1 = s_1['bias']
    if isinstance(s_2, nn.Conv2d):
        w_s_2 = s_2.weight
        b_s_2 = s_2.bias
    else:
        w_s_2 = s_2['weight']
        b_s_2 = s_2['bias']
    
    fused = torch.nn.Conv2d(
        s_1.in_channels,
        s_2.out_channels,
        kernel_size=s_2.kernel_size,
        stride=s_2.stride,
        padding=1,
        bias=True
    )

    w_s_2_tmp = w_s_2.view(w_s_2.size(0), w_s_2.size(1), w_s_2.size(2) * w_s_2.size(3))

    if b_s_1 is not None and b_s_2 is not None:
        b_sum = torch.sum(w_s_2_tmp, dim=2)
        new_bias = torch.matmul(b_sum, b_s_1) + b_s_2
    elif b_s_1 is None and b_s_2 is not None:
        new_bias = b_s_2  #without Bias
    else:
        new_bias = torch.zeros(s_2.weight.size(0))

    new_weight = F.conv_transpose2d(w_s_2, w_s_1)
    print(new_weight.shape)

    fused.weight.data = new_weight
    fused.bias.data = new_bias
    return fused

m = nn.ZeroPad2d(3) 

ori_3_3_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride = 1, padding = 0, bias=True)
ori_3_3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 0, bias=True)
ori_3_3_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 0, bias=True)



reweight_3_3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=7//2, bias=True)


temp2 = reparameter_33(ori_3_3_1, ori_3_3_2)
print(temp2.weight.data.shape)

temp3 = reparameter_33(temp2, ori_3_3_3)

reweight_3_3.weight.data = temp3.weight.data.clone()
reweight_3_3.bias.data = temp3.bias.data.clone()

model = nn.Sequential(m, ori_3_3_1, ori_3_3_2,ori_3_3_3)
model2 = nn.Sequential(reweight_3_3)

#x = torch.tensor(np.array(Image.open('0802x4.png'))).unsqueeze(0).permute(0,3,1,2).float()
#x = x/255.0
x = torch.randn(1,3,2560,2560)
out = model(x)
out2 = model2(x)

print(out.shape)
print(out2.shape)

print(torch.mean(torch.abs(out - out2)))


