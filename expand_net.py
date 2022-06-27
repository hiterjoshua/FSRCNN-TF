import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def compute_cl(s_1, s_2):
    """
    Compute weights from s_1 and s_2
    :param s_1: 1*1 conv layer
    :param s_2: 3*3 conv layer
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

    w_s_1_ = w_s_1.view(w_s_1.size(0), w_s_1.size(1) * w_s_1.size(2) * w_s_1.size(3))
    w_s_2_tmp = w_s_2.view(w_s_2.size(0), w_s_2.size(1),  w_s_2.size(2) * w_s_2.size(3))

    new_weight = torch.Tensor(w_s_2.size(0), w_s_1.size(1), w_s_2.size(2)*w_s_2.size(3))
    for i in range(w_s_2.size(0)):
        tmp = w_s_2_tmp[i, :, :].view( w_s_2.size(1),  w_s_2.size(2) * w_s_2.size(3))
        new_weight[i, :, :] = torch.matmul(w_s_1_.t(), tmp)
    new_weight = new_weight.view(w_s_2.size(0), w_s_1.size(1),  w_s_2.size(2), w_s_2.size(3))

    if b_s_1 is not None and b_s_2 is not None:
        b_sum = torch.sum(w_s_2_tmp, dim=2)
        new_bias = torch.matmul(b_sum, b_s_1) + b_s_2  # with bias
    elif b_s_1 is None and b_s_2 is not None:
        new_bias = b_s_2  #without Bias
    else:
        new_bias = torch.zeros(s_2.weight.size(0))

    return {'weight': new_weight, 'bias': new_bias}


def compute_cl_2(s_1, s_2):
    """
    compute weights from former computation and last 1*1 conv layer
    :param s_1: 3*3 conv layer
    :param s_2: 1*1 conv layer
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

    w_s_1_ = w_s_1.view(w_s_1.size(0), w_s_1.size(1),  w_s_1.size(2) * w_s_1.size(3))
    w_s_2_ = w_s_2.view(w_s_2.size(0), w_s_2.size(1) * w_s_2.size(2) * w_s_2.size(3))
    new_weight_ = torch.Tensor(w_s_2.size(0), w_s_1.size(1), w_s_1.size(2)*w_s_1.size(3))
    for i in range(w_s_1.size(1)):
        tmp = w_s_1_[:, i, :].view(w_s_1.size(0),  w_s_1.size(2) * w_s_1.size(3))
        new_weight_[:, i, :] = torch.matmul(w_s_2_, tmp)
    new_weight = new_weight_.view(w_s_2.size(0), w_s_1.size(1),  w_s_1.size(2), w_s_1.size(3))

    if b_s_1 is not None and b_s_2 is not None:
        new_bias = torch.matmul(w_s_2_, b_s_1) + b_s_2  # with bias
    elif b_s_1 is None and b_s_2 is not None:
        new_bias = b_s_2  #without Bias
    else:
        new_bias = torch.zeros(s_2.weight.size(0))

    return {'weight': new_weight, 'bias': new_bias}


def compute_ck(s_1, s_2):
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

    w_s_2_tmp = w_s_2.view(w_s_2.size(0), w_s_2.size(1), w_s_2.size(2) * w_s_2.size(3))

    if b_s_1 is not None and b_s_2 is not None:
        b_sum = torch.sum(w_s_2_tmp, dim=2)
        new_bias = torch.matmul(b_sum, b_s_1) + b_s_2
    elif b_s_1 is None and b_s_2 is not None:
        new_bias = b_s_2  #without Bias
    else:
        new_bias = torch.zeros(s_2.weight.size(0))

    new_weight = F.conv_transpose2d(w_s_2, w_s_1)

    return {'weight': new_weight, 'bias': new_bias}


# Testing
# we need to turn off gradient calculation because we didn't write it
torch.set_grad_enabled(False)
squeezenet1_1 = torchvision.models.squeezenet1_1(pretrained=True)
# removing all learning variables, etc
squeezenet1_1.eval()

# # 3*3 and 3*3
# input = torch.randn(1, 16, 256, 256)
# squeezenet1_1.features[3].expand3x3.padding=0
# squeezenet1_1.features[12].expand3x3.padding=0
# model_linear = torch.nn.Sequential(
#     nn.ZeroPad2d(2),
#     squeezenet1_1.features[3].expand3x3,
#     squeezenet1_1.features[12].expand3x3
# )
# f1 = model_linear.forward(input)
# fused = torch.nn.Conv2d(
#     squeezenet1_1.features[3].expand3x3.in_channels,
#     squeezenet1_1.features[12].expand3x3.out_channels,
#     kernel_size=squeezenet1_1.features[3].expand3x3.kernel_size,
#     stride=squeezenet1_1.features[3].expand3x3.stride,
#     padding= 5//2,
#     bias=True
# )
# res = compute_ck(model_linear[1], model_linear[2])
# fused.weight.data = res['weight']
# fused.bias.data = res['bias']
# f2 = fused.forward(input)
# d = torch.mean(torch.abs(f1 - f2))
# print("error 3*3 and 3*3:",d)


# # 3*3 and 1*1
# input = torch.randn(1, 16, 256, 256)
# model_linear = torch.nn.Sequential(
#     squeezenet1_1.features[3].expand3x3,
#     squeezenet1_1.features[3].squeeze
# )
# f1 = model_linear.forward(input)
# fused = torch.nn.Conv2d(
#     squeezenet1_1.features[3].expand3x3.in_channels,
#     squeezenet1_1.features[3].squeeze.out_channels,
#     kernel_size=3,
#     stride=squeezenet1_1.features[3].expand3x3.stride,
#     padding=1,
#     bias=True
# )
# res = compute_cl_2(model_linear[0], model_linear[1])
# fused.weight.data = res['weight']
# fused.bias.data = res['bias']
# f2 = fused.forward(input)
# d = torch.mean(torch.abs(f1 - f2))
# print("error 3*3 and 1*1:",d)


# 1*1 and 3*3
squeezenet1_1.features[3].expand3x3.padding=0
model_linear = torch.nn.Sequential(
    nn.ZeroPad2d(1),
    squeezenet1_1.features[3].squeeze,
    squeezenet1_1.features[3].expand3x3
)
input = torch.randn(1, 64, 256, 256)
f1 = model_linear.forward(input)
fused = torch.nn.Conv2d(
    squeezenet1_1.features[3].squeeze.in_channels,
    squeezenet1_1.features[3].expand3x3.out_channels,
    kernel_size=squeezenet1_1.features[3].expand3x3.kernel_size,
    stride=squeezenet1_1.features[3].expand3x3.stride,
    padding=1,
    bias=True
)
res = compute_cl(model_linear[1], model_linear[2])
fused.weight.data = res['weight']
fused.bias.data = res['bias']
f2 = fused.forward(input)
d = torch.mean(torch.abs(f1 - f2))
print("error 1*1 and 3*3:",d)


# 1*1 and 3*3 plus shortcut
squeezenet1_1.features[3].expand3x3.padding=0
model_linear = torch.nn.Sequential(
    nn.ZeroPad2d(1),
    squeezenet1_1.features[3].squeeze,
    squeezenet1_1.features[3].expand3x3
)
input = torch.randn(1, 64, 256, 256)
f1 = model_linear.forward(input) + input
fused = torch.nn.Conv2d(
    squeezenet1_1.features[3].squeeze.in_channels,
    squeezenet1_1.features[3].expand3x3.out_channels,
    kernel_size=squeezenet1_1.features[3].expand3x3.kernel_size,
    stride=squeezenet1_1.features[3].expand3x3.stride,
    padding=1,
    bias=True
)
res = compute_cl(model_linear[1], model_linear[2])
kernel_identity = torch.zeros((64, 64, 3, 3))
for i in range(64):
    kernel_identity[i, i, 1, 1] = 1
fused.weight.data = res['weight'] + kernel_identity
fused.bias.data = res['bias']
f2 = fused.forward(input)
d = torch.mean(torch.abs(f1 - f2))
print("error 1*1 and 3*3 plus shortcut:",d)

# 1*1, 3*3, 3*3, 1*1 
squeezenet1_1.features[3].expand3x3.padding=0
squeezenet1_1.features[12].expand3x3.padding=0
model_linear = torch.nn.Sequential(
    nn.ZeroPad2d(2),
    squeezenet1_1.features[3].squeeze,
    squeezenet1_1.features[3].expand3x3,
    squeezenet1_1.features[12].expand3x3,
    squeezenet1_1.features[9].squeeze
)
input = torch.randn(1, 64, 256, 256)
f1 = model_linear.forward(input) 
fused = torch.nn.Conv2d(
    squeezenet1_1.features[3].squeeze.in_channels,
    squeezenet1_1.features[9].squeeze.out_channels,
    kernel_size=5,
    stride=squeezenet1_1.features[3].expand3x3.stride,
    padding=5//2,
    bias=True
)
res = compute_cl(model_linear[1], model_linear[2])
fused.weight.data = res['weight']
fused.bias.data = res['bias']
res = compute_ck(fused, model_linear[3])
fused.weight.data = res['weight']
fused.bias.data = res['bias']
res = compute_ck(fused, model_linear[4])
fused.weight.data = res['weight'] 
fused.bias.data = res['bias']
f2 = fused.forward(input)
d = torch.mean(torch.abs(f1 - f2))
print("error 1*1, 3*3, 3*3, 1*1 :",d)

'''
ipdb> squeezenet1_1.features[3].squeeze
Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
ipdb> squeezenet1_1.features[3].expand3x3
Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ipdb> squeezenet1_1.features[12].expand3x3
Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ipdb> squeezenet1_1.features[9].squeeze
Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1))
'''