import torch
import torch.onnx
import onnx
# import onnx_tf
from onnx_tf.backend import prepare
import network 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# create random input
input_data = torch.randn(1,1,750,1000).cuda()

# create network
#model = network.FSRCNN(num_channels=1, scale_factor=4, d=56, s=12, m=4)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#model.load_state_dict(torch.load('./pth/face_grass_hs_b_no_pretrain_epoch300_batchsize4.pth', map_location=device))
model = torch.load('./pth/epoch71_batchsize8-TM-x4.pth')
model.eval()

# Forward pass
output = model(input_data)

# 设置输入张量名，多个输入就是多个名
input_names = ["input"]
# 设置输出张量名
output_names = ["output"]

# Export model to onnx
onnx_path = "./onnx/"
tf_path = "./pb/"
if not os.path.exists(onnx_path):
    os.mkdir(onnx_path)
if not os.path.exists(tf_path):
    os.mkdir(tf_path)
filename_onnx = onnx_path + "epoch71_batchsize8-TM-x4.onnx"
filename_tf = tf_path + "epoch71_batchsize8-TM-x4.pb"

torch.onnx.export(model, input_data, filename_onnx, input_names=input_names, 
output_names=output_names, opset_version=11, do_constant_folding=True)

# Export model to tensorflow
onnx_model = onnx.load(filename_onnx)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(filename_tf)