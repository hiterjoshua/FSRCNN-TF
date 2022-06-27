import tensorflow as tf
import torch
import cv2
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
 
#with tf.device('/cpu:3'):
with tf.device('/device:GPU:3'):
    image = "./4.0_112_1.00_x4.jpg"
    image_np = cv2.imread(image)
    img_yuv = cv2.cvtColor(image_np, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)

    input_data = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)
    #input_data = input_data.cuda()
   
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    output_graph_path = "./pb/my_model.pb"

    with open(output_graph_path, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")


    config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=False)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        input = sess.graph.get_tensor_by_name("input:0")  # "input" 是在pth文件转为onnx文件时定义好的，名字要一致
        output = sess.graph.get_tensor_by_name("output:0") # ”output“ 也是
        #input_data =  torch.randn(1,1,256,256).cpu() # 输入要测试的数据，格式要一致
    

        predictions = sess.run(output, feed_dict={input: input_data})
        print("predictions:", predictions)

        sr_y = predictions[0][0].astype(np.uint8)
        sr_u = cv2.resize(u, (int(u.shape[1] *4), int(u.shape[0]*4)))
        sr_v = cv2.resize(v, (int(v.shape[1] *4), int(v.shape[0]*4)))
        yuv = cv2.merge([sr_y, sr_u ,sr_v])
        img_sr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        cv2.imwrite('sr.png', img_sr)




# import tensorflow as tf
# from torchvision import transforms
# import numpy as np
# from PIL import Image

# transform = transforms.Compose([
#     transforms.Resize((200, 200)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])


# with tf.Graph().as_default():
#     # output_graph_def = tf.GraphDef() #tensorflow1.4版本
#     output_graph_def = tf.compat.v1.GraphDef()
#     output_graph_path = ‘my.pb'
#     with open(output_graph_path, 'rb') as f:
#         output_graph_def.ParseFromString(f.read())
#         _ = tf.import_graph_def(output_graph_def, name="")
#     # with tf.Session() as sess:  #tensorflow1.4版本
#     with tf.compat.v1.Session() as sess:
#         image = “demo.jpg"
#         image_np = Image.open(image)
#         img_input = transform(image_np).unsqueeze(0)
#         image_np_expanded = img_input.numpy()
#         # sess.run(tf.global_variables_initializer())  #tensorflow1.4版本
#         sess.run(tf.compat.v1.global_variables_initializer())
#         input = sess.graph.get_tensor_by_name("head_input:0")
#         output = sess.graph.get_tensor_by_name("output:0")
#         predictions = sess.run(output, feed_dict={input: image_np_expanded})
#         index = np.argmax(predictions)
#         print("predictions:", predictions)
#         print("index:", index)

