import tensorflow as tf
import fsrcnn
import data_utils
import run
import os
import cv2
import numpy as np
import pathlib
import argparse
from PIL import Image
import numpy
from tensorflow.python.client import device_lib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #gets rid of avx/fma warning
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# TODO: 
# Overlapping patches
# seperate learning rate for deconv layer
# switch out deconv layer for different models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=False, help='Train the model')
    parser.add_argument('--test', type=str, default=True, help='Run tests on the model')
    parser.add_argument('--export', type=str, default=True, help='Export the model as .pb')
    parser.add_argument('--load_flag', type=str, default=True, help='Load previous model for training')
    parser.add_argument('--finetune', type=str, default=False, help='Finetune model on General100 dataset')
    parser.add_argument('--small', help='Run FSRCNN-small', action="store_true")
    
    parser.add_argument('--scale', type=int, help='Scaling factor of the model', default=4)
    parser.add_argument('--batch', type=int, help='Batch size of the training', default=8)
    parser.add_argument('--patch_size', type=int, help='cropped patch size of the image', default=128)
    parser.add_argument('--epochs', type=int, help='Number of epochs during training', default=300)
    parser.add_argument('--lr', type=float, help='Learning_rate', default=0.001)
    parser.add_argument('--d', type=int, help='Variable for d', default=56)
    parser.add_argument('--s', type=int, help='Variable for s', default=12)
    parser.add_argument('--m', type=int, help='Variable for m', default=1) #4 for 9layers and 1 for 5layers
    
    parser.add_argument('--traindir', type=str, default="/data1/datasets/d_realsr_3w/LR/", help='Path to train images')
    parser.add_argument('--ckpt_path', type=str, default="./CKPT_dir_sratch_5layers_x4/", help='model output path')\
    #"./CKPT_dir_l2/" "./CKPT_dir_l1/" "./CKPT_dir_scratch/"          "./CKPT_dir_pretrain/"
    parser.add_argument('--finetunedir', type=str, default='/data1/datasets/d_realsr_3w/', help='Path to finetune images')
    parser.add_argument('--validdir', type=str, default='./images/', help='Path to validation images')
    #parser.add_argument('--image', type=str, default="/data1/hukunlei/result/validation/sr_0415", help='Specify test image')
    parser.add_argument('--image', type=str, default="/data1/hukunlei/result/validation/d_realsr_3w/", help='Specify test image')
    parser.add_argument('--output_path', type=str, default="/data1/hukunlei/result/output/CKPT_dir_5layers_x4_220", help='Path to test output')

    
    args = parser.parse_args()

    # INIT
    scale = args.scale
    fsrcnn_params = (args.d, args.s, args.m) #d,s,m
    traindir = args.traindir

    augmented_path = "./augmented"
    small = args.small

    lr_size = 10
    if(scale == 3):
        lr_size = 7
    elif(scale == 4):
        lr_size = 6
        
    hr_size = lr_size * scale
    
    # FSRCNN-small
    if small:
        fsrcnn_params = (32, 5, 1)

    # Set checkpoint paths for different scales and models

    if scale == 2:
        ckpt_path_pretrain = "./CKPT_dir_pretrain/x2/"
        if small:
            ckpt_path_pretrain = "./CKPT_dir_pretrain/x2_small/"
    elif scale == 3:
        ckpt_path_pretrain = "./CKPT_dir_pretrain/x3/"
        if small:
            ckpt_path_pretrain = "./CKPT_dir_pretrain/x3_small/"
    elif scale == 4:
        ckpt_path_pretrain = "./CKPT_dir_sratch_step/"   #load previous model and relay training
        if small:
            ckpt_path_pretrain = "./CKPT_dir_pretrain/x4_small/"
    else:
        print("Upscale factor scale is not supported. Choose 2, 3 or 4.")
        exit()
    
    # Set gpu 
    config = tf.ConfigProto() #log_device_placement=True
    config.gpu_options.allow_growth = True

    # Create run instance
    run = run.run(config, lr_size, ckpt_path_pretrain, scale, fsrcnn_params, small, args)

    if args.train:
        # if finetune, load model and train on general100
        if args.finetune:
            traindir = args.finetunedir
            augmented_path = "./augmented_general100"

        # augment (if not done before) and then load images 
        # data_utils.augment(traindir, save_path=augmented_path)
        #run.train(augmented_path)

        run.train(args.traindir)

    if args.test:
        #run.testFromPb(args.image)
        run.test(args.image)
        #run.upscale(args.image)

    if args.export:
        run.export()
    
    print("I ran successfully.")