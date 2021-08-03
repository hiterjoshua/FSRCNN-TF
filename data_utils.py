import pathlib
import os
from PIL import Image
import numpy as np
import cv2 
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import imutils #rotating images properly
import random


def img_aug(input_img, gt_img):
    # random rotate
   
    # rotate
    if input_img.shape[0] != input_img.shape[1]:
        rotate = random.choice([0, 2])
    else:
        rotate = random.randint(0, 3)

    if rotate != 0:
        input_img = np.rot90(input_img, rotate)
        gt_img = np.rot90(gt_img, rotate)

    # horizontal flip
    if np.random.random() >= 0.5:
        input_img = cv2.flip(input_img, flipCode = 1)
        gt_img = cv2.flip(gt_img, flipCode = 1)
            
    return input_img, gt_img

def make_dataset(paths, patch_size, scale):
    """
    Python generator-style dataset. Creates low-res and corresponding high-res patches. 
    discarding the original image generate and save style, directly feed the augmented data to the network
    add by hukunlei 20210715
    """
    print('Epoch: ', len(paths), paths)
    for p in paths:
        # read 
        im = cv2.imread(p.decode(), 3).astype(np.float32)

        p = p.decode()
        hr_p_pre = os.path.split(os.path.split(p)[0])[0]
        back_name = os.path.split(p)[1]
        _x1_position = back_name.find('_x1')
        fore_name = back_name[:_x1_position]
        if scale == 4:
            hr_p = hr_p_pre + '/HR/' +  fore_name +'_x4' +  '_' + back_name.split('_')[-1]
        if scale == 2:
            hr_p = hr_p_pre + '/HR/' +  fore_name +'_x2' +  '_' + back_name.split('_')[-1]
        # print('LR: ', p, '  HR: ', hr_p)

        hr_im = cv2.imread(hr_p, 3).astype(np.float32)
        
        # convert to YCrCb (cv2 reads images in BGR!), and normalize
        im_ycc_lr = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb) / 255.0
        im_ycc_hr = cv2.cvtColor(hr_im, cv2.COLOR_BGR2YCrCb) / 255.0
        
        # only work on the luminance channel Y
        lr = im_ycc_lr[:,:,0]
        hr = im_ycc_hr[:,:,0]

        if patch_size != 0:
            max_x = im_ycc_lr.shape[1] - patch_size
            max_y = im_ycc_lr.shape[0] - patch_size
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)
            input_img = lr[y: y + patch_size, x: x + patch_size]
            gt_img = hr[y * scale: y * scale + patch_size * scale, x * scale: x * scale + patch_size * scale]

        input_img, gt_img = img_aug(input_img, gt_img)
        lr = input_img.reshape((patch_size, patch_size, 1))
        hr = gt_img.reshape((patch_size*scale, patch_size*scale, 1))
        yield lr, hr


# def make_dataset(paths, scale):
#     """
#     Python generator-style dataset. Creates low-res and corresponding high-res patches.
#     """
#     # set lr and hr sizes
#     size_lr = 10
#     if(scale == 3):
#         size_lr = 7
#     elif(scale == 4):
#         size_lr = 6
#     size_hr = size_lr * scale
    
#     for p in paths:
#         # read 
#         im = cv2.imread(p.decode(), 3).astype(np.float32)
        
#         # convert to YCrCb (cv2 reads images in BGR!), and normalize
#         im_ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb) / 255.0

#         # -- Creating LR and HR images
#         # make current image divisible by scale (because current image is the HR image)
#         im_ycc_hr = im_ycc[0:(im_ycc.shape[0] - (im_ycc.shape[0] % scale)),
#                            0:(im_ycc.shape[1] - (im_ycc.shape[1] % scale)), :]
#         im_ycc_lr = cv2.resize(im_ycc_hr, (int(im_ycc_hr.shape[1] / scale), int(im_ycc_hr.shape[0] / scale)), 
#                            interpolation=cv2.INTER_CUBIC)
        
#         # only work on the luminance channel Y
#         lr = im_ycc_lr[:,:,0]
#         hr = im_ycc_hr[:,:,0]
        
#         numx = int(lr.shape[0] / size_lr)
#         numy = int(lr.shape[1] / size_lr)
        
#         for i in range(0, numx):
#             startx = i * size_lr
#             endx = (i * size_lr) + size_lr
            
#             startx_hr = i * size_hr
#             endx_hr = (i * size_hr) + size_hr
            
#             for j in range(0, numy):
#                 starty = j * size_lr
#                 endy = (j * size_lr) + size_lr
#                 starty_hr = j * size_hr
#                 endy_hr = (j * size_hr) + size_hr

#                 crop_lr = lr[startx:endx, starty:endy]
#                 crop_hr = hr[startx_hr:endx_hr, starty_hr:endy_hr]
        
#                 x = crop_lr.reshape((size_lr, size_lr, 1))
#                 y = crop_hr.reshape((size_hr, size_hr, 1))
#                 yield x, y

def make_val_dataset(paths, scale):
    """
    Python generator-style dataset for the validation set. Creates input and ground-truth.
    """
    for p in paths:
        # read
        im = cv2.imread(p.decode(), 3).astype(np.float32)
        # convert to YCrCb (cv2 reads images in BGR!), and normalize
        im_ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb) / 255.0
        
        # make current image divisible by scale (because current image is the HR image)
        im_ycc_hr = im_ycc[0:(im_ycc.shape[0] - (im_ycc.shape[0] % scale)),
                           0:(im_ycc.shape[1] - (im_ycc.shape[1] % scale)), :]
        im_ycc_lr = cv2.resize(im_ycc_hr, (int(im_ycc_hr.shape[1] / scale), int(im_ycc_hr.shape[0] / scale)), 
                           interpolation=cv2.INTER_CUBIC)
        
        # only work on the luminance channel Y
        lr = np.expand_dims(im_ycc_lr[:,:,0], axis=2)
        hr = np.expand_dims(im_ycc_hr[:,:,0], axis=2)
        
        yield lr, hr

def getpaths(path):
    """
    Get all image paths from folder 'path'
    """
    data = pathlib.Path(path)
    all_image_paths = list(data.glob('*'))
    all_image_paths = [str(p) for p in all_image_paths]
    return all_image_paths

def augment(dataset_path, save_path):
    if(not os.path.isdir(save_path)):
        print("Making augmented images...")
        os.mkdir(save_path)

        do_augmentations(dataset_path, save_path)
        
        #count new images
        save_path, dirs, files = next(os.walk(save_path))
        file_count = len(files)
        
        print("{} augmented images are stored in the folder {}".format(file_count, save_path))

def rotate(img):
    """
    Function that rotates an image 90 degrees 4 times.

    returns:
    4 image arrays each rotated 90 degrees
    """
    rotated90 = imutils.rotate_bound(img, 90)
    rotated180 = imutils.rotate_bound(img, 180)
    rotated270 = imutils.rotate_bound(img, 270)

    return img, rotated90, rotated180, rotated270

def downscale(img):
    """
    Downscales an image 0.9x, 0.8x, 0.7x and 0.6x.

    Returns:
    5 image arrays
    """
    (w, h) = img.shape[:2]
    img09 = cv2.resize(img, dsize=(int(h*0.9),int(w*0.9)), interpolation=cv2.INTER_CUBIC)
    img08 = cv2.resize(img, dsize=(int(h*0.8),int(w*0.8)), interpolation=cv2.INTER_CUBIC)
    img07 = cv2.resize(img, dsize=(int(h*0.7),int(w*0.7)), interpolation=cv2.INTER_CUBIC)
    img06 = cv2.resize(img, dsize=(int(h*0.6),int(w*0.6)), interpolation=cv2.INTER_CUBIC)

    return img, img09, img08, img07, img06


def augment_image(img):
    """
    Rotates and downscales an image. Creates 20x images.
    """
    augmented_images = []

    rotated_images = rotate(img)
    
    for img in rotated_images:
        downscaled_images = downscale(img)
        
        for im in downscaled_images:
            augmented_images.append(im)

    return augmented_images

def do_augmentations(dataset_path, save_path):
    """
    Does augmentations on all images in folder 'path'.
    """
    # get all image paths from folder
    dir = pathlib.Path(dataset_path)
    all_image_paths = list(dir.glob('*'))
    all_image_paths = [str(x) for x in all_image_paths]

    im_counter = 0
    # do augmentations
    for path in all_image_paths:
        # open current image as array
        img = Image.open(path)
        img = np.array(img)

        augm_counter = 0
        # get augmented images
        augmented_images = augment_image(img)
        for im in augmented_images: #save them all to ./augmented
            x = Image.fromarray(im)
            x.save(save_path + "/img{}aug{}.png".format(im_counter, augm_counter))
            augm_counter += 1
        im_counter += 1