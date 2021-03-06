from utils import (
    read_data, 
    thread_train_setup,
    train_input_setup,
    test_input_setup,
    save_params,
    merge,
    array_image_save
  )

import time
import os
import numpy as np

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from PIL import Image
import pdb
import cv2

# Based on http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html
class FSRCNN(object):
  
    def __init__(self, sess, config):
        self.sess = sess
        self.fast = config.fast
        self.train = config.train
        self.c_dim = config.c_dim
        self.is_grayscale = (self.c_dim == 1)
        self.epoch = config.epoch
        self.scale = config.scale
        self.stride = config.stride
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.momentum = config.momentum
        self.threads = config.threads
        self.params = config.params

        self.dataset = config.dataset
        self.real3w = config.real3w
        self.use_h5 = config.use_h5
        self.test_data = config.test_data

        # Different image/label sub-sizes for different scaling factors x2, x3, x4
        scale_factors = [[14, 20], [11, 21], [10, 24]]
        self.image_size, self.label_size = scale_factors[self.scale - 2]
        # Testing uses different strides to ensure sub-images line up correctly
        if not self.train:
            self.stride = [10, 7, 6][self.scale - 2]
        if self.real3w:
            self.image_size, self.label_size = 194, 776

        # Different model layer counts and filter sizes for FSRCNN vs FSRCNN-s (fast), (s, d, m) in paper
        model_params = [[56, 12, 4], [32, 5, 1]]
        self.model_params = model_params[self.fast]
        
        self.checkpoint_dir = config.checkpoint_dir
        self.output_dir = config.output_dir
        self.data_dir = config.data_dir
        self.build_model()


    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
        # Batch size differs in training vs testing
        self.batch = tf.placeholder(tf.int32, shape=[], name='batch')

        # FSCRNN-s (fast) has smaller filters and less layers but can achieve faster performance
        s, d, m = self.model_params

        expand_weight, deconv_weight = 'w{}'.format(m + 3), 'w{}'.format(m + 4)
        self.weights = {
            'w1': tf.Variable(tf.random_normal([5, 5, 1, s], stddev=0.0378, dtype=tf.float32), name='w1'),
            'w2': tf.Variable(tf.random_normal([1, 1, s, d], stddev=0.3536, dtype=tf.float32), name='w2'),
            expand_weight: tf.Variable(tf.random_normal([1, 1, d, s], stddev=0.189, dtype=tf.float32), name=expand_weight),
            deconv_weight: tf.Variable(tf.random_normal([9, 9, 1, s], stddev=0.0001, dtype=tf.float32), name=deconv_weight)
        }

        expand_bias, deconv_bias = 'b{}'.format(m + 3), 'b{}'.format(m + 4)
        self.biases = {
            'b1': tf.Variable(tf.zeros([s]), name='b1'),
            'b2': tf.Variable(tf.zeros([d]), name='b2'),
            expand_bias: tf.Variable(tf.zeros([s]), name=expand_bias),
            deconv_bias: tf.Variable(tf.zeros([1]), name=deconv_bias)
        }

        # Create the m mapping layers weights/biases
        for i in range(3, m + 3):
            weight_name, bias_name = 'w{}'.format(i), 'b{}'.format(i)
            self.weights[weight_name] = tf.Variable(tf.random_normal([3, 3, d, d], stddev=0.1179, dtype=tf.float32), name=weight_name)
            self.biases[bias_name] = tf.Variable(tf.zeros([d]), name=bias_name)

        self.pred = self.model()

        # Loss function (MSE)
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.labels - self.pred), reduction_indices=0))
        self.loss += 1 - tf.reduce_mean(tf.image.ssim(self.labels, self.pred, max_val=1.0)) #addby hukunlei 20210712
        
        self.saver = tf.train.Saver()

    def run(self):
        # SGD with momentum
        self.train_op = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.loss)

        tf.initialize_all_variables().run()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if self.params:
            save_params(self.sess, self.weights, self.biases)
        elif self.train:
            self.run_train()
        else:
            self.run_test_file()

    def run_train(self):
        write = tf.summary.FileWriter("logs/",self.sess.graph)
        start_time = time.time()
        print("Beginning training setup...")
        if not self.use_h5:
            if self.threads == 1:
                train_input_setup(self)
            else:
                thread_train_setup(self)
        print("Training setup took {} seconds with {} threads".format(time.time() - start_time, self.threads))

        data_dir = os.path.join('./{}'.format(self.checkpoint_dir), "train.h5")
        train_data, train_label = read_data(data_dir)
        print("Total setup time took {} seconds with {} threads".format(time.time() - start_time, self.threads))

        print("Training...")
        start_time = time.time()
        start_average, end_average, counter = 0, 0, 0

        #merged = tf.summary.merge_all()

        for ep in range(self.epoch):
            # Run by batch images
            batch_idxs = len(train_data) // self.batch_size
            batch_average = 0
            for idx in range(0, batch_idxs):
                batch_images = train_data[idx * self.batch_size : (idx + 1) * self.batch_size]
                batch_labels = train_label[idx * self.batch_size : (idx + 1) * self.batch_size]

                counter += 1
                _, err = self.sess.run([self.train_op, self.loss], \
                         feed_dict={self.images: batch_images, self.labels: batch_labels, self.batch: self.batch_size})
                batch_average += err

                if counter % 10 == 0:
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                        % ((ep+1), counter, time.time() - start_time, err))
                #tf.summary.scalar("loss",self.loss)

                # Save every 1000 steps
                if counter % 1000 == 0:
                    self.save(self.checkpoint_dir, counter)

                #_, result = self.sess.run(merged,feed_dict={self.images: batch_images, self.labels: batch_labels, self.batch: self.batch_size})
                #write.add_summary(result, ep*len(train_data) + idx*self.batch_size)

            batch_average = float(batch_average) / batch_idxs
            if ep < (self.epoch * 0.2):
                start_average += batch_average
            elif ep >= (self.epoch * 0.8):
                end_average += batch_average

        # Compare loss of the first 20% and the last 20% epochs
        start_average = float(start_average) / (self.epoch * 0.2)
        end_average = float(end_average) / (self.epoch * 0.2)
        print("Start Average: [%.6f], End Average: [%.6f], Improved: [%.2f%%]" \
            % (start_average, end_average, 100 - (100*end_average/start_average)))

        # Linux desktop notification when training has been completed
        # title = "Training complete - FSRCNN"
        # notification = "{}-{}-{} done training after {} epochs".format(self.image_size, self.label_size, self.stride, self.epoch);
        # notify_command = 'notify-send "{}" "{}"'.format(title, notification)
        # os.system(notify_command)

    
    def run_test(self):
        nx, ny = test_input_setup(self)
        data_dir = os.path.join('./{}'.format(self.checkpoint_dir), "test.h5")
        test_data, test_label = read_data(data_dir)

        print("Testing...")

        start_time = time.time()
        result = self.pred.eval({self.images: test_data, self.labels: test_label, self.batch: nx * ny})
        print("Took %.3f seconds" % (time.time() - start_time))

        result = merge(result, [nx, ny])
        result = result.squeeze()
        image_path = os.path.join(os.getcwd(), self.output_dir)
        image_path = os.path.join(image_path, "test_image.png")

        array_image_save(result * 255, image_path)

    def run_test_file(self):
        dirs = os.listdir(self.test_data)
        for dir in dirs:
            dir_path = os.path.join(self.test_data, dir)
            for file in os.listdir(dir_path):
                LR_path = os.path.join(dir_path, file)
                print(LR_path)
                image_LR = cv2.imread(LR_path)
                LR_yuv = cv2.cvtColor(image_LR, cv2.COLOR_BGR2YUV)
                input_, u, v = cv2.split(LR_yuv)
                input_ = input_.astype(np.float) / 255

                input_ = cv2.resize(input_, (194,194), interpolation = cv2.INTER_AREA)
                image_Label = cv2.resize(input_, (776,776), interpolation = cv2.INTER_AREA)

                input_ = input_[np.newaxis, :, : , np.newaxis]
                image_Label = image_Label[np.newaxis, :, : , np.newaxis]
                print("Testing...")

                start_time = time.time()
                sr_y = self.pred.eval({self.images: input_, self.labels: image_Label, self.batch: 1})
                print(file, "Took %.3f seconds" % (time.time() - start_time))
                sr_y = sr_y.squeeze() 
                sr_y = (sr_y*255).astype(np.uint8) 
                sr_y = cv2.resize(sr_y, (int(u.shape[1] *4), int(u.shape[0]*4))) 

                sr_u = cv2.resize(u, (int(u.shape[1] *4), int(u.shape[0]*4)))
                sr_v = cv2.resize(v, (int(v.shape[1] *4), int(v.shape[0]*4)))
                yuv = cv2.merge([sr_y, sr_u ,sr_v])
                img_sr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                
                dir_path = os.path.join(self.output_dir, dir)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                image_path = os.path.join(dir_path, file)

                array_image_save(img_sr * 255, image_path)

    def model(self):
        # Feature Extraction change from VALID to SAME by hukunlei20210712
        conv_feature = self.prelu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'], 1)

        # Shrinking
        conv_shrink = self.prelu(tf.nn.conv2d(conv_feature, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'], 2)

        # Mapping (# mapping layers = m)
        prev_layer, m = conv_shrink, self.model_params[2]
        for i in range(3, m + 3):
            weights, biases = self.weights['w{}'.format(i)], self.biases['b{}'.format(i)]
            prev_layer = self.prelu(tf.nn.conv2d(prev_layer, weights, strides=[1,1,1,1], padding='SAME') + biases, i)

        # Expanding
        expand_weights, expand_biases = self.weights['w{}'.format(m + 3)], self.biases['b{}'.format(m + 3)]
        conv_expand = self.prelu(tf.nn.conv2d(prev_layer, expand_weights, strides=[1,1,1,1], padding='SAME') + expand_biases, 7)

        # Deconvolution
        deconv_output = [self.batch, self.label_size, self.label_size, self.c_dim]
        deconv_stride = [1,  self.scale, self.scale, 1]
        deconv_weights, deconv_biases = self.weights['w{}'.format(m + 4)], self.biases['b{}'.format(m + 4)]
        conv_deconv = tf.nn.conv2d_transpose(conv_expand, deconv_weights, output_shape=deconv_output, strides=deconv_stride, padding='SAME') + deconv_biases

        return conv_deconv

    def prelu(self, _x, i):
        """
        PreLU tensorflow implementation
        """
        alphas = tf.get_variable('alpha{}'.format(i), _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg

    def save(self, checkpoint_dir, step):
        model_name = "FSRCNN.model"
        model_dir = "%s_%s" % ("fsrcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("fsrcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False