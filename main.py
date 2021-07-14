from model import FSRCNN

import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pprint
import os

flags = tf.app.flags
flags.DEFINE_boolean("fast", False, "Use the fast model (FSRCNN-s) [False]")
flags.DEFINE_integer("epoch", 10, "Number of epochs [10]")
flags.DEFINE_integer("batch_size", 8, "The size of batch images [8]")
flags.DEFINE_float("learning_rate", 1e-3, "The learning rate of gradient descent algorithm [1e-3]")
flags.DEFINE_float("momentum", 0.9, "The momentum value for the momentum SGD [0.9]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color [1]")
flags.DEFINE_integer("scale", 4, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 4, "The size of stride to apply to input image [4]")
flags.DEFINE_string("checkpoint_dir", "checkpoint-real2w", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("output_dir", "/data1/hukunlei/result/output/fsr-tf", "Name of test output directory [result]")
flags.DEFINE_string("data_dir", "Train", "Name of data directory to train on [FastTrain]")
flags.DEFINE_boolean("train", False, "True for training, false for testing [True]")
flags.DEFINE_integer("threads", 1, "Number of processes to pre-process data with [1]")
flags.DEFINE_boolean("params", False, "Save weight and bias parameters [False]")

#addby hukunlei
flags.DEFINE_boolean("real3w", True, "Use Real3w dataset for training [False]")
flags.DEFINE_boolean("use_h5", True, "Use pre-extracted dataset for training [False]")
flags.DEFINE_string("dataset", "/data1/datasets/d_realsr_3w/", "Name of data directory to train on real3w")
flags.DEFINE_string("test_data", "/data1/hukunlei/result/validation/sr_0415", "Name of data directory to train on real3w")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.fast:
        FLAGS.checkpoint_dir = 'fast_{}'.format(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)


    with tf.Session() as sess:
        fsrcnn = FSRCNN(sess, config=FLAGS)
        fsrcnn.run()
   
        
if __name__ == '__main__':
    tf.app.run()
