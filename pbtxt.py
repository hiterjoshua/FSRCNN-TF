 
import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format
import os

# def convert_pb_to_pbtxt(filename_tf, filename_tx):
#     with gfile.FastGFile(filename_tf, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         tf.import_graph_def(graph_def, name='')
#         tf.train.write_graph(graph_def, "./pb/", "my_model.pb", as_text=True)
#     return
    
 
def convert_pb_to_pbtxt(filename_tf, filename_txt):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        #output_graph_path = "./pb/my_model.pb"
        path_list = os.path.split(filename_txt)

        with open(filename_tf, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
            tf.train.write_graph(output_graph_def, path_list[0], path_list[1], as_text=True)
    print("===== Transfer from pb to pbtxt sucessfully!!! =====")


def convert_pbtxt_to_pb(filename_tf, filename_txt):
    """Returns a `tf.GraphDef` proto representing the data in the given pbtxt file.
    Args:
        filename: The name of a file containing a GraphDef pbtxt (text-formatted
        `tf.GraphDef` protocol buffer data).
    """
    path_list = os.path.split(filename_tf)
    with tf.gfile.FastGFile(filename_txt, 'r') as f:
        graph_def = tf.GraphDef()
        file_content = f.read()
    
        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def , path_list[0], path_list[1], as_text = False)
    print("===== Transfer to pb from pbtxt sucessfully!!! =====")


# Export pb model to pbtxt
filename_txt = "./pb/my_model.pbtxt"
filename_tf = "./pb/my_model.pb"
#convert_pb_to_pbtxt(filename_tf, filename_txt)
convert_pbtxt_to_pb(filename_tf, filename_txt)
