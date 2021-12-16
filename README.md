# FSRCNN_Tensorflow

This a officially tensorflow version FSRCNN code

* when train on tmo datasets, change data_util.py:54 to be true
* when load previous models, change main.py:83 file position to use the ckpt_path_pretrain paramerter. you can decide ckpt_path_pretrain for which folder and function train:78 to decide which specific model for load. 


* previously, the pretrained model usually set the learnning rate to be 0.001, when finetuned on face or scenery datasets, the lr was set to 5e-6 or e-5
