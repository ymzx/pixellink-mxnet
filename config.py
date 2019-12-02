# -*- coding: utf-8 -*-
# @Time    : 2019/7/5
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : config.py
# @Software: PyCharm

use_rotate = False
use_crop = False
link_weight = 1
pixel_weight = 2

# r_mean,g_mean,b_mean
r_mean = 123.0
g_mean = 117.0
b_mean = 104.0

train_images_dir = "data/img/"
train_labels_dir = "data/label/"
saving_model_dir = "models/"

all_trains = 64
batch_size = 8
dilation = False
version = '2s'
neg_pos_ratio = 3
epoch = 500

learning_rate = 1e-3
weight_decay = 5e-4


# pixel 和 link 预测值设置
pixel_threshold = 0.98
link_threshold = 0.98
short_side_threshold = 5
box_mask_area_threshold = 10
