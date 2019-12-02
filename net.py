# -*- coding: utf-8 -*-
# @Time    : 2019/7/8
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : net.py
# @Software: PyCharm
import config
import mxnet as mx
from mxnet.gluon import nn

def sum_matrix(mat1, mat2):
    '''
    size在放大和缩小过程中由于奇和偶的原因，通过ceil和floor导致维度不一致。
    :param mat1:
    :param mat2:
    :return:
    '''
    num1, channel1, height1, width1 = mat1.shape
    num2, channel2, height2, width2 = mat2.shape
    height = max(height1, height2)
    width = max(width1, width2)
    if height1 != height:
        mat1 = mx.ndarray.pad(mat1, mode="constant", constant_value=0, pad_width=(0, 0, 0, 0, 0, 1, 0, 0))
    if height2 != height:
        mat2 = mx.ndarray.pad(mat2, mode="constant", constant_value=0, pad_width=(0, 0, 0, 0, 0, 1, 0, 0))
    if width1 != width:
        mat1 = mx.ndarray.pad(mat1, mode="constant", constant_value=0, pad_width=(0, 0, 0, 0, 0, 0, 0, 1))
    if width2 != width:
        mat2 = mx.ndarray.pad(mat2, mode="constant", constant_value=0, pad_width=(0, 0, 0, 0, 0, 0, 0, 1))
    mat = mat1 + mat2
    return mat

def custom_conv2d(channels, kernel_size, strides, padding, activation, dilation=None):
    net = nn.Sequential()
    if not dilation:
        net.add(nn.Conv2D(channels, kernel_size=kernel_size, strides=strides, padding=padding),nn.BatchNorm(), nn.Activation(activation))
    else:
        net.add(nn.Conv2D(channels, kernel_size=kernel_size, strides=strides, padding=padding, dilation=dilation), nn.BatchNorm(),
                nn.Activation(activation))
    return net




class Net(nn.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            # self.net_part1 = nn.Sequential() 部署时候用该方法封装

            # part1
            self.conv1_1 = custom_conv2d(channels=64, kernel_size=3, activation='relu', strides=1, padding=1)
            self.conv1_2 = custom_conv2d(channels=64, kernel_size=3, activation='relu', strides=1, padding=1)
            self.pool1 = nn.MaxPool2D(pool_size=(2, 2))

            # part2
            self.conv2_1 = custom_conv2d(channels=128, kernel_size=3, activation='relu', strides=1, padding=1)
            self.conv2_2 = custom_conv2d(channels=128, kernel_size=3, activation='relu', strides=1, padding=1)
            self.pool2 = nn.MaxPool2D(pool_size=(2,2))

            # part3
            self.conv3_1 = custom_conv2d(channels=256, kernel_size=3, activation='relu', strides=1, padding=1)
            self.conv3_2 = custom_conv2d(channels=256, kernel_size=3, activation='relu', strides=1, padding=1)
            self.conv3_3 = custom_conv2d(channels=256, kernel_size=3, activation='relu', strides=1, padding=1)
            self.pool3 = nn.MaxPool2D(pool_size=(2, 2))

            # part4
            self.conv4_1 = custom_conv2d(channels=512, kernel_size=3, activation='relu', strides=1, padding=1)
            self.conv4_2 = custom_conv2d(channels=512, kernel_size=3, activation='relu', strides=1, padding=1)
            self.conv4_3 = custom_conv2d(channels=512, kernel_size=3, activation='relu', strides=1, padding=1)
            self.pool4 = nn.MaxPool2D(pool_size=(2, 2))

            # part5
            self.conv5_1 = custom_conv2d(channels=512, kernel_size=3, activation='relu', strides=1, padding=1)
            self.conv5_2 = custom_conv2d(channels=512, kernel_size=3, activation='relu', strides=1, padding=1)
            self.conv5_3 = custom_conv2d(channels=512, kernel_size=3, activation='relu', strides=1, padding=1)
            self.pool5 = nn.MaxPool2D(pool_size=(3, 3), strides=1, padding=1)

            # part6
            if config.dilation:
                self.conv6 = custom_conv2d(channels=1024, kernel_size=3, strides=1, padding=6, activation='relu', dilation=6)
            else:
                self.conv6 = custom_conv2d(channels=1024, kernel_size=3, strides=1, padding=1, activation='relu')

            # part7
            self.conv7 = custom_conv2d(channels=1024, kernel_size=1, strides=1, padding=0, activation='relu')

            # output
            # out1
            self.out1_1 = nn.Conv2D(in_channels=128, channels=2,kernel_size=1,strides=1, padding=0)
            self.out1_2 = nn.Conv2D(in_channels=128, channels=16, kernel_size=1, strides=1, padding=0)

            # out2
            self.out2_1 = nn.Conv2D(in_channels=256, channels=2,kernel_size=1,strides=1, padding=0)
            self.out2_2 = nn.Conv2D(in_channels=256, channels=16, kernel_size=1, strides=1, padding=0)

            # out3
            self.out3_1 = nn.Conv2D(in_channels=512, channels=2,kernel_size=1,strides=1, padding=0)
            self.out3_2 = nn.Conv2D(in_channels=512, channels=16, kernel_size=1, strides=1, padding=0)

            # out4
            self.out4_1 = nn.Conv2D(in_channels=512, channels=2,kernel_size=1,strides=1, padding=0)
            self.out4_2 = nn.Conv2D(in_channels=512, channels=16, kernel_size=1, strides=1, padding=0)

            # out5
            self.out5_1 = nn.Conv2D(in_channels=1024, channels=2, kernel_size=1, strides=1, padding=0)
            self.out5_2 = nn.Conv2D(in_channels=1024, channels=16, kernel_size=1, strides=1, padding=0)

            # upsample,out = (in - 1) × strides - 2 × padding + kernel_size
            self.upsample1 = nn.Conv2DTranspose(in_channels=2, channels=2, kernel_size=4,strides=2, padding=1,
                                           weight_initializer=mx.init.Bilinear(), use_bias=False)
            self.upsample2 = nn.Conv2DTranspose(in_channels=16, channels=16, kernel_size=4, strides=2, padding=1,
                                           weight_initializer=mx.init.Bilinear(), use_bias=False)


    def forward(self, x):
        x = self.pool1(self.conv1_2(self.conv1_1(x)))
        x = self.conv2_2(self.conv2_1(x))
        l1_1x = self.out1_1(x)
        l1_2x = self.out1_2(x)

        x = self.conv3_3(self.conv3_2(self.conv3_1(self.pool2(x))))
        l2_1x = self.out2_1(x)
        l2_2x = self.out2_2(x)

        x = self.conv4_3(self.conv4_2(self.conv4_1(self.pool3(x))))
        l3_1x = self.out3_1(x)
        l3_2x = self.out3_2(x)

        x = self.conv5_3(self.conv5_2(self.conv5_1(self.pool4(x))))
        l4_1x = self.out4_1(x)
        l4_2x = self.out4_2(x)

        x = self.conv7(self.conv6(self.pool5(x)))
        l5_1x = self.out5_1(x)
        l5_2x = self.out5_2(x)

        # upsample
        up1_1 = self.upsample1(sum_matrix(l5_1x, l4_1x))
        up2_1 = self.upsample1(sum_matrix(up1_1, l3_1x))
        if config.version == "2s":
            up3_1 = self.upsample1(sum_matrix(up2_1, l2_1x))
            out_1 = sum_matrix(up3_1, l1_1x)
        else:
            out_1 = sum_matrix(up2_1, l2_1x)

        up1_2 = self.upsample2(sum_matrix(l5_2x, l4_2x))
        up2_2 = self.upsample2(sum_matrix(up1_2, l3_2x))
        if config.version == "2s":
            up3_2 = self.upsample2(sum_matrix(up2_2, l2_2x))
            out_2 = sum_matrix(up3_2, l1_2x)
        else:
            out_2 = sum_matrix(up2_2, l2_2x)
        return [out_1, out_2]





