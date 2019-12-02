# -*- coding: utf-8 -*-
# @Time    : 2019/7/9
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : loss.py
# @Software: PyCharm
import mxnet as mx
from mxnet import nd, gluon
from custom_softmax_cross_entropy_loss import SoftmaxCrossEntropyLoss
import numpy as np
from util import try_gpu
import config



class PixelLinkLoss():
    def __init__(self):
        self.pixel_cross_entropy_layer = SoftmaxCrossEntropyLoss(axis=1, batch_axis=0)
        self.link_cross_entropy_layer = SoftmaxCrossEntropyLoss(axis=1, batch_axis=0)
        self.pixel_cross_entropy = None
        self.pixel_cross_entropy_neg = None
        self.pixel_cross_entropy_pos = None
        self.link_cross_entropy = None
        self.pos_pixel_weight = None
        self.neg_pixel_weight = None
        self.pixel_weight = None
        self.pos_link_weight = None
        self.neg_link_weight = None
        self.area = None
        self.neg_area = None
        self.ctx = try_gpu()



    def pixel_loss(self, input, target, pos_weight, neg_pixel_masks):
        batch_size = input.shape[0]
        self.pos_pixel_weight = pos_weight # (2, 256, 256)
        self.pixel_cross_entropy_sum, self.pixel_cross_entropy = self.pixel_cross_entropy_layer(input, target) # (2,) (2, 1, 256, 256)
        self.area = nd.sum(nd.reshape(target, (batch_size, -1)), axis=1) # (2,)
        self.neg_pixel_weight = nd.zeros_like(self.pos_pixel_weight, dtype=np.uint8)
        self.neg_area = []
        for i in range(batch_size):
            wrong_input = self.pixel_cross_entropy[i][0].asnumpy()[np.where(neg_pixel_masks[i].asnumpy()==1)]
            r_pos_area = (self.area[i] * config.neg_pos_ratio).asscalar() # 转化为标量
            if r_pos_area <= 50: r_pos_area = 10000
            self.neg_area.append(min(r_pos_area, wrong_input.shape[0]))
            topk = nd.topk(nd.array(wrong_input, ctx=self.ctx), ret_typ='value', k=int(self.neg_area[i]))
            pixel_cross_entropy_neg = self.pixel_cross_entropy[i][0] * neg_pixel_masks[i]
            self.neg_pixel_weight[i].asnumpy()[np.where(pixel_cross_entropy_neg.asnumpy() >= topk[-1].asscalar())] = 1
            self.neg_pixel_weight[i] = nd.array(self.neg_pixel_weight[i], dtype='float32', ctx=self.ctx) # np 转化为 nd
        self.pixel_weight = self.pos_pixel_weight + self.neg_pixel_weight
        weighted_pixel_cross_entropy_pos = nd.expand_dims(self.pos_pixel_weight, axis=1) * self.pixel_cross_entropy
        weighted_pixel_cross_entropy_pos = nd.reshape(weighted_pixel_cross_entropy_pos, (batch_size, -1))
        weighted_pixel_cross_entropy_neg = nd.expand_dims(self.neg_pixel_weight, axis=1) * self.pixel_cross_entropy
        weighted_pixel_cross_entropy_neg = nd.reshape(weighted_pixel_cross_entropy_neg, (batch_size, -1)) # (2, 65536)
        self.neg_area = nd.array(self.neg_area, ctx=self.ctx)
        return nd.mean(nd.sum(weighted_pixel_cross_entropy_pos, axis=1) / (self.area + self.neg_area)),\
               nd.mean(nd.sum(weighted_pixel_cross_entropy_neg, axis=1) / (self.area + self.neg_area))

    def link_loss(self, input, target, neighbors=8):
        batch_size = input.shape[0]
        self.pos_link_weight = (target == 1).astype('float32') * nd.broadcast_axes(nd.expand_dims(self.pixel_weight, axis=1), axis=1, size=neighbors) # (2, 8, 256, 256)
        self.neg_link_weight = (target == 0).astype('float32') * nd.broadcast_axes(nd.expand_dims(self.pixel_weight, axis=1), axis=1, size=neighbors)
        sum_pos_link_weight = nd.sum(nd.reshape(self.pos_link_weight,(batch_size, -1)), axis=1)
        sum_neg_link_weight = nd.sum(nd.reshape(self.neg_link_weight,(batch_size, -1)), axis=1)
        self.link_cross_entropy = []
        for i in range(neighbors):
            assert input.shape[1] == 16
            this_input = input[:, [2 * i, 2 * i + 1]]
            this_target = target[:, i]
            self.link_cross_entropy.append(self.link_cross_entropy_layer(this_input, this_target)[1])
        self.link_cross_entropy = nd.concat(*self.link_cross_entropy, dim=1) # (2, 8, 256, 256)
        loss_link_pos = []
        loss_link_neg = []
        ctx = try_gpu()
        for i in range(batch_size):
            if sum_pos_link_weight[i].asscalar() == 0:
                loss_link_pos_temp = nd.zeros(self.pos_link_weight[0].shape, ctx,'float32')
                loss_link_pos.append(nd.expand_dims(loss_link_pos_temp, axis=0))
            else:
                loss_link_pos_temp = self.pos_link_weight[i] * self.link_cross_entropy[i] / sum_pos_link_weight[i]
                loss_link_pos.append(nd.expand_dims(loss_link_pos_temp, axis=0))
            if sum_neg_link_weight[i].asscalar() == 0:
                loss_link_neg_temp = nd.zeros(self.neg_link_weight[0].shape, ctx,'float32')
                loss_link_neg.append(nd.expand_dims(loss_link_neg_temp, axis=0))
            else:
                loss_link_neg_temp = self.neg_link_weight[i] * self.link_cross_entropy[i] / sum_neg_link_weight[i] # (8, 256, 256)
                loss_link_neg.append(nd.expand_dims(loss_link_neg_temp, axis=0))
        loss_link_pos = nd.concat(*loss_link_pos, dim=0)
        loss_link_neg = nd.concat(*loss_link_neg, dim=0) # (2, 8, 256, 256)
        loss_link_pos = nd.sum(nd.reshape(loss_link_pos, (batch_size, -1)), axis=1)
        loss_link_neg = nd.sum(nd.reshape(loss_link_neg, (batch_size, -1)), axis=1)
        return nd.mean(loss_link_pos), nd.mean(loss_link_neg)


