# -*- coding: utf-8 -*-
# @Time    : 2019/7/9
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : train.py
# @Software: PyCharm
import time, os, mxnet as mx
import config
from mxnet import autograd,nd
from loss import PixelLinkLoss


def train(epoch, dataloader, net, optimizer, ctx, iter_num):
    for i in range(epoch):
        for i_batch, sample in enumerate(dataloader):
            start = time.time()
            images = sample[0].as_in_context(ctx) # torch.Size([batchsize, 3, 512, 512])
            pixel_masks = sample[1].as_in_context(ctx)
            neg_pixel_masks = sample[2].as_in_context(ctx)
            pixel_pos_weights = sample[3].as_in_context(ctx)
            link_masks = sample[4].as_in_context(ctx)
            loss_instance = PixelLinkLoss()
            with autograd.record():
                out_1, out_2 = net(images)# (2, 2, 256, 256),(2, 16, 256, 256)
                pixel_loss_pos, pixel_loss_neg = loss_instance.pixel_loss(out_1, pixel_masks, pixel_pos_weights, neg_pixel_masks)
                link_loss_pos, link_loss_neg = loss_instance.link_loss(out_2, link_masks)
                pixel_loss = pixel_loss_pos + pixel_loss_neg
                link_loss = link_loss_pos + link_loss_neg
                losses = config.pixel_weight * pixel_loss + config.link_weight * link_loss
            losses.backward()
            optimizer.step(images.shape[0])
            end = time.time()
            print('losses:', losses.asscalar(), 'pixel_loss', pixel_loss.asscalar(),'link_loss', link_loss.asscalar(), 'time_cost:', round(end-start, 3), 's')
        if (iter_num + i + 1) % 2 == 0:
            params_file = os.path.join(config.saving_model_dir, 'epoch_%s' % (iter_num + i + 1) + '.params')
            net.save_parameters(params_file)
            # test_acc = evaluate_accuracy(test_iter, net, ctx)
            # # val_acc = evaluate_accuracy(val_iter, net, ctx)
            # with open('train_info.txt', 'a') as fw:
            #     fw.write(str(iter_num + epoch + 1) + '\t' + str(round(train_l_sum / n, 4)) + '\t' + str(
            #         round(train_acc_sum / m, 4)) + '\t' + str(round(test_acc, 4)) + '\t' + str(round(time.time() - start, 2)) + '\n')
