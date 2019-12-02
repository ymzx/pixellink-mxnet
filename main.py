# -*- coding: utf-8 -*-
# @Time    : 2019/7/5
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : main.py.py
# @Software: PyCharm
from mxnet.gluon import data as gdata
from net import Net
import datasets
import config
import sys
from train import train
import mxnet as mx
from util import try_gpu, load_latest_params

def main():
    dataset = datasets.PixelLinkDataset(config.train_images_dir, config.train_labels_dir)
    # Gluon的DataLoader允许使用多进程来加速数据读取（暂不支持Windows操作系统）
    num_workers = 0 if sys.platform.startswith('win32') else 4
    dataloader = gdata.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)
    ctx = try_gpu()
    print('训练平台：%s'%(ctx))
    my_net = Net()
    # 加载已保存的模型参数
    para_flag, para_latest, iter_num = load_latest_params(config.saving_model_dir)
    if para_flag:
        my_net.initialize(mx.init.Xavier(), ctx=ctx)
        my_net.load_parameters(para_latest, allow_missing=True, ctx=ctx)
        print('加载已有模型%s' % (para_latest))
    else:
        my_net.initialize(mx.init.Xavier(), ctx=ctx)# 'adam'
    optimizer = mx.gluon.Trainer(my_net.collect_params(), 'sgd', {'learning_rate': config.learning_rate, 'wd': config.weight_decay})
    train(config.epoch, dataloader, my_net, optimizer, ctx, iter_num)

if __name__ == "__main__":
    main()