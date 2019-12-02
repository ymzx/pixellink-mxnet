# -*- coding: utf-8 -*-
# @Time    : 2019/11/7
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : predict.py
# @Software: PyCharm
import config
from postprocess import mask_filter, mask_to_box
from mxnet import nd, image
import cv2,numpy as np
import mxnet as mx
from net import Net
from ImgLib.ImgTransform import ReadImage, ZeroMeanImage, normalize_image


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    # try:
    #     ctx = mx.gpu()
    #     _ = nd.array([0], ctx=ctx)
    # except mx.base.MXNetError:
    ctx = mx.cpu()
    return ctx

def result_draw(img, boxes, color=(0, 0, 255)):

    """
    img: HWC
    boxes: list, len(boxes)=batch_size
    """
    for box in boxes:
        img = cv2.drawContours(img, [box.astype(int)], -1, color, thickness=1)
    return img

def predict_main(img_path, model_params):
    img = ReadImage(img_path) # HWC,numpy
    H, W, C = img.shape
    # zero mean
    feature = ZeroMeanImage(img, config.r_mean, config.g_mean, config.b_mean)
    # feature = normalize_image(nd.array(img))
    feature = nd.array(feature) # numpy转nd
    # HWC to CHW
    X = feature.transpose((2, 0, 1)).expand_dims(axis=0)  # 3D转为4D
    net = Net()
    ctx = try_gpu()
    net.initialize(mx.init.Xavier(), ctx=ctx)
    net.load_parameters(model_params,ctx=ctx) # load
    pixel_mask, link_mask = net(X.as_in_context(ctx))
    pixel_mask, link_mask = mask_filter(pixel_mask, link_mask) # pixel_mask (1, 256, 256)
    all_boxes = mask_to_box(pixel_mask, link_mask)
    ratio = 2 if config.version == '2s'else 4
    # 默认对第一张进行画图
    one_boxes = []
    for box in all_boxes[0]:
        one_boxes.append(box*ratio)
    label_img = result_draw(img, one_boxes)
    # pred = pixel_mask.transpose((1, 2, 0))
    # resize_pred = image.imresize(pred, W, H) # default=cv2.INTER_LINEAR
    # result = nd.reshape(resize_pred, (H, W))*255
    return label_img


if __name__ == '__main__':
    # img_path = r'D:\ocr_data_set\data\img\img_7.jpg'
    # model_params = r'models/epoch_52.params'
    # out = predict_main(img_path, model_params)
    # cv2.imshow('1',out) # asnumpy 耗时30s
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 批量预测
    import os,time
    model_params = r'models/epoch_28.params'
    imgs_path = r'D:\ocr_data_set\data\img'
    img_dirs = os.listdir(imgs_path)
    save_path = r'D:\object_detection_on_gitserver\object_detection\extract_handwriting_text\pixellink-with-mxnet\v2\test_result'
    for img_dir in img_dirs:
        t1 = time.time()
        image_path = os.path.join(imgs_path, img_dir)
        out = predict_main(image_path, model_params)
        t2 = time.time()
        print('耗时：', round(t2-t1,3), 's')
        img_save_path = os.path.join(save_path, img_dir)
        cv2.imwrite(img_save_path, out)
