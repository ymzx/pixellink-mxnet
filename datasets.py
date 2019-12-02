# -*- coding: utf-8 -*-
# @Time    : 2019/7/5
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : datasets.py
# @Software: PyCharm
import ImgLib.ImgTransform as ImgTransform
from mxnet.gluon.data import Dataset
from PIL import Image
import config
import codecs
import random
import cv2
import os
import numpy as np
from mxnet import nd


class ICDAR15Dataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.all_labels = self.read_labels(labels_dir, config.all_trains)

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, index):  # 让对象实现迭代功能
        if isinstance(index, int):
            return {'image': self.read_image(self.images_dir, index), 'label': self.all_labels[index]}

    def read_image(self, dir, index):
        index += 1
        filename = os.path.join(dir, "img_" + str(index) + ".jpg")
        image = ImgTransform.ReadImage(filename)
        return image

    def read_datasets(self, dir, num):
        res = []
        for i in range(1, num+1):
            image = Image.open(dir+ "img_" + str(i) + ".jpg")
            res.append(image)
            if i % 100 == 0:
                print(i)
        return res

    def read_labels(self, dir, num):
        res = []
        for i in range(1, num+1):
            with codecs.open(dir + "gt_img_" + str(i) + ".txt", encoding="utf-8_sig") as file:
                data = file.readlines()
                tmp = dict()
                tmp["coor"], tmp["content"], tmp["ignore"], tmp["area"] = [], [], [], []
                for line in data:
                    content = line.split(",")
                    coor = [int(n) for n in content[:8]]
                    tmp["coor"].append(coor)
                    content[8] = content[8].strip("\r\n")
                    tmp["content"].append(content[8])
                    if content[8] == "###":
                        tmp["ignore"].append(True)
                    else:
                        tmp["ignore"].append(False)
                    coor = np.array(coor).reshape([4,2])
                    tmp["area"].append(cv2.contourArea(coor))
                res.append(tmp)
        return res



class PixelLinkDataset(ICDAR15Dataset):
    def __init__(self, images_dir, labels_dir, train=True):
        super(PixelLinkDataset, self).__init__(images_dir, labels_dir)
        self.train = train

    def __getitem__(self, index):
        if self.train:
            image, label = self.train_data_transform(index)
        else:
            image, label = self.test_data_transform(index)
        image = nd.array(image)
        pixel_mask, neg_pixel_mask, pixel_pos_weight, link_mask = \
            PixelLinkDataset.label_to_mask_and_pixel_pos_weight(label, list(image.shape[1:]), version=config.version)
        return image, pixel_mask, neg_pixel_mask, pixel_pos_weight, link_mask

    def test_data_transform(self, index):
        img = self.read_image(self.images_dir, index)
        labels = self.all_labels[index]
        labels, img, size = ImgTransform.ResizeImageWithLabel(labels, (512, 512), data=img)
        # img = ImgTransform.normalize_image(img)
        img = ImgTransform.ZeroMeanImage(img, config.r_mean, config.g_mean, config.b_mean)
        img = img.transpose(2, 0, 1)
        return img, labels

    def train_data_transform(self, index):
        img = self.read_image(self.images_dir, int(index))
        labels = self.all_labels[int(index)]

        rotate_rand = random.random() if config.use_rotate else 0
        crop_rand = random.random() if config.use_crop else 0
        # rotate
        if rotate_rand > 0.5:
            labels, img, angle = ImgTransform.RotateImageWithLabel(labels, data=img)
        # crop
        if crop_rand > 0.5:
            scale = 0.1 + random.random() * 0.9
            labels, img, img_range = ImgTransform.CropImageWithLabel(labels, data=img, scale=scale)
            labels = PixelLinkDataset.filter_labels(labels, method="rai")
        # resize
        labels, img, size = ImgTransform.ResizeImageWithLabel(labels, (512, 512), data=img)
        # zero mean
        img = ImgTransform.ZeroMeanImage(img, config.r_mean, config.g_mean, config.b_mean)
        # img = ImgTransform.normalize_image(nd.array(img))
        # HWC to CHW
        img = img.transpose((2, 0, 1))
        return img, labels

    @staticmethod
    def filter_labels(labels, method):
        """
        method: "msi" for min area ignore, "rai" for remain area ignore
        """
        def distance(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

        def min_side_ignore(label):
            label = np.array(label).reshape(4, 2)
            dists = []
            for i in range(4):
                dists.append(distance(label[i], label[(i+1)%4]))
            if min(dists) < 10:
                return True # ignore it
            else:
                return False

        def remain_area_ignore(label, origin_area):
            label = np.array(label).reshape(4, 2)
            area = cv2.contourArea(label)
            if area / origin_area < 0.2:
                return True
            else:
                return False
        if method == "msi":
            ignore = list(map(min_side_ignore, labels["coor"]))
        elif method == "rai":
            ignore = list(map(remain_area_ignore, labels["coor"], labels["area"]))
        else:
            ignore = [False] * 8
        labels["ignore"] = list(map(lambda a, b: a or b, labels["ignore"], ignore))
        return labels

    @staticmethod
    def label_to_mask_and_pixel_pos_weight(label, img_size, version="2s", neighbors=8):
        """
        8 neighbors:
            0 1 2
            7 - 3
            6 5 4
        """
        def is_valid_coor(h_index, w_index, h, w):
            if h_index < 0 or w_index < 0:
                return False
            elif h_index >= h or w_index >= w:
                return False
            else:
                return True

        def get_neighbors(h_index, w_index):
            res = []
            res.append([h_index - 1, w_index - 1])
            res.append([h_index - 1, w_index])
            res.append([h_index - 1, w_index + 1])
            res.append([h_index, w_index + 1])
            res.append([h_index + 1, w_index + 1])
            res.append([h_index + 1, w_index])
            res.append([h_index + 1, w_index - 1])
            res.append([h_index, w_index - 1])
            return res

        factor = 2 if version == "2s" else 4
        label_coor = np.array(label["coor"]).reshape([-1, 1, 4, 2])
        pixel_mask_size = [int(i / factor) for i in img_size]
        img_size = [int(i / factor) for i in img_size]# 更新img_size
        link_mask_size = [neighbors, ] + pixel_mask_size
        pixel_mask = np.zeros(pixel_mask_size, dtype=np.uint8)
        pixel_weight = np.zeros(pixel_mask_size, dtype=np.float)
        link_mask = np.zeros(link_mask_size, dtype=np.uint8)
        label_coor = (label_coor / factor).astype(int)
        bbox_masks = []
        num_positive_bboxes = 0
        for i, coor in enumerate(label_coor):
            pixel_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
            cv2.drawContours(pixel_mask_tmp, coor, -1, 1, thickness=-1)
            bbox_masks.append(pixel_mask_tmp)
            if not label["ignore"][i]:
                pixel_mask += pixel_mask_tmp
                num_positive_bboxes += 1
        pos_pixel_mask = (pixel_mask == 1).astype(np.int)
        num_pos_pixels = np.sum(pos_pixel_mask)
        sum_mask = np.sum(bbox_masks, axis=0)
        neg_pixel_mask = (sum_mask != 1).astype(np.int)
        for bbox_index, bbox_mask in enumerate(bbox_masks):
            bbox_positive_pixel_mask = bbox_mask * pos_pixel_mask
            num_pos_pixel = np.sum(bbox_positive_pixel_mask)
            if num_pos_pixel > 0:
                per_bbox_weight = num_pos_pixels * 1.0 / num_positive_bboxes
                per_pixel_weight = per_bbox_weight / num_pos_pixel
                pixel_weight += bbox_positive_pixel_mask * per_pixel_weight
            for link_index in range(neighbors):
                link_mask[link_index][np.where(bbox_positive_pixel_mask)] = 1
            bbox_border_mask = np.ones(pixel_mask_size, dtype=np.int)
            bbox_border_mask *= bbox_positive_pixel_mask
            bbox_border_cords = np.where(bbox_border_mask)
            border_points = list(zip(*bbox_border_cords))
            def in_bbox(nx, ny):
                return bbox_positive_pixel_mask[nx, ny]
            for h_index, w_index in border_points:
                neighbors_pixels = get_neighbors(h_index, w_index)
                for nei_index, [nei_h_index, nei_w_index] in enumerate(neighbors_pixels):
                    if not is_valid_coor(h_index, w_index, *img_size) or not is_valid_coor(nei_h_index, nei_w_index, *img_size):
                        link_mask[nei_index, h_index, w_index] = 0
                        continue
                    if not in_bbox(nei_h_index, nei_w_index):
                        link_mask[nei_index, h_index, w_index] = 0
        return nd.array(pixel_mask), nd.array(neg_pixel_mask), nd.array(pixel_weight), nd.array(link_mask)



