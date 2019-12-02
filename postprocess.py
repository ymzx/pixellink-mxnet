# -*- coding: utf-8 -*-
# @Time    : 2019/11/12
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : postprocess.py
# @Software: PyCharm
import numpy as np, cv2, time
from mxnet import nd
from config import pixel_threshold, link_threshold, short_side_threshold, box_mask_area_threshold


def mask_filter(pixel_mask, link_mask, neighbors=8, scale=4):

    """
    pixel_mask: batch_size * 2 * H * W
    link_mask: batch_size * 16 * H * W
    """
    link_shape = link_mask.shape
    batch_size, mask_height, mask_width = link_shape[0],link_shape[2],link_shape[3]
    pixel_class = nd.softmax(pixel_mask, axis=1)
    pixel_class = (pixel_class[:, 1] > pixel_threshold).astype(np.uint8)
    link_neighbors = nd.zeros((batch_size, neighbors, mask_height, mask_width), dtype=np.uint8)
    for i in range(neighbors):
        tmp = nd.softmax(link_mask[:, [2 * i, 2 * i + 1]],axis=1)
        link_neighbors[:, i] = (tmp[:, 1] > link_threshold).astype(np.uint8)
        link_neighbors[:, i] = nd.logical_and(link_neighbors[:, i], pixel_class)
    return pixel_class, link_neighbors


def is_valid_cord(x, y, w, h):
    return x >= 0 and x < w and y >= 0 and y < h

def get_neighbors(h_index, w_index):
    res = []
    res.append((h_index - 1, w_index - 1))
    res.append((h_index - 1, w_index))
    res.append((h_index - 1, w_index + 1))
    res.append((h_index, w_index + 1))
    res.append((h_index + 1, w_index + 1))
    res.append((h_index + 1, w_index))
    res.append((h_index + 1, w_index - 1))
    res.append((h_index, w_index - 1))
    return res


def mask_to_box(pixel_mask, link_mask):
    """
    pixel_mask: batch_size * H * W
    link_mask: batch_size * 8 * H * W
    """
    def distance(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    def short_side_filter(bounding_box):
        for i, point in enumerate(bounding_box):
            if distance(point, bounding_box[(i + 1) % 4]) < short_side_threshold:
                return True
        return False

    batch_size = pixel_mask.shape[0]
    all_boxes = []
    for i in range(batch_size):
        res_mask = func(pixel_mask[i], link_mask[i])
        box_num = np.max(res_mask)
        print('box_num', box_num)
        bounding_boxes = []
        print('res_mask',res_mask.shape, 'pixel_mask', pixel_mask.shape)
        for i in range(1, box_num + 1):
            box_mask = (res_mask == i).astype(np.uint8)
            if box_mask.sum() < box_mask_area_threshold:
                continue
            box_mask, contours, _ = cv2.findContours(box_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            if len(contours) == 0: continue
            bounding_box = cv2.minAreaRect(contours[0])
            bounding_box = cv2.boxPoints(bounding_box)
            if short_side_filter(bounding_box): continue
            bounding_boxes.append(bounding_box)
        all_boxes.append(bounding_boxes)
    return all_boxes


def func(pixel_cls, link_cls):

    def joint(pointa, pointb):
        roota = find_root(pointa)
        rootb = find_root(pointb)
        if roota != rootb:
            group_mask[rootb] = roota

    def find_root(pointa):
        root = pointa
        while group_mask.get(root) != -1:
            root = group_mask.get(root)
        return root

    pixel_cls = pixel_cls.asnumpy()
    link_cls = link_cls.asnumpy()
    pixel_points = list(zip(*np.where(pixel_cls)))
    h, w = pixel_cls.shape
    group_mask = dict.fromkeys(pixel_points, -1)
    for point in pixel_points:
        h_index, w_index = point[0], point[1]
        neighbors = get_neighbors(h_index, w_index)
        for i, neighbor in enumerate(neighbors):
            nh_index, nw_index = neighbor
            if not is_valid_cord(nh_index, nw_index, w, h): continue
            if pixel_cls[nh_index, nw_index] == 1 and link_cls[i, h_index, w_index] == 1:
                joint(point, neighbor)
    mask = np.zeros_like(pixel_cls, dtype=np.int32)
    root_map = {}
    for point in pixel_points:
        h_index, w_index = point[0], point[1]
        root = find_root(point)
        if root not in root_map:
            root_map[root] = len(root_map) + 1
        mask[h_index, w_index] = root_map[root]
    return mask
