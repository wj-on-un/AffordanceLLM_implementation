import math
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

def extract_item_category(temp_list, train=True):
    if train == True:
        hard_split_train = ['apple', 'axe', 'badminton racket', 'banana', 'baseball bat', 'baseball', 'basketball', 'bed', 'bench', 'bicycle', 'binoculars', 'book', 'bottle', 'bowl', 'broccoli', 'camera', 'carrot', 'cell phone', 'chair', 'couch', 'cup', 'discus', 'drum', 'fork', 'frisbee', 'golf clubs', 'hammer', 'hot dog']
    else:
        hard_split_train = ['javelin', 'keyboard', 'knife', 'laptop', 'microwave', 'motorcycle', 'orange', 'oven', 'punching bag', 'refrigerator', 'rugby ball', 'scissors', 'skateboard', 'skis', 'snowboard', 'soccer ball', 'suitcase', 'surfboard', 'tennis racket', 'toothbrush', 'wine glass', 'pen']
    for i in range(len(hard_split_train)):
        check_item = hard_split_train[i].split(" ")
        if len(check_item) >= 2:
            temp_item = "_".join(check_item)
        else:
            temp_item = "_".join(check_item)
        if temp_list.find(temp_item) != -1:
            temp_list = temp_list.replace(temp_item, "")
            break
    # print(check_item)
    split_temp_list = temp_list.split("_")
    temp_category = ""
    for i in range(len(split_temp_list)):
        if not split_temp_list[i] == "":
            if i == 0:
                temp_category += split_temp_list[i]
            else:
                temp_category += "_" + split_temp_list[i] 
        else:
            break
    return temp_item, temp_category

def expand2square(pil_img, background_color):
    # print("pil_imag.size", pil_img.size)
    width, height = pil_img.size
    if width == height:
        # print("0")
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        # print("1")
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        # print("2")
        return result
    
def trim_background(pil_img, background_color):
    # 이미지의 데이터를 얻음
    img_data = pil_img.getdata()

    # 배경색과 일치하지 않는 픽셀의 좌표를 추적
    non_background_pixels = [
        (x, y) for y in range(pil_img.height) for x in range(pil_img.width)
        if img_data[y * pil_img.width + x] != background_color
    ]

    # 이미지의 바운딩 박스를 찾음
    if not non_background_pixels:
        return pil_img, (0, 0, 0, 0)  # 배경색 외에 아무 것도 없을 경우 원본 리턴

    # 비배경 픽셀의 최소/최대 좌표를 통해 바운딩 박스를 생성
    min_x = min([x for x, y in non_background_pixels])
    max_x = max([x for x, y in non_background_pixels])
    min_y = min([y for x, y in non_background_pixels])
    max_y = max([y for x, y in non_background_pixels])

    # 바운딩 박스를 기준으로 이미지를 자름
    cropped_img = pil_img.crop((min_x, min_y, max_x + 1, max_y + 1))
    return cropped_img, (min_x, min_y, max_x + 1, max_y + 1)

import numpy as np
import torch

def cal_kl(pred: np.ndarray, gt: np.ndarray ,eps=1e-12) -> np.ndarray:
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    kld =np.sum(map2 *np.log(map2 /(map1 +eps) +eps))
    return kld

def cal_sim(pred: np.ndarray, gt: np.ndarray,eps=1e-12) -> np.ndarray:
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    intersection = np.minimum(map1, map2)


    return np.sum(intersection)

def image_binary(image, threshold):
    output = np.zeros(image.size).reshape(image.shape)
    for xx in range(image.shape[0]):
        for yy in range(image.shape[1]):
            if (image[xx][yy] > threshold):
                output[xx][yy] = 1
    return output

def cal_nss(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = pred / 255.0
    gt = gt / 255.0
    std = np.std(pred)
    u = np.mean(pred)

    smap = (pred - u) / std
    fixation_map = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-12)
    fixation_map = image_binary(fixation_map, 0.1)

    nss = smap * fixation_map

    nss = np.sum(nss) / np.sum(fixation_map+1e-12)

    return nss