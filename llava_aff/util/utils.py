from enum import Enum

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn


CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output segmentation mask.",
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict

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

import torch.nn.functional as F
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()  # alpha 값을 CUDA 텐서로 변환
        self.alpha_value = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, num_boxes):
        prob = inputs
        BCE_loss = F.binary_cross_entropy_with_logits(prob, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = BCE_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha_value >= 0:
            alpha_t = self.alpha_value * targets + (1 - self.alpha_value) * (1 - targets)
            loss = alpha_t * loss
            # print(loss)
            # print(loss.shape)
        # targets = targets.type(torch.long).view(-1)
        # at = self.alpha.gather(0, targets)
        # pt = torch.exp(-BCE_loss.view(-1))
        # F_loss = at * (1 - pt) ** self.gamma * BCE_loss.view(-1)
        return loss.mean(1).sum() / num_boxes
    
criterion = WeightedFocalLoss()