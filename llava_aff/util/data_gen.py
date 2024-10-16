import warnings
from transformers import logging as transformers_logging
# Suppress specific warnings
warnings.filterwarnings("ignore")

import os
import json
import re
import copy
import numpy as np
import torch
import random
from glob import glob
from PIL import Image
import transformers
from llava_aff.util.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava_aff import conversation as conversation_lib
from llava_aff.mm_utils import tokenizer_image_token

from typing import Dict, Optional, Sequence, List
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                         )
    try:
        eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                    data_path=data_args.data_path_valid,
                                    data_args=data_args,
                                             )
        # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=None)
    except:
        return dict(train_dataset=train_dataset,
                    eval_dataset=None,
                    data_collator=None)

def divide_action_object(remove_length, main_path, train):
    temp_list = main_path[remove_length:]
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
    return temp_category, temp_item

from torch.utils.data import Dataset
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        
#         self.glob_files = glob(self.data_args.exo_image_folder)
#         exocentric_trainset = [f for f in self.glob_files if "Seen" in f or "Unseen" in f]
#         self.exo_temp_dict = {}

#         for i, temp_image in enumerate(exocentric_trainset):
#             temp_class = temp_image[:-4].split("/")
#             action_class = temp_class[7]
#             object_class = temp_class[8]
#             try:
#                 if temp_dict[object_class] != None:
#                     if action_class not in temp_dict[object_class]:
#                         temp_dict[object_class][action_class] = []
#             except:
#                 self.exo_temp_dict[object_class] = {action_class:[]}

#             self.exo_temp_dict[object_class][action_class].append(temp_image)

        # print("/n/n/n/n/n/n")
        # print(data_args)
        # print("/n/n/n/n/n/n")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        # print("""source\n\n""", sources)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            processor = self.data_args.image_processor
            if self.data_args.stage != "pretrain":
                depth_image_file = self.list_data_dict[i]['depth']
                gt_image_file = self.list_data_dict[i]['gt_path']
                depth_image_file = self.list_data_dict[i]['depth']
                new_depth_image_file = "/".join(gt_image_file.split("/")[:2]) + "/depth/depth_large/" + "/".join(gt_image_file.split("/")[3:])
            
                from torchvision import transforms
                gt_image = Image.open(os.path.join(image_folder, gt_image_file))
                target_size = (768,768)
                resize_transform = transforms.Resize(target_size)
                gt_transform = transforms.ToTensor()
                
                depth_image_rgb = Image.open(os.path.join(image_folder, new_depth_image_file))
                depth_image = depth_image_rgb

                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                    
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    if self.data_args.stage != "pretrain":
                        depth_image = expand2square(depth_image, tuple(int(x*255) for x in processor.image_mean))
                        gt_image = expand2square(gt_image, 0)
                        depth_image = processor.preprocess(depth_image, return_tensors="pt")['pixel_values'][0]
                        gt_image = gt_transform(gt_image)
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    depth_image = processor.preprocess(depth_image, return_tensors="pt")['pixel_values'][0]
                    gt_image = gt_transform(gt_image)
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                if self.data_args.image_aspect_ratio == 'pad':
                    depth_image = processor.preprocess(depth_image, return_tensors="pt")['pixel_values'][0]
                    gt_image = gt_transform(gt_image)
                    gt_image = gt_image
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            # print("3")
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            if self.data_args.stage != "pretrain":
                data_dict['depth'] = depth_image
                data_dict['gt_image'] = gt_image
                data_dict["gt_image_path"] = "_".join(gt_image_file.split("/"))
                # remove_length = len("_".join("_".join(gt_image_file.split("/")).split("_")[:3])) + 1
                # temp_action, temp_object = divide_action_object(remove_length, "_".join(gt_image_file.split("/")))
                # print(2)
                # print(temp_action, temp_object)
                # data_dict["random_exo"] = random.sample(self.exo_temp_dict[temp_object][temp_action], 3)
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            if self.data_args.stage != "pretrain":
                data_dict['depth'] = torch.zeros(3, crop_size['height'], crop_size['width'])
                data_dict['gt_image'] = torch.zeros(1, crop_size['height'], crop_size['width'])
                data_dict["gt_image_path"] = "_".join(gt_image_file.split("/"))
                # remove_length = len("_".join("_".join(gt_image_file.split("/")).split("_")[:3])) + 1
                # temp_action, temp_object = divide_action_object(remove_length, "_".join(gt_image_file.split("/")))
                # print(1)
                # print(temp_action, temp_object)
                # data_dict["random_exo"] = random.sample(self.exo_temp_dict[temp_object][temp_action], 3)
        return data_dict

from dataclasses import dataclass, field

def collate_fn(
    instances, tokenizer=None, conv_version="llava_v1", local_rank=-1
):
    input_ids, labels = tuple([instance[key] for instance in instances]
                             for key in ("input_ids", "labels"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                             batch_first=True,
                                             padding_value=IGNORE_INDEX)
    input_ids = input_ids[:, :tokenizer.model_max_length]
    labels = labels[:, :tokenizer.model_max_length]
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

    if 'image' in instances[0]:
        # print(instances)
        images = [instance['image'] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images)
        else:
            batch['images'] = images
            
    if 'depth' in instances[0]:
        depth_images = [instance['depth'] for instance in instances]
        if all(x is not None and x.shape == depth_images[0].shape for x in depth_images):
            batch['depth_images'] = torch.stack(depth_images)
        else:
            batch['depth_images'] = depth_images
            
    if 'gt_image' in instances[0]:
        gt_images = [instance['gt_image'] for instance in instances]
        if all(x is not None and x.shape == gt_images[0].shape for x in gt_images):
            batch['gt_images'] = torch.stack(gt_images)
        else:
            batch['gt_images'] = gt_images
            
    if "gt_image_path" in instances[0]:
        gt_images_path = [instance['gt_image_path'] for instance in instances]
        batch["gt_images_path"] = gt_images_path
        
    # if "random_exo" in instances[0]:
    #     random_exo_path = [instance["random_exo"] for instance in instances]
    #     batch["random_exo_path"] = random_exo_path

    return batch

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    # print("conversation_lib.default_conversation.sep_style", conversation_lib.default_conversation.sep_style)
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def preprocess_multimodal(
    sources: Sequence[str],
    data_args
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources

def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "agd20k": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )