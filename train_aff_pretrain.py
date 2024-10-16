import warnings
from transformers import logging as transformers_logging
# Suppress specific warnings
warnings.filterwarnings("ignore")

import argparse
import os
import shutil
import sys
import time
from functools import partial
import json
import re
import copy
import deepspeed
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from llava_aff.util.utils import (IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU, extract_item_category, cal_kl, cal_sim, cal_nss, WeightedFocalLoss)
from llava_aff.util.data_gen import preprocess_v1, preprocess_plain, preprocess_multimodal, preprocess, collate_fn, LazySupervisedDataset, make_supervised_data_module
import cv2
import torch.nn.functional as F
from llava_aff.model.AFF import AffordanceForCasualLM
from llava_aff import conversation as conversation_lib
from llava_aff.mm_utils import tokenizer_image_token

from llava_aff.model.mask_decoder.mask_decoder import Neck, MaskDecoder
from llava_aff.model.mask_decoder.prompt_encoder import PromptEncoder
from llava_aff.model.mask_decoder.twowaytrans import TwoWayTransformer
from llava_aff.model.mask_decoder.common import LayerNorm2d

def TrainingArguments(args):
    parser = argparse.ArgumentParser(description="Affordance Model Training Arguments")
    # Model Arguments
    parser.add_argument("--vision_tower", default="google/owlvit-base-patch32", type=str)
    parser.add_argument("--version", default="lmsys/vicuna-7b-v1.5", type=str)
    parser.add_argument("--conv_version", default="v1", type=str)
    parser.add_argument("--train_mask_decoder", default=True)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--mm_hidden_size", default=768, type=int)
    parser.add_argument("--mm_patch_merge_type", default='flat', type=str)
    parser.add_argument("--image_size", default=768, type=int, help="image size")
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--mm_use_im_start_end", default=False, type=bool)
    
    # Training Arguments
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"],
        help="precision for inference", )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--data_path", default="None", type=str, 
                        help="data_train_revise_path.json path ex) /home/ubuntu/LLava_pre/blip_laion_cc_sbu_558k.json")
    parser.add_argument("--data_path_valid", default=None, type=str)
    parser.add_argument("--image_folder", default=None, type=str,
                        help="LLava pretrain dataset path, ex) /home/ubuntu/LLava_pre")
    parser.add_argument("--log_base_dir", default=None, type=str,
                        help="Main save folder path ex) /home/ubuntu/aff_saves")
    parser.add_argument("--exp_name", default="aff_try_1", type=str,
                        help="tensorboard and weight save training name ex) aff_pretrain_try_1 -> /home/ubuntu/aff_saves/aff_pretrain_try_1")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=32, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=1, type=int, ) # default=10
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)

    parser.add_argument("--lora_enable", default=False, type=bool)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)

    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)

    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=True)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_llama_2", type=str, choices=["llava_v1", "llava_llama_2"], )

    ## Data Arguments
    parser.add_argument("--stage", default="finetune", type=str)
    parser.add_argument("--dataset_time", default="train", type=str)
    parser.add_argument("--image_processor", default=None)
    parser.add_argument("--image_aspect_ratio", default="square", type=str)
    parser.add_argument("--mm_projector_type", default="mlp2x_gelu", type=str)
    
    parser.add_argument("--load_mm_projector_file_path", default="./", type=str)
    parser.add_argument("--load_mm_projector", default=False)
    
    parser.add_argument("--save_output_dir", default="output_dir", type=str)
    parser.add_argument("--transconv_h", default=256, type=int)
    
    return parser.parse_args(args)

def main(args):
    args = TrainingArguments(args)
    # print(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("<seg_patch>")
    args.seg_token_idx = tokenizer("<seg_patch>", add_special_tokens=False).input_ids[0]
    
    model_args = {
            "train_mask_decoder": args.train_mask_decoder,
            "mm_patch_merge_type": args.mm_patch_merge_type,
            "out_dim": args.out_dim,
            "ce_loss_weight": args.ce_loss_weight,
            "dice_loss_weight": args.dice_loss_weight,
            "bce_loss_weight": args.bce_loss_weight,
            "seg_token_idx": args.seg_token_idx,
            "save_output_dir": args.save_output_dir,
            "vision_tower": args.vision_tower,
            "use_mm_start_end": args.use_mm_start_end,
            "mm_vision_select_layer": args.mm_vision_select_layer,
            "mm_hidden_size": args.mm_hidden_size,
            "stage": args.stage,
            "dataset_time": args.dataset_time,
            "mm_projector_type": args.mm_projector_type,
        }
    
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    transformers_logging.set_verbosity_error()
    
    model = AffordanceForCasualLM.from_pretrained(
            args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=False, **model_args
        )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    
    model.resize_token_embeddings(len(tokenizer))
    
    if model.config.stage == "pretrain":
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
            
    if args.lora_enable:
        lora_r = args.lora_r
        if lora_r > 0:

            def find_linear_layers(model, lora_target_modules):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    if (
                        isinstance(module, cls)
                        and all(
                            [
                                x not in name
                                for x in [
                                    "visual_model",
                                    "vision_tower",
                                    "mm_projector",
                                ]
                            ]
                        )
                        and any([x in name for x in lora_target_modules])
                    ):
                        lora_module_names.add(name)
                return sorted(list(lora_module_names))

            lora_alpha = args.lora_alpha
            lora_dropout = args.lora_dropout
            lora_target_modules = find_linear_layers(
                model, args.lora_target_modules.split(",")
            )
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
        for n, p in model.named_parameters():
            if any(
                [
                    x in n
                    for x in ["lm_head", "embed_tokens", "mm_projector", "neck", "Neck", "mask_decoder", "prompt_deocder", "text_hidden_fcs"]
                ]
            ):
                # print("n: ", n, "p.shape: ", p.shape)
                p.requires_grad = True
        model.print_trainable_parameters()

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    if args.conv_version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if args.vision_tower is not None:
        args.image_processor = vision_tower.image_processor
        args.is_multimodal = True
        if args.stage == "finetune":
            image_aspect_ratio = "pad"
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokneizer_model_max_length = tokenizer.model_max_length
        model.config.stage = args.stage
        model.config.dataset_time = args.dataset_time

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                                  data_args=args)
    
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=data_module["train_dataset"],
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_version=args.conv_version,
            local_rank=args.local_rank
        ),
        config=ds_config,
    )
    
    print(model)

    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0
    # print("Model Architecture\n", model)
    # print("Training Args\n", args)
    
    for epoch in range(args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )
        
        if args.stage != "pretrain":
            if args.dataset_time == "train":
            # if epoch % 2 == 1:
                save_dir = os.path.join(args.log_dir, "ckpt_model_" + str(epoch))
                model_engine.save_checkpoint(save_dir)
            # print(model_engine)
        else:
            save_mm_dir = os.path.join(args.log_dir, "ckpt_model_" + str(epoch) + "/mm_projector/")
            os.makedirs(save_mm_dir, exist_ok=True)
            keys_to_match = ['mm_projector']
            weight_to_save = get_mm_adapter_state_maybe_zero_3(model_engine.model.named_parameters(), keys_to_match)
            torch.save(weight_to_save, os.path.join(save_mm_dir, f"mm_projector.bin"))
            mm_projector_folder = os.path.join(parent_folder, "mm_projector")

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return
    
def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    progress_seq_length = len(train_loader)
    
    progress = ProgressMeter(
        progress_seq_length,
        [
            batch_time,
            losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(len(train_loader)):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
            
            output_dict = model(**input_dict, pretraining=True)

            loss = output_dict["loss"]

            losses.update(loss.item(), input_dict["images"].size(0))

            model.backward(loss)
            model.step()
                
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                if not model.module.config.stage == "pretrain":
                    ce_losses.all_reduce()
                    fical_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step + global_step * epoch)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step + global_step * epoch
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step + global_step * epoch
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()

        if args.dataset_time == "train":
            if global_step != 0:
                curr_lr = scheduler.get_last_lr()
                if args.local_rank == 0:
                    writer.add_scalar("train/lr", curr_lr[0], global_step)
                
    return train_iter

if __name__ == "__main__":
    main(sys.argv[1:])
