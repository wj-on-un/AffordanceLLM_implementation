import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms as transforms
from transformers import BitsAndBytesConfig
from transformers import OwlViTVisionModel, OwlViTImageProcessor, OwlViTVisionConfig

from llava_aff.util.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_aff.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)

from llava_aff.model.mask_decoder.mask_decoder import Neck, MaskDecoder
from llava_aff.model.mask_decoder.prompt_encoder import PromptEncoder
from llava_aff.model.mask_decoder.twowaytrans import TwoWayTransformer
from llava_aff.model.mask_decoder.common import LayerNorm2d

from PIL import Image
import numpy as np
    
class AffordanceMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(AffordanceMetaModel, self).__init__(config)

        self.config = config
        self.prompt_embed_dim = 256
        self.image_size = 768
        self.vit_patch_size = 32
        self.image_embedding_size = self.image_size // self.vit_patch_size
        self.config.train_mask_decoder = kwargs["train_mask_decoder"]

        if self.config.train_mask_decoder:
            self.config.out_dim = kwargs["out_dim"]
        else:
            self.neck =  nn.Sequential(
                nn.Conv2d(in_channels=768*2, out_channels=768*2, kernel_size=1, stride=1, bias=False),
                LayerNorm2d(768*2),  # Assuming LayerNorm2d is meant to be BatchNorm2d
                nn.Conv2d(in_channels=768*2, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                LayerNorm2d(256)  # Again assuming LayerNorm2d is meant to be BatchNorm2d
            )
            
            self.mask_decoder = MaskDecoder(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=self.prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                    ),
                transformer_dim=self.prompt_embed_dim,
                    iou_head_depth=3,
                    iou_head_hidden_dim=256,
            )
    
            self.prompt_encoder = PromptEncoder(
                embed_dim=self.prompt_embed_dim,
                image_embedding_size=(self.image_embedding_size, self.image_embedding_size),
                input_image_size=(self.image_size, self.image_size),
                mask_in_chans=16,
            )
            
            in_dim = config.hidden_size
            out_dim = 256
            text_fc = [
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, out_dim),
                nn.Dropout(0.0),
            ]
    
            self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
    
class AffordanceModel(AffordanceMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(AffordanceModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False

## lmsys/vicuna-7b-v1.5
class AffordanceForCasualLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "google/owlvit-base-patch32"
            )
            config.mm_vision_select_layer = kwargs.pop("mm_vision_select_layer", -2)
            config.mm_hidden_size = kwargs.pop("mm_hidden_size", 1024)
            config.mm_patch_merge_type = kwargs.pop("mm_patch_merge_type", 'flat')
            config.stage = kwargs.pop("stage", "pretrain")
            # config.dataset_time = kwargs.pop("dataset_time", "train")
            config.mm_projector_type = kwargs.pop("mm_projector_type", "mlp2x_gelu")
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower

        self.seg_token_idx = kwargs.pop("seg_token_idx")
        super().__init__(config)
        self.model = AffordanceModel(config, **kwargs)
        self.post_init()

    def get_visual_embs(self, pixel_values):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.vision_tower(pixel_values[i].unsqueeze(0))
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)
    
    def prepare_inputs_labels_for_multimodal(self, input_ids, labels, attention_mask, position_ids, input_image_features):
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = input_image_features[cur_image_idx]
                cur_input_embeds_1 = self.model.embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]].to(torch.long))
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.model.embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = input_image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return new_input_embeds, new_labels, attention_mask

    def model_forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: torch.LongTensor,
        images: torch.FloatTensor,
        depth_images: torch.FloatTensor = None,
        gt_images: torch.FloatTensor = None,
        gt_images_path = None,
        position_ids = None,
        inference: bool=False,
        pretraining: bool=True,
        **kwargs
    ):
        
        # print(self.model)
        if pretraining:
            image_embeddings = self.get_visual_embs(images)
            image_mm_out = self.model.mm_projector(image_embeddings)
        else:
            image_embeddings = self.get_visual_embs(images)
            depth_image_embeddings = self.get_visual_embs(depth_images)
            
            ## vit image embedding to mm_projector
            image_mm_out = self.model.mm_projector(image_embeddings)
            depth_mm_out = self.model.mm_projector(depth_image_embeddings)

        
        batch_size, sequence_tokens, hidden_size = image_mm_out.shape
        if pretraining:
            image_features = image_mm_out.view(batch_size, int(sequence_tokens/4), int(hidden_size*4))
        else:
            image_features = image_mm_out.view(batch_size, int(sequence_tokens/4), int(hidden_size*4))
            depth_features = depth_mm_out.view(batch_size, int(sequence_tokens/4), int(hidden_size*4))

        if pretraining:
            input_image_features = image_features
        else:
            ## mm_project embedding concat (like 2 images)
            input_image_features = torch.cat((image_features, depth_features), dim=1)

        llm_input_embeds, llm_labels, attention_mask = self.prepare_inputs_labels_for_multimodal(input_ids, labels, attention_mask, position_ids, input_image_features)
        
        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=llm_input_embeds,
            labels=llm_labels,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=None
        )
        
        ## pre-training
        if pretraining:
            loss = outputs.loss
            return {
                "loss": loss,
                "outputs": outputs,
                "image_mm_out": image_mm_out
            }
        ## Fine-tuning (Refer to LISA. - https://github.com/dvlab-research/LISA/blob/main/model/LISA.py)
        else:
            ## vit image embedding concat
            image_and_depth_features = torch.cat((image_embeddings, depth_image_embeddings), dim=-1)
            image_and_depth_features = image_and_depth_features.view(batch_size, 24, 24, 768*2)
            neck_output = self.model.neck(image_and_depth_features.permute(0,3,1,2))
            
            hidden_states = []
            outputs_last_hidden_states = outputs.hidden_states
            if inference == False:
                hidden_states.append(self.model.text_hidden_fcs[0](outputs_last_hidden_states[-1]))
            else:
                hidden_states.append(self.model.text_hidden_fcs[0](outputs_last_hidden_states[-1]))
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

            seg_token_mask = llm_labels == self.seg_token_idx
            
            pred_embeddings = last_hidden_state[seg_token_mask]
            seg_token_counts = seg_token_mask.int().sum(-1)
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )
            
            pred_embeddings_ = []
            for i in range(len(seg_token_offset) -1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i+1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)

                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=neck_output[i].unsqueeze(0),
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_masks.append(low_res_masks[:, 0])
                
            gt_masks = gt_images
            
            loss = outputs.loss
            return {
                "loss": loss,
                "outputs": outputs,
                "image_mm_out": image_mm_out,
                "depth_mm_out": depth_mm_out,
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "gt_masks_path": gt_images_path
            }
        
        
        
        