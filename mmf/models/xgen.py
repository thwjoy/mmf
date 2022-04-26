# Copyright (c) Facebook, Inc. and its affiliates.
import collections.abc
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.models.transformers.heads.utils import build_heads_dict
from mmf.modules.encoders import TransformerEncoder, ViTEncoder
from mmf.modules.losses import MMFLoss
from mmf.utils.build import build_encoder, build_classifier_layer, build_image_encoder, build_text_encoder
from mmf.utils.modeling import get_bert_configured_parameters
from omegaconf import MISSING, OmegaConf
import numpy as np
from torch import Tensor, nn
import torch.nn.functional as F

# mmf_run config="configs/experiments/cross_gen/defaults.yaml" \
#     model=cross_gen \
#     dataset=hateful_memes \
#     run_type=train_val

class Tokens2Image(nn.Module):
    """
    map from a set of feature vectors (one per patch) back to an image.
    """

    def __init__(self, dIn, P, outputSize):
        super(Tokens2Image, self).__init__()
        self.P = P
        self.outputSize = tuple(outputSize)
        self.project = nn.Linear(dIn, P * P * 3)
        self.project.bias.data.fill_(0.5)

    def forward(self, tokens): 
        tokens = self.project(tokens).transpose(2, 1)
        I = F.fold(tokens, self.outputSize, self.P, stride=self.P)  # B x 3 x H x W
        return torch.tanh(I)
 
class XGenImageEmbedding(nn.Module):
    """
    Patch embedding used for ViLT.
    https://arxiv.org/pdf/2102.03334.pdf
    Implementation based off
    https://github.com/dandelin/ViLT/blob/master/vilt/modules/vilt_module.py
    Using huggingface ViT modules.
    Can be built with random init or the embeddings weights from an exisiting
    ViT model from huggingface. Model list: availible at
    https://huggingface.co/models?other=vit&sort=downloads
    """
    def __init__(
        self,
        random_init: bool = True,
        pretrained_model_name: str = "google/vit-base-patch16-224",
        image_size: Optional[List] = None,
        hidden_dropout_prob: Optional[float] = None,
        hidden_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        num_channels: Optional[int] = None,
        keep_frac: Optional[float] = 0.25,
        *args,
        **kwargs
    ):
        super().__init__()
        config = OmegaConf.create(
            {"random_init": random_init, "pretrained_model_name": pretrained_model_name}
        )
        if image_size is not None:
            config.image_size = image_size
        if hidden_dropout_prob is not None:
            config.hidden_dropout_prob = hidden_dropout_prob
        if hidden_size is not None:
            config.hidden_size = hidden_size
        if patch_size is not None:
            config.patch_size = patch_size
        if num_channels is not None:
            config.num_channels = num_channels

        config.do_patch_embeddings = False
        self.encoder = ViTEncoder(config)
        # self.embedding = self.encoder.embeddings

    def forward(self, image: Tensor) -> Tensor:
        if image.dim() == 5:
            image = image.permute(1, 0, 2, 3, 4).flatten(start_dim=0, end_dim=1)
        img_embeddings = self.encoder(image)
        return img_embeddings

class ProjectionHead(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x




@registry.register_model("xgen")
class XGen(BaseModel):
    @classmethod
    def config_path(cls):
        return "configs/models/xgen/defaults.yaml"

    def __init__(self, config):
        super().__init__(config)
        self.build()

    def build(self):
        self.text_embeddings = TransformerEncoder(self.config.text_encoder.params)
        self.image_embeddings = ViTEncoder(self.config.image_encoder.params)
        self.joint_encoder = ViTEncoder(self.config.joint_encoder.params)

        self.mask_ratio_im = 0.25
        self.mask_ratio_text = 0.85
        self.patch_size = self.config.image_encoder.params.patch_size
        self.im_shape = self.config.image_encoder.params.image_size

        self.text_masked_token = torch.nn.Parameter(torch.randn(1, 1, self.config.joint_encoder.params.hidden_dim))
        self.dec_masked_token_im = torch.nn.Parameter(torch.randn(1, 1, self.config.joint_encoder.params.hidden_dim))

        vocab_size = self.text_embeddings.embeddings.word_embeddings.weight.shape[0]

        if "cross_transformative_loss" in self.config.losses:
            self.tokens2im = Tokens2Image(self.config.joint_encoder.params.hidden_dim,
                                      self.patch_size,
                                      self.im_shape)

            self.tokens2text = torch.nn.Linear(self.config.joint_encoder.params.hidden_dim, vocab_size)
            self.joint_decoder = ViTEncoder(self.config.joint_decoder.params)

        
        if "contrastive_loss" in self.config.losses:
            self.proj_im = ProjectionHead(self.config.joint_encoder.params.hidden_dim)
            self.proj_text = ProjectionHead(self.config.joint_encoder.params.hidden_dim)
            self.log_temp = nn.Parameter(torch.tensor(0.0)) # init to t = 1

        # build network head
        if "classifier" in self.config:
            self.classifier = build_classifier_layer(self.config.classifier)

        # head_configs = self.config.get("heads", {})
        # self.tasks = self.config.get("tasks", head_configs.keys())
        # if isinstance(self.tasks, str):
        #     self.tasks = self.tasks.split(",")

        # self.losses = nn.ModuleDict()
        # self.heads_dict = build_heads_dict(head_configs, self.tasks, self.losses)
        # self.modality_keys = self.modality_type = ["text", "image"]

    # def init_losses(self):
    #     loss_configs = self.config.get("losses", {})
    #     for loss_name, loss_config in loss_configs.items():
    #         self.losses[loss_name] = MMFLoss(loss_config)

    def _get_perm(self, N, device):
        perm = torch.randperm(N, device=device)
        invPerm = torch.empty_like(perm)
        invPerm[perm] = torch.arange(N, device=device)
        return perm, invPerm

    def forward(self, sample_list):
        text_embedding = self.text_embeddings(
            sample_list["input_ids"], sample_list["segment_ids"], return_sequence=True
        )
        image_embedding, _ = self.image_embeddings(sample_list["image"])

        # Feed through encoder
        n_img_tokens = image_embedding.size(1) - 1
        n_text_tokens = text_embedding.size(1)

        # get perms for 
        perm_im, invPerm_im = self._get_perm(n_img_tokens, device=image_embedding.device)
        s_im = int(self.mask_ratio_im * n_img_tokens)

        # replace txt with masking tokens
        perm_text, invPerm_text = self._get_perm(n_text_tokens, device=text_embedding.device)
        s_text = int(self.mask_ratio_text * n_text_tokens)
        text_tokens_keep = text_embedding[:, perm_text[:s_text]]
        text_embedding = torch.cat([text_tokens_keep,
                                    self.text_masked_token.expand_as(text_embedding[:, perm_text[s_text:]])], dim=1)
        text_embedding = text_embedding[:, invPerm_text]

        attention_mask_joint, attention_mask_im, attention_mask_text = self.get_attention_mask(
            sample_list, text_embedding, image_embedding, invPerm_im[:s_im]
        )

        embeddings = torch.cat([text_embedding, image_embedding], dim=1)

        # forward seqnece when masking out image and then text
        sequence_im, _ = self.joint_encoder(embeddings, attention_mask=attention_mask_im)
        sequence_text, _ = self.joint_encoder(embeddings, attention_mask=attention_mask_text)

        # apply masking tokens
        
        bs = image_embedding.size(0)
        sequence_im[:, perm_im[s_im:], :] = self.dec_masked_token_im.expand(bs, len(perm_im[s_im:]), -1)
        sequence_text[:, :image_embedding.size(1), :] = self.dec_masked_token_im.expand(bs, image_embedding.size(1), -1)

        ret_dict = {}

        if "cross_transformative_loss" in self.config.losses:
            decoded_im, _ = self.joint_decoder(sequence_im, attention_mask=attention_mask_joint)
            decoded_text, _ = self.joint_decoder(sequence_text, attention_mask=attention_mask_joint)

            r_img = self.tokens2im(decoded_im[:, 1:image_embedding.size(1), :])
            r_text = self.tokens2text(decoded_text[:, image_embedding.size(1):, :])
            x_img = self.tokens2im(decoded_text[:, 1:image_embedding.size(1), :])
            x_text = self.tokens2text(decoded_im[:, image_embedding.size(1):, :])   

            mask_im = self.imageMask(perm_im[:s_im], self.im_shape, r_img.device).expand(bs, -1, -1, -1)

            ret_dict.update({'r_img': r_img, 
                            'r_text': r_text,
                            'x_img': x_img, 
                            'x_text': x_text,
                            'mask_im': mask_im,
                            'mask_text': perm_text[s_text:],
                            'attn_mask': sample_list['input_mask']})

        elif "contrastive_loss" in self.config.losses:
            ret_dict.update({'embedding_1': self.proj_text(sequence_im.mean(1)),
                            'embedding_2': self.proj_im(sequence_text.mean(1)),
                            'temperature': self.log_temp.exp()})

        # # perfom head
        scores = None
        if "classifier" in self.config:
            sequence, _ = self.joint_encoder(embeddings, attention_mask=attention_mask_im)
            scores = self.classifier(sequence[:, 0, :])
            ret_dict["scores"] = scores

        return ret_dict

    def imageMask(self, sampleIds, outputSize, device):
        outputSize = tuple(outputSize)
        N = outputSize[0] // self.patch_size * outputSize[1] // self.patch_size
        # B x P*P x H/P*W/P
        mask = torch.ones((1, 3 * self.patch_size * self.patch_size, N), device=device)
        mask[0, :, sampleIds] = 0
        # B x C x H x W
        mask = F.fold(mask, outputSize, self.patch_size, stride=self.patch_size)
        return mask.to(torch.bool) 

    def get_attention_mask(
        self,
        sample_list: Dict[str, Tensor],
        text_embedding: Tensor,
        image_embedding: Tensor,
        im_inds
    ) -> Tensor:
        text_mask = getattr(sample_list, "input_mask", None)
        image_mask = getattr(sample_list, "image_mask", None)

        if text_mask is None and image_mask is None:
            return None

        if text_mask is None:
            text_mask = torch.ones(
                text_embedding.size()[:-1],
                dtype=text_embedding.dtype,
                device=text_embedding.device,
            )

        if image_mask is None:
            image_mask = torch.zeros(
                image_embedding.size()[:-1],
                dtype=image_embedding.dtype,
                device=image_embedding.device,
            )

        image_mask[:, im_inds] = 1.0

        attention_mask_joint = torch.cat((image_mask, text_mask), dim=-1)
        attention_mask_im = torch.cat((torch.zeros_like(image_mask), text_mask), dim=-1)
        attention_mask_text = torch.cat((image_mask, torch.zeros_like(text_mask)), dim=-1)
        return attention_mask_joint, attention_mask_im, attention_mask_text

