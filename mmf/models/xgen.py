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
from mmf.utils.build import build_encoder
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
        self.embedding = self.encoder.embeddings
        # self.keep_frac = keep_frac 
      
        # posEmbeddingDomainSize = int((image_size[0] * image_size[1]) // (patch_size**2))
        # self.position_embedding = nn.Parameter(
        #     torch.randn(1, posEmbeddingDomainSize, self.encoder.hf_config.hidden_size)
        # )

    def forward(self, image: Tensor) -> Tensor:
        if image.dim() == 5:
            image = image.permute(1, 0, 2, 3, 4).flatten(start_dim=0, end_dim=1)
        img_embeddings = self.embedding(image)
        return img_embeddings

        # N = img_embeddings.size(1)
        # B = img_embeddings.size(0)
        # S = int(self.keep_frac * (N - 1))
        # # # get the permutation and inverse permutation used to sample S tokens from the input

        # perm = torch.cat([torch.zeros((1), dtype=torch.long, device=img_embeddings.device),
        #                   torch.randperm(N - 1, device=img_embeddings.device) + 1])
        # # make sure 0 is the first index
        
        # # mask out tokens
        # invPerm = torch.empty_like(perm)
        # invPerm[perm] = torch.arange(N, device=img_embeddings.device)
        # # # encode the randomly permuted set of input tokens
        # img_embeddings_samples = img_embeddings[:, perm[0:S+1], :] # include cls
        # pos_samples = self.position_embedding[:, perm[1:S+1] - 1, :] # does not include cls

        # # pass tokens through encoder
        # pos_embeds = torch.cat([torch.zeros_like(self.position_embedding[:, 0, :].unsqueeze(1)),
        #                         pos_samples], dim=1)

        # output, _ = self.encoder(img_embeddings_samples + pos_embeds)

        # return output, perm[0:S], invPerm

class XGenTextEmbedding(nn.Module):
    def __init__(
        self,
        random_init: bool = True,
        bert_model_name: str = "bert-base-uncased",
        hidden_size: Optional[int] = None,
        max_position_embeddings: Optional[int] = None,
        keep_frac: Optional[float] = 0.25,
        *args,
        **kwargs
    ):

        super().__init__()
        config = OmegaConf.create(
            {"bert_model_name": bert_model_name, "random_init": random_init}
        )
        if hidden_size is not None:
            config.hidden_size = hidden_size
        if max_position_embeddings is not None:
            config.max_position_embeddings = max_position_embeddings

        self.keep_frac = keep_frac
        self.text_encoder = TransformerEncoder(config)
        self.text_embeddings = self.text_encoder.embeddings

        self.position_embedding = nn.Parameter(
            torch.randn(1, config.max_position_embeddings, self.text_encoder.config.hidden_size)
        )


    def forward(self, input_ids: Tensor, segment_ids: Tensor) -> Tensor:
        text_embedding = self.text_embeddings(input_ids, token_type_ids=segment_ids)

        # N = text_embedding.size(1)
        # B = text_embedding.size(0)
        # S = int(self.keep_frac * (N - 1))
        # # # get the permutation and inverse permutation used to sample S tokens from the input

        # perm = torch.cat([torch.zeros((1), dtype=torch.long, device=text_embedding.device),
        #                   torch.randperm(N - 1, device=text_embedding.device) + 1])
        # # make sure 0 is the first index
        
        # # mask out tokens
        # invPerm = torch.empty_like(perm)
        # invPerm[perm] = torch.arange(N, device=text_embedding.device)
        # # # encode the randomly permuted set of input tokens
        # text_embedding_samples = text_embedding[:, perm[0:S+1], :] # include cls
        # pos_samples = self.position_embedding[:, perm[1:S+1] - 1, :] # does not include cls

        # # pass tokens through encoder
        # pos_embeds = torch.cat([torch.zeros_like(self.position_embedding[:, 0, :].unsqueeze(1)),
        #                         pos_samples], dim=1)

        # output = self.text_encoder.module(None, inputs_embeds=text_embedding_samples + pos_embeds)[0]

        return text_embedding


@registry.register_model("xgen")
class XGen(BaseModel):
    # @dataclass
    # class Config(BaseModel.Config):
    #     name: str = "XGen"
    #     text_embeddings: Any = MISSING
    #     image_encoder: Any = MISSING

    @classmethod
    def config_path(cls):
        return "configs/models/xgen/defaults.yaml"

    def __init__(self, config):
        super().__init__(config)
        self.build()

    def build(self):
        self.text_embeddings = XGenTextEmbedding(**self.config.text_embeddings)
        self.image_embeddings = XGenImageEmbedding(**self.config.image_encoder.params)
        self.joint_encoder = ViTEncoder(self.config.joint_encoder.params)
        self.joint_decoder = ViTEncoder(self.config.joint_decoder.params)

        self.mask_ratio_im = 0.25
        self.mask_ratio_text = 0.85
        self.patch_size = self.config.image_encoder.params.patch_size
        self.im_shape = self.config.image_encoder.params.image_size

        self.text_masked_token = torch.nn.Parameter(torch.randn(1, 1, self.config.joint_encoder.params.hidden_dim))
        self.dec_masked_token_im = torch.nn.Parameter(torch.randn(1, 1, self.config.joint_encoder.params.hidden_dim))
        self.dec_masked_token_text = torch.nn.Parameter(torch.randn(1, 1, self.config.joint_encoder.params.hidden_dim))

        # patch_size = int(np.prod(self.config.image_encoder.params.image_size) // 
        #                  self.config.image_encoder.params.patch_size**2)

        vocab_size = self.text_embeddings.text_encoder.embeddings.word_embeddings.weight.shape[0]

        
        self.tokens2im = Tokens2Image(self.config.joint_encoder.params.hidden_dim,
                                      self.patch_size,
                                      self.im_shape)

        self.tokens2text = torch.nn.Linear(self.config.joint_encoder.params.hidden_dim, vocab_size)

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
            sample_list["input_ids"], sample_list["segment_ids"]
        )
        image_embedding = self.image_embeddings(sample_list["image"])

        # Feed through encoder
        embeddings = torch.cat([text_embedding, image_embedding], dim=1)

        n_img_tokens = image_embedding.size(1) - 1
        n_text_tokens = text_embedding.size(1)

        # get perms for 
        perm_im, invPerm_im = self._get_perm(n_img_tokens, device=embeddings.device)
        s_im = int(self.mask_ratio_im * n_img_tokens)

        # replace txt with masking tokens
        perm_text, invPerm_text = self._get_perm(n_text_tokens, device=embeddings.device)
        s_text = int(self.mask_ratio_text * n_text_tokens)
        text_tokens_mask = self.text_masked_token.expand_as(text_embedding[:, perm_text[s_text:]])
        text_tokens_keep = text_embedding[:, perm_text[:s_text]]
        text_embedding = torch.cat([text_tokens_keep, text_tokens_mask], dim=1)
        text_embedding = text_embedding[:, invPerm_text]

        attention_mask_joint, attention_mask_im, attention_mask_text = self.get_attention_mask(
            sample_list, text_embedding, image_embedding, invPerm_im[:s_im]
        )

        # forward seqnece when masking out image and then text
        sequence_im, _ = self.joint_encoder(embeddings, attention_mask=attention_mask_im)
        sequence_text, _ = self.joint_encoder(embeddings, attention_mask=attention_mask_text)

        # apply masking tokens
        
        bs = image_embedding.size(0)
        sequence_im[:, perm_im[s_im:], :] = self.dec_masked_token_im.expand(bs, len(perm_im[s_im:]), -1)
        sequence_text[:, :image_embedding.size(1), :] = self.dec_masked_token_im.expand(bs, image_embedding.size(1), -1)

        decoded_im, _ = self.joint_decoder(sequence_im, attention_mask=attention_mask_joint)
        decoded_text, _ = self.joint_decoder(sequence_text, attention_mask=attention_mask_joint)

        r_img = self.tokens2im(decoded_im[:, 1:image_embedding.size(1), :])
        r_text = self.tokens2text(decoded_text[:, image_embedding.size(1):, :])
        x_img = self.tokens2im(decoded_text[:, 1:image_embedding.size(1), :])
        x_text = self.tokens2text(decoded_im[:, image_embedding.size(1):, :])   

        mask_im = self.imageMask(perm_im[:s_im], self.im_shape, r_img.device).expand(bs, -1, -1, -1)

        return {'r_img': r_img, 
                'r_text': r_text,
                'x_img': x_img, 
                'x_text': x_text,
                'mask_im': mask_im,
                'mask_text': perm_text[s_text:],
                'attn_mask': sample_list['input_mask']}
                # 'feats_text': self.proj_text(tokens_enc_text.mean(0)),
                # 'feats_im': self.proj_im(tokens_enc_img.mean(0))}

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

    # def _infer_itm_labels(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
    #     input_ids = sample_list["input_ids"]
    #     itm_labels = {}
    #     if "is_correct" in sample_list:
    #         itm_labels["is_correct"] = sample_list["is_correct"]
    #     else:
    #         itm_labels["is_correct"] = torch.tensor(
    #             True, dtype=torch.long, device=input_ids.device
    #         )

    #     return itm_labels

    # def _infer_mlm_labels(
    #     self, sample_list: Dict[str, Tensor], image_embeddings_size: Tuple[int, int]
    # ):
    #     input_ids = sample_list["input_ids"]
    #     mlm_labels = {}
    #     current_text_idx = 0
    #     if "lm_label_ids" in sample_list:
    #         if sample_list["lm_label_ids"].dim() > 2:
    #             mlm_labels["text"] = sample_list["lm_label_ids"][:, current_text_idx]
    #             current_text_idx += 1
    #         else:
    #             mlm_labels["text"] = sample_list["lm_label_ids"]
    #     else:
    #         mlm_labels["text"] = torch.full(
    #             input_ids.size(),
    #             fill_value=-1,
    #             dtype=torch.long,
    #             device=input_ids.device,
    #         )
    #     mlm_labels["image"] = torch.full(
    #         image_embeddings_size,
    #         fill_value=-1,
    #         dtype=torch.long,
    #         device=input_ids.device,
    #     )
    #     mlm_labels["combined_labels"] = torch.cat(
    #         [mlm_labels["text"], mlm_labels["image"]], dim=-1
    #     )
    #     return mlm_labels

    # def _encode_mlm(self, sample_list: Dict[str, Tensor], image_embedding: Tensor):
    #     assert "lm_label_ids" in sample_list

    #     input_ids = sample_list.get("input_ids_masked", sample_list["input_ids"])
    #     segment_ids = sample_list["segment_ids"]
    #     text_embedding = self.text_embeddings(input_ids, segment_ids)

    #     embeddings = torch.cat([image_embedding, text_embedding], dim=1)
    #     attention_mask = self.get_attention_mask(
    #         sample_list, text_embedding, image_embedding
    #     )
    #     sequence, _ = self.encoder(embeddings, attention_mask=attention_mask)
    #     if sequence.dim() != 3:
    #         sequence = sequence.unsqueeze(1)

    #     sample_list["hs_masked_for_mlm"] = sequence
