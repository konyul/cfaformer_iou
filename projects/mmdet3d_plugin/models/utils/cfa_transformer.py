# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_
from einops import rearrange
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from mmdet3d.models import builder

from projects.mmdet3d_plugin.models.dense_heads.meformer_head import pos2embed

@TRANSFORMER.register_module()
class CFATransformer(BaseModule):
    def __init__(
            self,
            use_type_embed=True,
            use_cam_embed=False,
            encoder=None,
            decoder=None,
            heads=None,
            separate_head=None,
            init_cfg=None,
            modal_seq=None,
            num_classes=None,
            numq_per_modal=100,
            query_modal_types=['fused', 'bev', 'img'],
            cross=False,
            modal_embedding=False,
            failure_pred=False,
    ):
        super(CFATransformer, self).__init__(init_cfg=init_cfg)

        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.use_type_embed = use_type_embed
        self.use_cam_embed = use_cam_embed
        self.modal_seq = modal_seq
        assert len(self.modal_seq) == self.decoder.num_layers, 'cfa modal_seq must same with decoder.num_layers'

        self.use_self_attn = 'self_attn' in self.decoder.layers[0].operation_order
        if self.use_self_attn:
            self.dist_scaler = nn.Parameter(torch.randn(1), requires_grad=True)
            self.dist_bias = nn.Parameter(torch.randn(1), requires_grad=True)
            self.num_heads = decoder["transformerlayers"]["attn_cfgs"][0]["num_heads"]
        if self.use_type_embed:
            self.bev_type_embed = nn.Parameter(torch.randn(self.embed_dims))
            self.rv_type_embed = nn.Parameter(torch.randn(self.embed_dims))
        else:
            self.bev_type_embed = None
            self.rv_type_embed = None

        if self.use_cam_embed:
            self.cam_embed = nn.Sequential(
                nn.Conv1d(16, self.embed_dims, kernel_size=1),
                nn.BatchNorm1d(self.embed_dims),
                nn.Conv1d(self.embed_dims, self.embed_dims, kernel_size=1),
                nn.BatchNorm1d(self.embed_dims),
                nn.Conv1d(self.embed_dims, self.embed_dims, kernel_size=1),
                nn.BatchNorm1d(self.embed_dims)
            )
        else:
            self.cam_embed = None

        self.cross = cross

        self.numq_per_modal = numq_per_modal
        self.numq_per_modal_or = numq_per_modal * 7 // 10
        self.numq_per_modal_dn = numq_per_modal * 3 // 10
        assert self.numq_per_modal_or + self.numq_per_modal_dn == self.numq_per_modal, "or+dn must be euqal with total number of query"
        self.num_classes = num_classes
        self.query_modal_types = query_modal_types

        # modal embedding
        self.modal_embedding = modal_embedding
        if self.modal_embedding:
            embed_dims = self.decoder.embed_dims
            self.modal_embed = {'fused': nn.Parameter(torch.Tensor(embed_dims)).cuda(),
                                'bev': nn.Parameter(torch.Tensor(embed_dims)).cuda(),
                                'img': nn.Parameter(torch.Tensor(embed_dims)).cuda()}

        # for KV
        self.task_heads = nn.ModuleList()
        for num_cls in num_classes:
            heads = copy.deepcopy(heads)
            heads.update(dict(cls_logits=(num_cls, 2)))
            separate_head.update(
                in_channels=self.embed_dims,
                heads=heads, num_cls=num_cls,
                groups=decoder.num_layers
            )
            self.task_heads.append(builder.build_head(separate_head))

        # for Q
        self.modality_proj = nn.ModuleDict({
            "fused": nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims)
            ),
            "bev": nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims)
            ),
            "img": nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims)
            )
        })
        self.box_pos_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims)
        )
        self.failure_pred = failure_pred

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True
        if self.modal_embedding:
            for modal in self.modal_embed.keys():
                normal_(self.modal_embed[modal])

    def select_top_queries(self, cls_scores, center, x_proj, reference, num_queries_per_modality, pad_size):
        batch_size, num_queries, num_classes = cls_scores.shape

        center_dim = center.shape[2]
        x_proj = x_proj.permute(1, 0, 2) # -> [B, num_q, C]
        
        selected_center_list = []
        selected_x_proj_list = []
        selected_reference_list = []
        selected_q_idx_list = []

        for b in range(batch_size):
            batch_cls_scores = cls_scores[b]
            batch_cls_scores_max = batch_cls_scores.max(1)[0]

            if self.training:
                tk_val, tk_ind_or = batch_cls_scores_max[pad_size:].topk(self.numq_per_modal_or)
                tk_val, tk_ind_dn = batch_cls_scores_max[:pad_size].topk(min(pad_size, self.numq_per_modal_dn))
                tk_ind = torch.cat([tk_ind_dn, tk_ind_or+pad_size], dim=0)
            else:
                tk_val, tk_ind = batch_cls_scores_max.topk(self.numq_per_modal)

            selected_center_list.append(center[b][tk_ind])
            selected_x_proj_list.append(x_proj[b][tk_ind])
            selected_reference_list.append(reference[b][tk_ind])
            selected_q_idx_list.append(tk_ind)

        selected_centers_tensor = torch.stack(selected_center_list)
        selected_x_proj_tensor = torch.stack(selected_x_proj_list)
        selected_reference_tensor = torch.stack(selected_reference_list)
        selected_q_idx_tensor = torch.stack(selected_q_idx_list)

        return selected_centers_tensor, selected_x_proj_tensor.permute(1, 0, 2), selected_reference_tensor, selected_q_idx_tensor

    def permute_dn(self, targ, dim):
        if dim == 0:
            part1 = torch.cat([targ[0:self.numq_per_modal_dn], 
                               targ[self.numq_per_modal:self.numq_per_modal+self.numq_per_modal_dn],
                               targ[self.numq_per_modal*2:self.numq_per_modal*2+self.numq_per_modal_dn]], dim=0)
            part2 = torch.cat([targ[self.numq_per_modal_dn:self.numq_per_modal], 
                               targ[self.numq_per_modal+self.numq_per_modal_dn:self.numq_per_modal*2],
                               targ[self.numq_per_modal*2+self.numq_per_modal_dn:self.numq_per_modal*3]], dim=0)
        elif dim == 1:
            part1 = torch.cat([targ[:, 0:self.numq_per_modal_dn], 
                               targ[:, self.numq_per_modal:self.numq_per_modal+self.numq_per_modal_dn],
                               targ[:, self.numq_per_modal*2:self.numq_per_modal*2+self.numq_per_modal_dn]], dim=1)
            part2 = torch.cat([targ[:, self.numq_per_modal_dn:self.numq_per_modal], 
                               targ[:, self.numq_per_modal+self.numq_per_modal_dn:self.numq_per_modal*2],
                               targ[:, self.numq_per_modal*2+self.numq_per_modal_dn:self.numq_per_modal*3]], dim=1)

        return torch.cat([part1, part2], dim=dim)

    def forward(self,
                x,
                x_img,
                bev_query_embed,
                rv_query_embed,
                bev_pos_embed,
                rv_pos_embed,
                img_metas,
                outs_dec,
                reference,
                outs,
                num_queries_per_modality,
                pc_range,
                ca_dict,
                task_id,
                pad_size,
                attn_masks=None,
                reg_branch=None):
        # prepare Q (camera+lidar+fused object query)
        # [-1] due to return_intermediate
        outs_dec = outs_dec[-1].transpose(0, 1)
        outs_dec = list(outs_dec.split(num_queries_per_modality, dim=0))
        x_proj = []

        for i, modality in enumerate(self.query_modal_types):
            x_proj.append(self.modality_proj[modality](outs_dec[i]))

        # select k query from each modal queries
        cls_scores = outs['cls_logits'][-1].sigmoid()
        cls_scores = list(cls_scores.split(num_queries_per_modality, dim=1))
        center = outs["center"][-1]
        center = list(center.split(num_queries_per_modality, dim=1))
        reference = list(reference.split(num_queries_per_modality, dim=1))
        center_top, x_proj_top, ref_top, mq_idx_top = [], [], [], []

        for i, modality in enumerate(self.query_modal_types):
            center_t, x_proj_t, ref_t, q_idx = self.select_top_queries(cls_scores[i], center[i], x_proj[i], reference[i], num_queries_per_modality, pad_size)
            if self.modal_embedding:
                x_proj_t = x_proj_t + self.modal_embed[modality].view(1, 1, -1)
            center_top.append(center_t)
            x_proj_top.append(x_proj_t)
            ref_top.append(ref_t)
            mq_idx_top.append(q_idx)

        target = torch.cat(x_proj_top, dim=0)
        center = torch.cat(center_top, dim=1)
        reference = torch.cat(ref_top, dim=1)
        mq_idx = torch.cat(mq_idx_top, dim=1)
        target, center = self.permute_dn(target, 0), self.permute_dn(center, 1)
        reference, mq_idx = self.permute_dn(reference, 1), self.permute_dn(mq_idx, 1)

        # TODO: when 'img', 'bev' are added, _bev_query_embed and _rv_query embed required
        box_pos_embed = pos2embed(center, self.embed_dims)
        query_box_pos_embed = self.box_pos_embedding(box_pos_embed).transpose(0, 1)

        attn_masks_in = [None]
        if self.use_self_attn:
            center_q = center.clone()
            center_kv = center.clone()
            dist = (center_q.unsqueeze(2) - center_kv.unsqueeze(1)).norm(p=2, dim=-1)
            batch_size, num_queries, _ = dist.shape
            num_modalities = 3
            queries_per_modality = num_queries // num_modalities
            
            # # # 마스크 초기화 (모든 값을 0으로 설정)
            # mask = torch.eye(num_queries,dtype=torch.float).unsqueeze(0).expand(batch_size,-1,-1).cuda()
            
            # for batch in range(batch_size):
            #     for src_query in range(num_queries):
            #         src_modality = src_query//queries_per_modality
            #         for tgt_modality in range(num_modalities):
            #             if src_modality != tgt_modality:
            #                 tgt_start = tgt_modality * queries_per_modality
            #                 tgt_end = (tgt_modality + 1) * queries_per_modality
            #                 distances = dist[batch, src_query, tgt_start:tgt_end]
            #                 nearest_center = distances.argmin().item() + tgt_start
            #                 # 마스크 업데이트
            #                 mask[batch, src_query, nearest_center] = 1.0
            # mask_clone = mask.clone()

            # 모달리티 인덱스 및 마스크 생성
            src_modality = torch.arange(num_queries).unsqueeze(0).cuda() // queries_per_modality
            tgt_modality = torch.arange(num_modalities).unsqueeze(0).cuda()
            diff_modality = (src_modality.unsqueeze(-1) != tgt_modality).expand(batch_size, -1, -1)
            # 가장 가까운 센터 찾기
            nearest_centers = dist.view(batch_size, num_queries, num_modalities, -1).argmin(dim=-1)
            nearest_centers += tgt_modality * queries_per_modality
            # 마스크 생성 및 업데이트
            dist_mask = torch.zeros_like(dist).cuda()
            dist_mask[torch.arange(batch_size).unsqueeze(-1).unsqueeze(-1),torch.arange(num_queries).unsqueeze(0).unsqueeze(-1),nearest_centers] = diff_modality.float()
            # 대각선 요소 설정
            dist_mask.diagonal(dim1=1, dim2=2).fill_(1.0)
            dist_mask = ~dist_mask.bool() # if float mask then mask is added
            # attn_masks = torch.zeros((target.shape[0], target.shape[0]), dtype=torch.bool, device=target.device)
            # attn_masks = torch.zeros_like(attn_masks, dtype=torch.float).float().masked_fill(attn_masks, float("-inf"))
            # attn_masks = attn_masks + dist_mask
            attn_masks = dist_mask
            attn_masks = attn_masks.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            attn_masks_in = [attn_masks, None]

        if self.modal_embedding:
            for idx, modal in enumerate(['fused', 'bev', 'img']):
                ca_dict['memory_l'][idx] = ca_dict['memory_l'][idx] + self.modal_embed[modal].view(1, 1, -1)
                ca_dict['memory_v_l'][idx] = ca_dict['memory_l'][idx] + self.modal_embed[modal].view(1, 1, -1)
        if self.failure_pred:
            target, weight_list = self.decoder(
                query=target,
                key=ca_dict['memory_l'],
                value=ca_dict['memory_v_l'],
                query_pos=query_box_pos_embed,
                key_pos=ca_dict['pos_embed_l'],
                attn_masks=attn_masks_in,
                reg_branch=reg_branch,
                modal_seq=self.modal_seq,
                failure_pred=self.failure_pred
            )
        else:
            target = self.decoder(
                query=target,
                key=ca_dict['memory_l'],
                value=ca_dict['memory_v_l'],
                query_pos=query_box_pos_embed,
                key_pos=ca_dict['pos_embed_l'],
                attn_masks=attn_masks_in,
                reg_branch=reg_branch,
                modal_seq=self.modal_seq,
                failure_pred=self.failure_pred
            )
            weight_list = None

        target = target.transpose(1, 2)
        outs = self.task_heads[task_id](target)

        center = (outs['center'] + reference[None, :, :, :2]).sigmoid()
        height = (outs['height'] + reference[None, :, :, 2:3]).sigmoid()
        _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
        _center[..., 0:1] = center[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        _center[..., 1:2] = center[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        _height[..., 0:1] = height[..., 0:1] * (pc_range[5] - pc_range[2]) + pc_range[2]
        outs['center'] = _center
        outs['height'] = _height

        return outs, mq_idx, weight_list
