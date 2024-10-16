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
import cv2
import matplotlib.pyplot as plt
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
            locality_aware_failure_pred=False,
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
        self.locality_aware_failure_pred = locality_aware_failure_pred

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True
        if self.modal_embedding:
            for modal in self.modal_embed.keys():
                normal_(self.modal_embed[modal])

    def select_top_queries(self, cls_scores, center, height, dim, x_proj, reference, num_queries_per_modality, pad_size):
        batch_size, num_queries, num_classes = cls_scores.shape

        center_dim = center.shape[2]
        x_proj = x_proj.permute(1, 0, 2) # -> [B, num_q, C]
        
        selected_center_list = []
        selected_height_list = []
        selected_dim_list = []
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
            selected_height_list.append(height[b][tk_ind])
            selected_dim_list.append(dim[b][tk_ind])
            selected_x_proj_list.append(x_proj[b][tk_ind])
            selected_reference_list.append(reference[b][tk_ind])
            selected_q_idx_list.append(tk_ind)

        selected_centers_tensor = torch.stack(selected_center_list)
        selected_height_tensor = torch.stack(selected_height_list)
        selected_dim_tensor = torch.stack(selected_dim_list)
        selected_x_proj_tensor = torch.stack(selected_x_proj_list)
        selected_reference_tensor = torch.stack(selected_reference_list)
        selected_q_idx_tensor = torch.stack(selected_q_idx_list)

        return selected_centers_tensor, selected_height_tensor, selected_dim_tensor, selected_x_proj_tensor.permute(1, 0, 2), selected_reference_tensor, selected_q_idx_tensor

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
        height = outs["height"][-1]
        height = list(height.split(num_queries_per_modality, dim=1))    
        dim = outs["dim"][-1]
        dim = list(dim.split(num_queries_per_modality, dim=1))    
        center = list(center.split(num_queries_per_modality, dim=1))
        reference = list(reference.split(num_queries_per_modality, dim=1))
        center_top, height_top, dim_top, x_proj_top, ref_top, mq_idx_top = [], [], [], [], [], []

        for i, modality in enumerate(self.query_modal_types):
            center_t, height_t, dim_t, x_proj_t, ref_t, q_idx = self.select_top_queries(cls_scores[i], center[i], height[i], dim[i], x_proj[i], reference[i], num_queries_per_modality, pad_size)
            if self.modal_embedding:
                x_proj_t = x_proj_t + self.modal_embed[modality].view(1, 1, -1)
            center_top.append(center_t)
            height_top.append(height_t)
            dim_top.append(dim_t)
            x_proj_top.append(x_proj_t)
            ref_top.append(ref_t)
            mq_idx_top.append(q_idx)

        target = torch.cat(x_proj_top, dim=0)
        center = torch.cat(center_top, dim=1)
        height = torch.cat(height_top, dim=1)
        dim = torch.cat(dim_top, dim=1)
        reference = torch.cat(ref_top, dim=1)
        mq_idx = torch.cat(mq_idx_top, dim=1)

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

            src_modality = torch.arange(num_queries).unsqueeze(0).cuda() // queries_per_modality # modality index 1,450
            tgt_modality = torch.arange(num_modalities).unsqueeze(0).cuda() # modality index 1,3
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
        
        if self.locality_aware_failure_pred:
            _position = torch.cat([center.clone(), height.clone()],dim=-1)
            _matrices = np.stack([np.stack(i['lidar2img']) for i in img_metas])
            _matrices = torch.tensor(_matrices).float().cuda()
            batch, _num_queries, _ =  _position.shape
            
            if False: # gt visualize
                _position = img_metas[0]['gt_bboxes_3d']._data.tensor[:,:3].unsqueeze(0).cuda()
                _position[:,:,-1] += img_metas[0]['gt_bboxes_3d']._data.tensor[:,5].unsqueeze(0).cuda()*0.5
                batch, _num_queries,_ = img_metas[0]['gt_bboxes_3d']._data.tensor[:,:3].unsqueeze(0).shape
                _matrices = _matrices[0:1]
            
            _position_4d = torch.cat([_position, torch.ones((batch, _num_queries, 1)).cuda()], dim=-1)
            
            # from mmdetection3d/mmdet3d/core/visualizer/image_vis.py
            pts_2d = torch.einsum('bni,bvij->bvnj', _position_4d, _matrices.transpose(2,3))
            pts_2d[..., 2] = torch.clip(pts_2d[..., 2], min=1e-5, max=99999)
            pts_2d[..., 0] /= pts_2d[..., 2]
            pts_2d[..., 1] /= pts_2d[..., 2]
            fov_inds = ((pts_2d[..., 0] < img_metas[0]['img_shape'][0][1])
                        & (pts_2d[..., 0] >= 0)
                        & (pts_2d[..., 1] < img_metas[0]['img_shape'][0][0])
                        & (pts_2d[..., 1] >= 0))
            
            _, num_views, _, _ = pts_2d.shape
            _fov_inds = torch.cat([torch.full((fov_inds.shape[0],1,fov_inds.shape[-1]),False).cuda(),fov_inds],dim=1)
            
            first_valid_view = _fov_inds.to(torch.float32).argmax(dim=1)
            first_valid_view[first_valid_view == 0] = -1
            first_valid_view[first_valid_view != -1] -= 1 
            batch_indices = torch.arange(batch).unsqueeze(1).expand(-1, _num_queries)
            point_indices = torch.arange(_num_queries).unsqueeze(0).expand(batch, -1)
            mask = first_valid_view != -1
            selected_pts = pts_2d[
                batch_indices[mask],
                first_valid_view[mask],
                point_indices[mask]
            ]
            pts_pers = torch.full((batch, _num_queries, 2), float('nan'), device=pts_2d.device)
            pts_pers[batch_indices[mask], point_indices[mask]] = selected_pts[...,[1,0]] # H,W
            ori_H,ori_W = img_metas[0]['img_shape'][0][:2]
            feat_H, feat_W = x_img.shape[2:]
            ratio = torch.tensor(feat_H/ori_H)
            pts_pers = torch.cat([first_valid_view.unsqueeze(-1),pts_pers],dim=-1)
            
            
            if False: # visualize
                def visualize_img(pts_pers, img_metas, cam_view=0, idx='0'):
                    imgfov_pts_2d = pts_pers[:1,:,:][pts_pers[:1,:,0]==cam_view][:,1:][:,[1,0]] # W, H
                    img = img_metas[0]['img'][0][cam_view].permute(1,2,0).cpu().numpy()
                    
                    cmap = plt.cm.get_cmap('hsv', 256)
                    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
                    mean=[103.530, 116.280, 123.675]
                    std=[57.375, 57.120, 58.395]
                    mean = torch.tensor(mean).view(1, 1, 3).numpy()
                    std = torch.tensor(std).view(1, 1, 3).numpy()
                    denormalized_image = img * std + mean
                    img = np.clip(denormalized_image, 0, 255).astype(np.uint8)
                    img = img.copy()
                    for i in range(imgfov_pts_2d.shape[0]):
                        # depth = imgfov_pts_2d[i, 2].item()
                        depth = 70
                        color = cmap[np.clip(int(70 * 10 / depth), 0, 255), :]
                        cv2.circle(
                            img,
                            center=(int(np.round(imgfov_pts_2d[i, 0].item())),
                                    int(np.round(imgfov_pts_2d[i, 1].item()))),
                            radius=1,
                            color=tuple(color),
                            thickness=3,
                        )
                    # 이미지 저장
                    output_path = f'projected_pts_img_{idx}.jpg'  # 저장할 경로와 파일명 지정
                    cv2.imwrite(output_path, img.astype(np.uint8))
                visualize_img(pts_pers, img_metas, 0, '0')
                visualize_img(pts_pers, img_metas, 1, '1')
                visualize_img(pts_pers, img_metas, 2, '2')
                visualize_img(pts_pers, img_metas, 3, '3')
                visualize_img(pts_pers, img_metas, 4, '4')
                visualize_img(pts_pers, img_metas, 5, '5')
        
            pts_pers[:,:,1:] = torch.floor(pts_pers[:,:,1:]*ratio)
            pts_pers[pts_pers[:,:,0]==-1] = 0.0
            pts_bev = torch.floor((center + 54.0) * (180 / 108))[:,:,[1,0]] # feature dim = y,x-> change center coordinates to y,x
            pts_idx = pts_bev[:,:,0]*180+pts_bev[:,:,1]
            
            # camera attn mask
            pts_pers_idx = pts_pers[:,:,0]*40*100+pts_pers[:,:,1]*100+pts_pers[:,:,2]
            batch_size, num_queries, _ = pts_pers.shape
            total_elements = 6*40*100
            H,W=40,100
            stride_view = 40*100
            stride_h = 100
            window_size = 15
            # 윈도우 오프셋 생성
            offsets = torch.arange(-(window_size // 2), window_size // 2 + 1, device=pts_pers_idx.device).cuda()
            window_offsets = offsets.unsqueeze(1) * stride_h + offsets.unsqueeze(0)
            window_offsets = window_offsets.view(-1)

            # 쿼리 인덱스에 윈도우 오프셋 적용
            indices = pts_pers_idx.unsqueeze(-1) + window_offsets.unsqueeze(0).unsqueeze(0)

            
            # 행과 열 인덱스 계산
            query_rows = (pts_pers_idx % (H * W)) // W
            query_cols = pts_pers_idx % W
            index_rows = (indices % (H * W)) // W
            index_cols = indices % W

            # 행과 열 유효성 체크
            valid_row = (index_rows - query_rows.unsqueeze(-1)).abs() <= window_size // 2
            valid_column = (index_cols - query_cols.unsqueeze(-1)).abs() <= window_size // 2

            # 전체 유효성 마스크
            valid = valid_row & valid_column

            # 범위를 벗어나는 인덱스 처리
            indices = torch.clamp(indices, 0, total_elements - 1).long()

            # Attention mask 생성 (모두 True로 초기화)
            img_attention_mask = torch.ones(batch_size, num_queries, total_elements, dtype=torch.bool, device=pts_pers_idx.device)
            batch_indices = torch.arange(batch_size, device=pts_pers_idx.device).long().view(-1, 1, 1)
            query_indices_range = torch.arange(num_queries, device=pts_pers_idx.device).long().view(1, -1, 1)

            valid_indices = valid.nonzero(as_tuple=True)
            img_attention_mask[
                batch_indices.expand_as(indices)[valid_indices[0], valid_indices[1], valid_indices[2]],
                query_indices_range.expand_as(indices)[valid_indices[0], valid_indices[1], valid_indices[2]],
                indices[valid_indices]
            ] = False
            
            # lidar attn mask
            batch_size, num_queries = pts_idx.shape
            total_elements = 180*180
            row_stride = 180
            window_size = 15
            offsets = torch.arange(-(window_size // 2), window_size // 2 + 1).cuda()
            y_offsets, x_offsets = torch.meshgrid(offsets, offsets)
            window_offsets = (y_offsets * row_stride + x_offsets).reshape(-1)
            # 각 쿼리에 대해 5x5 윈도우의 인덱스 계산
            indices = pts_idx.unsqueeze(-1) + window_offsets.unsqueeze(0).unsqueeze(0)
            # 유효한 인덱스만 유지 (0 이상 total_elements 미만)
            valid_indices = (indices >= 0) & (indices < total_elements)
            # 경계 체크: 행이 바뀌는 경우 제외
            query_columns = pts_idx % row_stride
            window_columns = (indices % row_stride).float() - query_columns.unsqueeze(-1).float()
            valid_indices &= (window_columns.abs() <= window_size // 2)
            # Attention mask 생성 (모두 True로 초기화)
            lidar_attention_mask = torch.ones(batch_size, num_queries, total_elements, dtype=torch.bool, device=pts_idx.device)

            batch_indices = torch.arange(batch_size, device=pts_idx.device).view(-1, 1, 1).expand_as(indices)
            query_indices_range = torch.arange(num_queries, device=pts_idx.device).view(1, -1, 1).expand_as(indices)

            # 유효한 인덱스만 사용하여 마스크 업데이트
            valid_mask = valid_indices & (indices < total_elements)
            lidar_attention_mask[
                batch_indices[valid_mask],
                query_indices_range[valid_mask],
                indices[valid_mask].long()
            ] = False
            fusion_attention_mask = torch.cat([lidar_attention_mask,img_attention_mask],dim=-1)
            attn_masks_in = [[fusion_attention_mask, lidar_attention_mask, img_attention_mask],None]
            
        if self.failure_pred or self.locality_aware_failure_pred:
            target, weight_list = self.decoder(
                query=target,
                key=ca_dict['memory_l'],
                value=ca_dict['memory_v_l'],
                query_pos=query_box_pos_embed,
                key_pos=ca_dict['pos_embed_l'],
                attn_masks=attn_masks_in,
                reg_branch=reg_branch,
                modal_seq=self.modal_seq,
                failure_pred=self.failure_pred,
                locality_aware_failure_pred=self.locality_aware_failure_pred
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
                failure_pred=self.failure_pred,
                locality_aware_failure_pred=self.locality_aware_failure_pred
            )
            weight_list = None
            
        target, center = self.permute_dn(target.squeeze(0), 0), self.permute_dn(center, 1)
        target = target.unsqueeze(0)
        reference, mq_idx = self.permute_dn(reference, 1), self.permute_dn(mq_idx, 1)
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
