# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import matplotlib.pyplot as plt
import torch
from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS

from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox

from mmcv.ops import nms as nms_mmcv
from mmdet3d.core.post_processing import nms_bev
from mmdet3d.core import xywhr2xyxyr
from mmdet3d.core.bbox.iou_calculators import bbox_overlaps_3d
from mmcv.ops import boxes_overlap_bev
from mmdet3d.ops.iou3d_nms.iou3d_nms_utils import boxes_iou_bev , nms_gpu , boxes_aligned_iou3d_gpu
# from mmdet3d.core.evaluation import bbox_overlaps
@BBOX_CODERS.register_module()
class MultiTaskBBoxCoder_iou(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass
    def hist(self, pred_loc, gt_loc):
        pred = pred_loc[gt_loc.nonzero().squeeze()]
        gt = gt_loc[gt_loc.nonzero().squeeze()]
        n = gt.shape[0]
        pred = pred.cpu()
        gt = gt.cpu()
        # 그래프 생성
        plt.figure(figsize=(12, 6))
        plt.scatter(range(n), pred.numpy().flatten(), label='Predicted', color='blue', alpha=0.6, s=30)
        plt.scatter(range(n), gt.numpy().flatten(), label='Ground Truth', color='red', alpha=0.6, s=30)

        # 같은 인덱스의 점들을 선으로 연결
        for i in range(n):
            plt.plot([i, i], [pred[i], gt[i]], color='gray', alpha=0.3, linestyle='--')

        plt.title('Prediction vs Ground Truth')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()

        # 그리드 추가
        plt.grid(True, linestyle='--', alpha=0.7)

        # x축 틱 설정 (10개 간격으로)
        plt.xticks(range(0, n, 10))

        # y축 범위 설정 (데이터의 최소값과 최대값을 기준으로)
        y_min = min(pred.min().item(), gt.min().item())
        y_max = max(pred.max().item(), gt.max().item())
        plt.ylim(y_min - 0.5, y_max + 0.5)

        # 그래프 레이아웃 조정
        plt.tight_layout()

        # 그래프를 이미지 파일로 저장
        plt.savefig('tensor_comparison_scatter_plot_with_lines2.png', dpi=300, bbox_inches='tight')
        print("그래프가 'tensor_comparison_scatter_plot_with_lines2.png' 파일로 저장되었습니다.")
    
    def hist_coeff(self, pred_loc, gt_loc):
        pred = pred_loc[gt_loc.nonzero().squeeze()]
        gt = gt_loc[gt_loc.nonzero().squeeze()]
        n = gt.shape[0]
        pred = pred.detach().cpu()
        gt = gt.detach().cpu()
        # pred와 gt의 순위 계산
        pred_ranks = torch.argsort(torch.argsort(pred)) + 1
        gt_ranks = torch.argsort(torch.argsort(gt)) + 1

        # 스피어만 상관계수 계산
        spearman_corr = torch.corrcoef(torch.stack((pred_ranks.float(), gt_ranks.float())))[0, 1].item()
        # 순위 차이 계산
        rank_diff = (pred_ranks - gt_ranks).abs()

        # 그래프 생성
        plt.figure(figsize=(12, 10))

        # 산점도: x축은 gt의 순위, y축은 pred의 순위
        plt.scatter(gt_ranks.numpy(), pred_ranks.numpy(), alpha=0.6)
        plt.plot([1, n], [1, n], color='r', linestyle='--')  # 이상적인 일치선

        plt.title(f'Rank Comparison: Predicted vs Ground Truth\nSpearman Correlation: {spearman_corr:.4f}')
        plt.xlabel('Ground Truth Rank')
        plt.ylabel('Predicted Rank')

        # 그래프 레이아웃 조정
        plt.tight_layout()

        # 그래프를 이미지 파일로 저장
        plt.savefig('rank_comparison_plot_test2.png', dpi=300, bbox_inches='tight')
        print("순위 비교 그래프가 'rank_comparison_plot2.png' 파일로 저장되었습니다.")

        # 분석 결과 출력
        print(f"스피어만 상관계수: {spearman_corr:.4f}")
        print(f"순위 차이의 평균: {rank_diff.float().mean().item():.2f}")
        print(f"순위 차이의 중앙값: {rank_diff.float().median().item():.2f}")
        print(f"순위 차이의 최대값: {rank_diff.max().item()}")

        # 순위 차이가 가장 큰 상위 5개 인덱스
        top_5_diff_indices = torch.argsort(rank_diff, descending=True)[:5]
        print("\n순위 차이가 가장 큰 상위 5개 인덱스:")
        for idx in top_5_diff_indices:
            print(f"인덱스 {idx.item()}: GT 순위 {gt_ranks[idx].item()}, Pred 순위 {pred_ranks[idx].item()}, 차이 {rank_diff[idx].item()}")

        # plt.show()  # 필요시 주석 해제하여 그래프를 화면에 표시

    def iou_nms(self, bboxes, scores, locs, iou_threshold):

        bev_bboxes = bboxes[:,:7]
        keep  = nms_gpu(bev_bboxes,locs.squeeze(),iou_threshold)
        keep_bboxes = bboxes[keep]

        overlaps = bbox_overlaps_3d(keep_bboxes, bboxes, coordinate='lidar')

        suppressed = overlaps > iou_threshold
        suppressed = suppressed.long().cummax(0)[0].cumsum(0)
        suppressed = (suppressed == 1)

        scores =  scores * locs
        span_scores = scores.view(1, overlaps.size(1)).repeat(overlaps.size(0), 1)
        span_scores[~suppressed] = scores.min() - 1

        keep_scores = span_scores.max(1)[0]
        keep_scores, srt_idx = keep_scores.sort(descending=True)

        return keep[srt_idx], keep_scores


    def batched_iou_nms(self, bboxes, scores, locs, labels, iou_threshold, gt_bboxes_3d, guide='none'):
        
        locs_extra = locs.squeeze()
        
        if len(gt_bboxes_3d[0][0]) == 0:
            locs = torch.zeros(locs.shape).cuda()
            locs = locs.squeeze(1)
        else:
            gt_bboxes_3d[0][0].tensor[:,2] = gt_bboxes_3d[0][0].tensor[:,2] + gt_bboxes_3d[0][0].tensor[:,5] * 0.5
            boxes_iou = boxes_iou_bev(bboxes[:,:7], gt_bboxes_3d[0][0].tensor[:,:7].cuda())            
            #boxes_3d_iou = boxes_aligned_iou3d_gpu(bboxes[:,:7], gt_bboxes_3d[0][0].tensor[:,:7].cuda())
            locs = boxes_iou.max(1)[0]

        # self.hist(locs_extra[:900], locs[:900])
        # self.hist_coeff(locs_extra[:900], locs[:900])
        nms_bboxes = bboxes

        if guide == 'rank':
             keep, keep_scores = self.iou_nms(nms_bboxes, scores, locs_extra, iou_threshold)
             return bboxes[keep], keep_scores, locs[keep], labels[keep]
        else:
            raise RuntimeError('guide type not supported: {}'.format(guide))

    def decode_single(self, cls_scores, bbox_preds, iou_pred, task_ids, bbox_targets_lists):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        
        max_num = self.max_num
        num_query = cls_scores.shape[0]
        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        task_index = torch.gather(task_ids, 1, labels.unsqueeze(1)).squeeze()
        iou_pred = iou_pred.sigmoid()
        # iou_pred = (iou_pred + 1) * 0.5
        bbox_preds = bbox_preds[task_index * num_query + bbox_index]
        iou_pred = iou_pred[task_index * num_query + bbox_index]
        boxes3d = denormalize_bbox(bbox_preds, self.pc_range)


        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)
            mask = (boxes3d[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (boxes3d[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask
            boxes3d = boxes3d[mask]
            scores = scores[mask]
            labels = labels[mask]
            iou_pred = iou_pred[mask]
            iou_preds = torch.clamp(iou_pred, min=0, max=1.) # for this ex, delete pls
            boxes3d, scores, iou_pred, labels = self.batched_iou_nms(boxes3d, scores, iou_pred, labels, 0.05, bbox_targets_lists, guide='rank')
            if len(boxes3d) > 300:
                boxes3d = boxes3d[:300]
                scores = scores[:300]
                labels = labels[:300]
        predictions_dict = {
            'bboxes': boxes3d,
            'scores': scores,
            'labels': labels
        }
        return predictions_dict

    def decode(self, preds_dicts, gt_bboxes_3d=None, gt_labels_3d=None):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        task_num = len(preds_dicts)
        
        pred_bbox_list, pred_logits_list, pred_iou_list, task_ids_list, rv_box_mask_lists = [], [], [], [], []
        for task_id in range(task_num):
            task_pred_dict = preds_dicts[task_id][0]
            task_pred_bbox = [task_pred_dict['center'][-1], task_pred_dict['height'][-1],
                              task_pred_dict['dim'][-1], task_pred_dict['rot'][-1]]
            if 'vel' in task_pred_dict:
                task_pred_bbox.append(task_pred_dict['vel'][-1])
            if 'iou' in task_pred_dict:
                task_pred_iou = task_pred_dict['iou'][-1]
                
            task_pred_bbox = torch.cat(task_pred_bbox, dim=-1)
            task_pred_logits = task_pred_dict['cls_logits'][-1]
            pred_bbox_list.append(task_pred_bbox)
            pred_logits_list.append(task_pred_logits)
            pred_iou_list.append(task_pred_iou)

            if "rv_box_mask" in task_pred_dict:
                rv_box_mask_lists.append(task_pred_dict["rv_box_mask"])
            else:
                rv_box_mask_lists.append(task_pred_dict["cls_logits"].new_ones(task_pred_dict["cls_logits"].shape[1], 6,
                                                                               task_pred_dict["cls_logits"].shape[
                                                                                   2]).to(torch.bool))

            task_ids = task_pred_logits.new_ones(task_pred_logits.shape).int() * task_id
            task_ids_list.append(task_ids)

        all_pred_logits = torch.cat(pred_logits_list, dim=-1)  # bs * nq * 10
        all_pred_bbox = torch.cat(pred_bbox_list, dim=1)  # bs * (task nq) * 10
        all_pred_iou = torch.cat(pred_iou_list,dim =-1)
        all_task_ids = torch.cat(task_ids_list, dim=-1)  # bs * nq * 10
        all_rv_box_masks = torch.cat(rv_box_mask_lists, dim=-1)
        
        batch_size = all_pred_logits.shape[0]
        predictions_list = []
        for i in range(batch_size):
            rv_box_mask = all_rv_box_masks[i].sum(dim=0) != 0
            if rv_box_mask.shape[0] != all_pred_bbox[i].shape[0]:
                box_mask = torch.cat([torch.ones_like(rv_box_mask), rv_box_mask])
            else:
                box_mask = rv_box_mask

            pred_logits = all_pred_logits[i][box_mask]
            pred_bbox = all_pred_bbox[i][box_mask]
            pred_iou = all_pred_iou[i][box_mask]
            task_ids = all_task_ids[i][box_mask]

            predictions_list.append(
                self.decode_single(pred_logits, pred_bbox, pred_iou, task_ids, gt_bboxes_3d))
        return predictions_list



@BBOX_CODERS.register_module()
class MultiTaskBBoxCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, task_ids):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num
        num_query = cls_scores.shape[0]

        cls_scores = cls_scores.sigmoid()
        if cls_scores.shape[0]<max_num:
            max_num = cls_scores.shape[0]
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        task_index = torch.gather(task_ids, 1, labels.unsqueeze(1)).squeeze()
        # scores, labels = cls_scores.max(-1)

        bbox_preds = bbox_preds[task_index * num_query + bbox_index]
        boxes3d = denormalize_bbox(bbox_preds, self.pc_range)

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)
            mask = (boxes3d[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (boxes3d[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = boxes3d[mask]
            scores = scores[mask]
            labels = labels[mask]

        predictions_dict = {
            'bboxes': boxes3d,
            'scores': scores,
            'labels': labels
        }
        return predictions_dict

    def decode(self, preds_dicts, gt_bboxes_3d=None, gt_labels_3d=None):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """

        task_num = len(preds_dicts)

        pred_bbox_list, pred_logits_list, task_ids_list, rv_box_mask_lists = [], [], [], []
        for task_id in range(task_num):
            task_pred_dict = preds_dicts[task_id][0]
            task_pred_bbox = [task_pred_dict['center'][-1], task_pred_dict['height'][-1],
                              task_pred_dict['dim'][-1], task_pred_dict['rot'][-1]]
            if 'vel' in task_pred_dict:
                task_pred_bbox.append(task_pred_dict['vel'][-1])
            task_pred_bbox = torch.cat(task_pred_bbox, dim=-1)
            task_pred_logits = task_pred_dict['cls_logits'][-1]
            pred_bbox_list.append(task_pred_bbox)
            pred_logits_list.append(task_pred_logits)

            if "rv_box_mask" in task_pred_dict:
                rv_box_mask_lists.append(task_pred_dict["rv_box_mask"])
            else:
                rv_box_mask_lists.append(task_pred_dict["cls_logits"].new_ones(task_pred_dict["cls_logits"].shape[1], 6,
                                                                               task_pred_dict["cls_logits"].shape[
                                                                                   2]).to(torch.bool))

            task_ids = task_pred_logits.new_ones(task_pred_logits.shape).int() * task_id
            task_ids_list.append(task_ids)

        all_pred_logits = torch.cat(pred_logits_list, dim=-1)  # bs * nq * 10
        all_pred_bbox = torch.cat(pred_bbox_list, dim=1)  # bs * (task nq) * 10
        all_task_ids = torch.cat(task_ids_list, dim=-1)  # bs * nq * 10
        all_rv_box_masks = torch.cat(rv_box_mask_lists, dim=-1)

        batch_size = all_pred_logits.shape[0]
        predictions_list = []
        for i in range(batch_size):
            rv_box_mask = all_rv_box_masks[i].sum(dim=0) != 0
            if rv_box_mask.shape[0] != all_pred_bbox[i].shape[0]:
                box_mask = torch.cat([torch.ones_like(rv_box_mask), rv_box_mask])
            else:
                box_mask = rv_box_mask

            pred_logits = all_pred_logits[i][box_mask]
            pred_bbox = all_pred_bbox[i][box_mask]
            task_ids = all_task_ids[i][box_mask]

            predictions_list.append(
                self.decode_single(pred_logits, pred_bbox, task_ids))
        return predictions_list
