import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.core.utils.center_utils import _transpose_and_gather_feat, \
  bbox3d_overlaps_iou, bbox3d_overlaps_giou, bbox3d_overlaps_diou
from mmdet3d.ops.iou3d_nms.iou3d_nms_utils import boxes_aligned_iou3d_gpu ,boxes_iou_bev
import matplotlib.pyplot as plt
class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    if mask.sum() == 0:
      return output.new_zeros((target.shape[-1]))
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float().unsqueeze(2)

    loss = F.l1_loss(pred*mask, target*mask, reduction='none')
    loss = loss / (mask.sum() + 1e-4)
    loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)
    return loss

class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self):
    super(FastFocalLoss, self).__init__()

  def forward(self, out, target, ind, mask, cat):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    mask = mask.float()
    gt = torch.pow(1 - target, 4)
    neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
    neg_loss = neg_loss.sum()

    pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos


class IouLoss(nn.Module):
  '''IouLoss loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  # cx, cy, cz, w, l, h, rot, vx, vy -> denormalized_pred, target
  def __init__(self):
    super(IouLoss, self).__init__()
  
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
        plt.savefig('tensor_comparison_scatter_plot_with_lines_train.png', dpi=300, bbox_inches='tight')
        print("그래프가 'tensor_comparison_scatter_plot_with_lines_train.png' 파일로 저장되었습니다.")
    
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
      plt.savefig('rank_comparison_plot_train.png', dpi=300, bbox_inches='tight')
      print("순위 비교 그래프가 'rank_comparison_plot_train.png' 파일로 저장되었습니다.")

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

  def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1-1e-4)
        return y

  def forward(self, iou_pred, box_pred, box_gt,num_total_pos):
    
    target = boxes_iou_bev(box_pred[:,:7],box_gt[:,:7])
    target = target.max(1)[0]
    iou_pred = self._sigmoid(iou_pred)
    #self.hist(iou_pred.squeeze(), target.squeeze())
    import pdb;pdb.set_trace()
    self.hist_coeff(iou_pred.squeeze(), target.squeeze())
    #target = (target-0.5)/0.5
    loss = F.smooth_l1_loss(iou_pred.squeeze(), target.squeeze(), reduction='mean')
    loss = 5* loss
    
    #loss = F.l1_loss(iou_pred.squeeze(), target.squeeze(), reduction='sum')
    #loss = loss / (num_total_pos + 1e-4)
    return loss

class IouRegLoss(nn.Module):
  '''Distance IoU loss for output boxes
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''

  def __init__(self, type="IoU"):
    super(IouRegLoss, self).__init__()

    if type == "IoU":
      self.bbox3d_iou_func = bbox3d_overlaps_iou
    elif type == "GIoU":
      self.bbox3d_iou_func = bbox3d_overlaps_giou
    elif type == "DIoU":
      self.bbox3d_iou_func = bbox3d_overlaps_diou
    else:
      raise NotImplementedError

  def forward(self, box_pred, box_gt):
    # if mask.sum() == 0:
    #   return box_pred.new_zeros((1))
    # mask = mask.bool()
    # pred_box = _transpose_and_gather_feat(box_pred, ind)
    iou = self.bbox3d_iou_func(box_pred[:,:7], box_gt[:,:7])
    loss = (1. - iou).sum() / (box_pred.size(0) + 1e-4)
    return loss