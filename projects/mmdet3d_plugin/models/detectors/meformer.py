# # ------------------------------------------------------------------------
# # Copyright (c) 2022 megvii-model. All Rights Reserved.
# # ------------------------------------------------------------------------
# # Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# # Copyright (c) OpenMMLab. All rights reserved.
# # ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin import SPConvVoxelization
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img
@DETECTORS.register_module()
class MEFormerDetector(MVXTwoStageDetector):
    def __init__(self,
                 use_grid_mask=False,
                 **kwargs):
        pts_voxel_cfg = kwargs.get('pts_voxel_layer', None)
        kwargs['pts_voxel_layer'] = None
        super(MEFormerDetector, self).__init__(**kwargs)

        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        if pts_voxel_cfg:
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg)
        self.count = 0

    def init_weights(self):
        """Initialize model weights."""
        super(MEFormerDetector, self).init_weights()

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img.float())
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    @force_fp32(apply_to=('pts', 'img_feats'))
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        if pts is None:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # vis
        img_metas[0]['img'] = img
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats or img_feats:
            losses_pts = self.forward_pts_train(
                pts_feats, img_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore
            )
            losses.update(losses_pts)
        return losses
    def visualize_feat(self, bev_feat, idx):
        feat = bev_feat.cpu().detach().numpy()
        min = feat.min()
        max = feat.max()
        image_features = (feat-min)/(max-min)
        image_features = (image_features*255)
        max_image_feature = np.max(np.transpose(image_features.astype("uint8"),(1,2,0)),axis=2)
        max_image_feature = cv2.applyColorMap(max_image_feature,cv2.COLORMAP_JET)
        cv2.imwrite(f"max_{idx}.jpg",max_image_feature)

    @force_fp32(apply_to=('pts_feats', 'img_feats'))
    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        if points is None:
            points = [None]
        if img is None:
            img = [None]
        for var, name in [(points, 'points'), (img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        return self.simple_test(points[0], img_metas[0], img[0], **kwargs)
        
    def visualize_img(self, gt_bboxes_3d, bbox_results, gt_labels, failure_case, img_metas, thickness=3):
        num_object = gt_bboxes_3d[0][0].tensor.shape[0]
        pred_boxes = bbox_results[0]['boxes_3d'][:num_object]
        scores = bbox_results[0]['scores_3d'][:num_object]
        gt_boxes = gt_bboxes_3d[0][0]
        gt_labels = gt_labels[0][0]
        labels = bbox_results[0]['labels_3d'][:num_object]
        def viz_view_img(img, std, mean, _matrices, img_metas, pred_boxes, gt_boxes, color=labels, idx=None, thickness=3):
            img = img.cpu().permute(1,2,0).numpy()
            denormalized_image = img * std + mean
            img = np.clip(denormalized_image, 0, 255).astype(np.uint8)
            # nuscenes colormap만 추가
            pred_img = draw_lidar_bbox3d_on_img(pred_boxes, img.copy(), _matrices, img_metas, color=labels, thickness=thickness)
            
            count = str(self.count).zfill(6) 
            cv2.imwrite(f"vis_img_{failure_case}/{count}_{idx}.png",pred_img)
            gt_img = draw_lidar_bbox3d_on_img(gt_boxes, img.copy(), _matrices, img_metas, color=gt_labels)
            # cv2.imwrite(f"vis_img/{count}_gt.png",gt_img)

        img = img_metas['img']
        _matrices = np.stack(img_metas['lidar2img'])
        _matrices = torch.tensor(_matrices).float()
        mean=[103.530, 116.280, 123.675]
        std=[57.375, 57.120, 58.395]
        mean = torch.tensor(mean).view(1, 1, 3).numpy()
        std = torch.tensor(std).view(1, 1, 3).numpy()
        cam_order=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_RIGHT','CAM_BACK_LEFT']
        for i in range(6):
            viz_view_img(img[i],std,mean,_matrices[i],img_metas, pred_boxes, gt_boxes, labels, cam_order[i], thickness=thickness)

    def visualize_bev(self, points, gt_bboxes_3d, bbox_results, gt_labels, failure_case):
        num_object = gt_bboxes_3d[0][0].tensor.shape[0]
        pred_boxes = bbox_results[0]['boxes_3d'].tensor[:num_object].cpu()
        scores = bbox_results[0]['scores_3d'][:num_object].cpu()
        gt_boxes = gt_bboxes_3d[0][0].tensor.cpu()
        gt_labels = gt_labels[0][0]
        labels = bbox_results[0]['labels_3d'][:num_object]
        points = points[0]
        point_cloud = points[points[:,-1]==0].cpu()
        
        def create_plot():
            fig, ax = plt.subplots(figsize=(15, 15))
            ax.set_xlim([-40, 40])
            ax.set_ylim([-60, 60])
            
            # Plot point cloud
            distances = np.sqrt(point_cloud[:, 0]**2 + point_cloud[:, 1]**2)
            max_dist = 60
            norm = plt.Normalize(0, max_dist)
            colors = plt.cm.RdYlBu_r(norm(distances))
            colors[:]=0
            ax.scatter(point_cloud[:, 0], point_cloud[:, 1], s=1, c=colors, alpha=1)
            
            return fig, ax
        
        def get_color(label_id):
            colormap = {
                0: 'darkorange',      # car
                1: 'coral',           # truck
                2: 'orange',          # cv
                3: 'sandybrown',      # bus
                4: 'peachpuff',       # trailer
                5: 'tan',             # barrier
                6: 'darkred',         # motorcycle
                7: 'red',             # bicycle
                8: 'blue',            # pedestrian
                9: 'yellow'           # traffic_cone
            }
            return colormap.get(int(label_id), 'green')
        
        # Visualize Ground Truth
        fig_gt, ax_gt = create_plot()
        ax_gt.set_title('Ground Truth', fontsize=16)
        
        for idx, (box, _label) in enumerate(zip(gt_boxes, gt_labels)):
            x, y, _, l, w, _, rot = box[:7].numpy()
            rot = -rot
            box_points = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2], [l/2, w/2]])
            rotation_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
            box_points = np.dot(box_points, rotation_matrix)
            box_points += np.array([x, y])
            
            color = get_color(_label)
            ax_gt.plot(box_points[:, 0], box_points[:, 1], c=color, linewidth=2)
        
        ax_gt.set_aspect('equal')
        plt.tight_layout()
        count = str(self.count).zfill(6)
        # plt.savefig(f"vis_pts/{count}_gt.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        # Visualize Predictions
        fig_pred, ax_pred = create_plot()
        ax_pred.set_title('Predictions', fontsize=16)
        
        def get_marker_style(idx):
            if idx < 10:  # Fusion
                return {'marker': 'o', 'markersize': 10}  
            elif idx < 20:  # LiDAR
                return {'marker': 's', 'markersize': 8}   
            else:  # Camera
                return {'marker': '^', 'markersize': 8}   
        
        # Add legend for prediction types
        ax_pred.plot([], [], c='indianred', linewidth=2, label='Fusion')
        ax_pred.plot([], [], c='indianred', linewidth=2, label='LiDAR')
        ax_pred.plot([], [], c='orange', linewidth=2, label='Camera')
        
        for idx, (box, score, label) in enumerate(zip(pred_boxes, scores, labels)):
            x, y, _, l, w, _, rot = box[:7].numpy()
            rot = -rot
            box_points = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2], [l/2, w/2]])
            rotation_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
            box_points = np.dot(box_points, rotation_matrix)
            box_points += np.array([x, y])
            
            color = get_color(label)
            marker_style = get_marker_style(idx)
            
            ax_pred.plot(box_points[:, 0], box_points[:, 1], c=color, linewidth=2)
        
        ax_pred.set_aspect('equal')
        ax_pred.legend(loc='upper right', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"vis_pts_{failure_case}/{count}.png", bbox_inches='tight', dpi=300)
        plt.close()
    @force_fp32(apply_to=('x', 'x_img'))
    def simple_test_pts(self, points, x, x_img, img_metas, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        if False:
            failure_case = 'lidar_drop'
            self.visualize_bev(points, gt_bboxes_3d, bbox_results, gt_labels_3d, failure_case)
            self.visualize_img(gt_bboxes_3d, bbox_results, gt_labels_3d, failure_case, img_metas[0])
        self.count+=1
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None):
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        img_metas[0]['img'] = img
        img_metas[0]['gt_bboxes_3d'] = gt_bboxes_3d
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        bbox_list = [dict() for i in range(len(img_metas))]
        if (pts_feats or img_feats) and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                points, pts_feats, img_feats, img_metas, rescale=rescale, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list


