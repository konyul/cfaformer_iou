bash tools/dist_test.sh projects/configs/iou.py ckpts/0905_iou_pred_b4_lr_0.0001_delete_dn_smoothL1_target_std.pth 4 --eval bbox
bash ./tools/dist_test.sh ./projects/configs/cfaformer_voxel0075_vov_1600x640_cbgs_smt.py ckpts/cfaformer_voxel0075_vov_1600x640_cbgs_smt.pth 4 --eval bbox
