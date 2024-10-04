bash tools/dist_train.sh projects/configs/cfaformer_voxel0075_vov_1600x640_cbgs_smt_nq150_fp.py 4 --work-dir work_dirs/sota_fp_w_loss

# for failures in  'lidar_drop' 'camera_drop' 'limited_fov' 'object_failure' 'beam_reduction' 'occlusion' 
# do
#   bash tools/dist_test.sh projects/configs/cfa_failure/cfaformer_voxel0075_vov_1600x640_cbgs_$failures.py work_dirs/sota_fp_w_loss/20241004-1121627/epoch_6.pth 4 --eval bbox
# done
