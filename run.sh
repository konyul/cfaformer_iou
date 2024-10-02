bash tools/dist_train.sh projects/configs/cfaformer_voxel0075_vov_1600x640_cbgs_smt_nq150_fp.py 4 --work-dir work_dirs/sota_fp

# for failures in  'lidar_drop' 'camera_drop' 'limited_fov' 'object_failure' 'beam_reduction' 'occlusion' 
# do
#   bash tools/dist_test.sh projects/configs/maq_failure/maqformer_voxel0075_vov_1600x640_cbgs_smt_$failures.py work_dirs/matransformer_smt_intraq_fixed_fused_smt/20240928-052742/epoch_2.pth 4 --eval bbox --out work_dirs/matransformer_smt_intraq_fixed_fused_smt/20240928-052742/$failures.pkl
# done