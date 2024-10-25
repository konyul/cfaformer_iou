
bash tools/dist_test.sh projects/configs/cfaformer_voxel0075_vov_1600x640_cbgs_smt_nq150_fp_feature.py work_dirs/fp_feature/20241018-152448/epoch_3.pth 4 --eval bbox
for failures in  'lidar_drop' 'camera_drop' 'limited_fov' 'object_failure' 'beam_reduction' 'occlusion' 
do
 bash tools/dist_test.sh projects/configs/cfa_failure_feature/cfaformer_voxel0075_vov_1600x640_cbgs_$failures.py work_dirs/fp_feature/20241018-152448/epoch_3.pth 4 --eval bbox
done



# bash tools/dist_train.sh projects/configs/moadformer_voxel0075_vov_1600x640_cbgs_smt_nq150_fp_feature_only_sel.py 4 --work-dir work_dirs/moad_fp_feature_mhattn_freeze --resume-from work_dirs/moad_fp_feature_mhattn_freeze/20241024-200819/epoch_1.pth

# bash tools/dist_test.sh projects/configs/moadformer_voxel0075_vov_1600x640_cbgs_smt_nq150_fp_feature_only_sel.py work_dirs/moad_fp_feature_mhflash_decoder_freeze_layer3/20241023-171104/epoch_3.pth 4 --eval bbox
# for failures in  'lidar_drop' 'camera_drop' 'limited_fov' 'object_failure' 'beam_reduction' 'occlusion' 
# do
#  bash tools/dist_test.sh projects/configs/moad_cfa_failure_feature/cfaformer_voxel0075_vov_1600x640_cbgs_$failures.py work_dirs/moad_fp_feature_mhflash_decoder_freeze_layer3/20241023-171104/epoch_3.pth 4 --eval bbox
# done


# bash tools/dist_test.sh projects/configs/cfaformer_voxel0075_vov_1600x640_cbgs_smt_nq150_fp_feature.py work_dirs/fp_feature_layer1/20241022-201327/epoch_4.pth 4 --eval bbox
# for failures in  'lidar_drop' 'camera_drop' 'limited_fov' 'object_failure' 'beam_reduction' 'occlusion' 
# do
#  bash tools/dist_test.sh projects/configs/cfa_failure_feature/cfaformer_voxel0075_vov_1600x640_cbgs_$failures.py work_dirs/fp_feature_layer1/20241022-201327/epoch_4.pth 4 --eval bbox
# done


