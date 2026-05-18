# unset PYTHONPATH
# export PYTHONPATH=$PYTHONPATH:/home/najo/NAS/VLANeXt/third_party/LIBERO

# LIBERO eval
# CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python -m scripts.libero_bench_eval

# python scripts/sim_eval.py \
#     --config config/sim_eval_config.yaml \
#     --checkpoint checkpoints/VLANeXt_droid.pt

# Sim eval
# CUDA_VISIBLE_DEVICES=2 python -m scripts.sim_eval \
#     --config config/sim_eval_config.yaml \
#     --checkpoint /data/public/NAS/VLANeXt/output_dir_step_4000 \
#     --train-config config/sim_train_spatial_config.yaml

# sim_eval_align_only
CUDA_VISIBLE_DEVICES=0 python -m scripts.sim_eval_align_only \
      --config config/sim_eval_align_config.yaml \
      --checkpoint /data/public/NAS/VLANeXt/output_dir_align_0.8B \
      --train-config config/sim_train_align_config.yamlw

python scripts/analyze_eval.py /data/public/NAS/VLANeXt/align_eval_step0.8B_exec1_diff10/metrics_summary.csv

rsync -av --progress -e 'ssh -p 51322' /home/irom/NAS/VLANeXt/dataset/real_align najo@10.130.4.79:/home/najo/NAS/VLANeXt/dataset
