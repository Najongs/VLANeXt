# unset PYTHONPATH
# export PYTHONPATH=$PYTHONPATH:/home/najo/NAS/VLANeXt/third_party/LIBERO

# LIBERO eval
# CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python -m scripts.libero_bench_eval

# python scripts/sim_eval.py \
#     --config config/sim_eval_config.yaml \
#     --checkpoint checkpoints/VLANeXt_droid.pt

# Sim eval
CUDA_VISIBLE_DEVICES=2 python -m scripts.sim_eval \
    --config config/sim_eval_config.yaml \
    --checkpoint /data/public/NAS/VLANeXt/output_dir_step_4000 \
    --train-config config/sim_train_spatial_config.yaml
