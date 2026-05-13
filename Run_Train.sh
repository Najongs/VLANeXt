# Single GPU
# CUDA_VISIBLE_DEVICES=0 python -m scripts.train --config config/libero_train_config.yaml

# Multi-GPU (Set distributed=true in config)
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=5 --master_port=29505 -m scripts.train --config config/libero_train_config.yaml

# CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 --master_port=29505 -m scripts.train --config /data/public/NAS/VLANeXt/config/sim_train_spatial_config.yaml
# GPU 0 hardware fail + NVML probe로 multi-GPU 불가 + PCI remove도 막힘 → single-GPU 학습.
# 살아있는 GPU 1 사용. config에서 distributed/deepspeed false로 맞춰둠.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    python -m scripts.train --config config/sim_train_align_siglip2_config.yaml

# Single GPU
# CUDA_VISIBLE_DEVICES=0 python -m scripts.train --config config/droid_train_config.yaml

# Multi-GPU (Set distributed=true in config)
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=29505 -m scripts.train --config config/droid_train_config.yaml
