# Single GPU
# CUDA_VISIBLE_DEVICES=0 python -m scripts.train --config config/libero_train_config.yaml

# Multi-GPU (Set distributed=true in config)
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=5 --master_port=29505 -m scripts.train --config config/libero_train_config.yaml

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 --master_port=29505 -m scripts.train --config /data/public/NAS/VLANeXt/config/sim_train_spatial_config.yaml

# Single GPU
# CUDA_VISIBLE_DEVICES=0 python -m scripts.train --config config/droid_train_config.yaml

# Multi-GPU (Set distributed=true in config)
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=29505 -m scripts.train --config config/droid_train_config.yaml
