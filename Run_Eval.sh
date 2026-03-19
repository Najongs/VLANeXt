unset PYTHONPATH
export PYTHONPATH=$PYTHONPATH:~/proj/VLANeXt/third_party/LIBERO

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python -m scripts.libero_bench_eval