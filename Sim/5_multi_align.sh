python run_parallel.py --script align --workers 20 --episodes 200 \
    --base-dir /data/public/NAS/VLANeXt/dataset/fine_align_00 --phantom-pos 0.0 0.0 --no-side-camera

python run_parallel.py --script align --workers 20 --episodes 200 \
    --base-dir /data/public/NAS/VLANeXt/dataset/fine_align_01 --phantom-pos 0.0 -0.1 --no-side-camera

# python run_parallel.py --script align --workers 20 --episodes 200 \
#     --base-dir /data/public/NAS/VLANeXt/dataset/fine_align_02 --phantom-pos 0.0 -0.2 --no-side-camera

python run_parallel.py --script align --workers 20 --episodes 200 \
    --base-dir /data/public/NAS/VLANeXt/dataset/fine_align_03 --phantom-pos 0.0 -0.3 --no-side-camera

python run_parallel.py --script align --workers 20 --episodes 200 \
    --base-dir /data/public/NAS/VLANeXt/dataset/fine_align_04 --phantom-pos 0.0 -0.4 --no-side-camera