python run_parallel.py --script align --workers 25 --episodes 400 \
    --base-dir /home/najo/NAS/VLANeXt/dataset/fine_align/fine_align_00 --phantom-pos 0.0 0.0 --no-side-camera --cameras tool_camera
# python run_parallel.py --script align --workers 20 --episodes 50 \
#     --base-dir /data/public/NAS/VLANeXt/dataset/fine_align/fine_align_01 --phantom-pos 0.0 -0.1 --no-side-camera --cameras tool_camera
# python run_parallel.py --script align --workers 20 --episodes 50 \
#     --base-dir /data/public/NAS/VLANeXt/dataset/fine_align/fine_align_03 --phantom-pos 0.0 -0.3 --no-side-camera --cameras tool_camera
# python run_parallel.py --script align --workers 20 --episodes 50 \
#     --base-dir /data/public/NAS/VLANeXt/dataset/fine_align/fine_align_04 --phantom-pos 0.0 -0.4 --no-side-camera --cameras tool_camera

# python Save_dataset_align_only.py --save-dir /data/public/NAS/VLANeXt/Sim/test_xyz --perturb -30 30 -30

# python run_parallel.py --script align --workers 1 --episodes 1 \
#     --base-dir /data/public/NAS/VLANeXt/dataset/fine_align/fine_align_02 --phantom-pos 0.0 -0.2 --no-side-camera

# python run_parallel.py --script approach --workers 10 --episodes 10 \
#     --base-dir /data/public/NAS/VLANeXt/dataset/approach/approach_00 --phantom-pos 0.0 -0.0 --no-side-camera --no-insertion --hold-steps 25 --cameras top_camera
# python run_parallel.py --script approach --workers 1 --episodes 5 \
#     --base-dir /data/public/NAS/VLANeXt/dataset/approach/approach_01 --phantom-pos 0.0 -0.1 --no-side-camera --no-insertion --hold-steps 25 --cameras top_camera
# python run_parallel.py --script approach --workers 20 --episodes 250 \
#     --base-dir /data/public/NAS/VLANeXt/dataset/approach/approach_03 --phantom-pos 0.0 -0.3 --no-side-camera --no-insertion --hold-steps 25 --cameras top_camera
# python run_parallel.py --script approach --workers 1 --episodes 5 \
#     --base-dir /data/public/NAS/VLANeXt/dataset/approach/approach_04 --phantom-pos 0.0 -0.4 --no-side-camera --no-insertion --hold-steps 25 --cameras top_camera
 
# python run_parallel.py --script insertion --workers 20 --episodes 250 \
#     --base-dir /data/public/NAS/VLANeXt/dataset/insertion/insertion_00 \
#     --phantom-pos 0.0 0.0 --approach-offset 5 --approach-xy-offset 2 --no-side-camera --perturb-mm 2 --perturb-prob 0.25 --perturb-frames 20 --cameras tool_camera
# python run_parallel.py --script insertion --workers 20 --episodes 250 \
#     --base-dir /data/public/NAS/VLANeXt/dataset/insertion/insertion_01 \
#     --phantom-pos 0.0 -0.1 --approach-offset 5 --approach-xy-offset 2 --no-side-camera --perturb-mm 2 --perturb-prob 0.25 --perturb-frames 20 --cameras tool_camera
# python run_parallel.py --script insertion --workers 20 --episodes 250 \
#     --base-dir /data/public/NAS/VLANeXt/dataset/insertion/insertion_03 \
#     --phantom-pos 0.0 -0.3 --approach-offset 5 --approach-xy-offset 2 --no-side-camera --perturb-mm 2 --perturb-prob 0.25 --perturb-frames 20 --cameras tool_camera
# python run_parallel.py --script insertion --workers 20 --episodes 250 \
#     --base-dir /data/public/NAS/VLANeXt/dataset/insertion/insertion_04 \
#     --phantom-pos 0.0 -0.4 --approach-offset 5 --approach-xy-offset 2 --no-side-camera --perturb-mm 2 --perturb-prob 0.25 --perturb-frames 20 --cameras tool_camera