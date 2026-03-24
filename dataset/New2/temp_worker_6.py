
import sys
import os
import time

# Change to script directory
os.chdir('/data/public/NAS/VLANeXt/Sim')
sys.path.insert(0, '/data/public/NAS/VLANeXt/Sim')

# Import and modify Save_dataset
import Save_dataset

# Override settings
Save_dataset.SAVE_DIR = r'/data/public/NAS/VLANeXt/dataset/New2/worker_6'
Save_dataset.MAX_EPISODES = 500

# Run
if __name__ == "__main__":
    print(f"[Worker 6] Starting with MAX_EPISODES=500, SAVE_DIR=/data/public/NAS/VLANeXt/dataset/New2/worker_6")
    Save_dataset.main()
    print(f"[Worker 6] Main completed, waiting for async file saves...")
    time.sleep(5)  # 비동기 저장 스레드 완료 대기
    print(f"[Worker 6] All done!")
