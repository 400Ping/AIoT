import os
import sys
import time
import numpy as np

# 確保可以載入 build 出來的 cuda_lib 模組
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "build"))
sys.path.append(os.path.join(CURRENT_DIR, "build", "Release"))

import cuda_lib


def main():
    H, W = 1080, 1920
    img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    
    # 1. Warm-up (先跑一次讓 GPU 熱身)
    print("正在熱身 GPU...")
    _ = cuda_lib.preprocess(img, 640, 640)
    
    # 2. 正式測試效能
    test_count = 1000
    print(f"開始測試 {test_count} 次迴圈...")
    
    start = time.time()
    for _ in range(test_count):
        # 這裡模擬連續推論的情境
        result = cuda_lib.preprocess(img, 640, 640)
    end = time.time()

    total_time = end - start
    avg_time = (total_time / test_count) * 1000 # ms
    fps = 1 / (total_time / test_count)

    print(f"========================================")
    print(f"平均時間: {avg_time:.4f} ms")
    print(f"推論速度: {fps:.2f} FPS (只含前處理)")
    print(f"========================================")
