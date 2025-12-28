import sys
import os
import time
import numpy as np

# 1. 設定路徑 (跟之前一樣，確保找得到 pyd)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "build"))
sys.path.append(os.path.join(current_dir, "build", "Release"))

try:
    import cuda_lib
    print(f"✅ 模組載入成功: {cuda_lib.__file__}")
except ImportError:
    print("❌ 找不到 cuda_lib 模組，請檢查編譯是否成功")
    sys.exit(1)

def main():
    # 模擬 1080p 輸入影像
    H, W = 1080, 1920
    img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    
    print(f"測試影像: {W}x{H} -> 640x640")
    
    # --- 階段 1: 熱身 (Warm-up) ---
    # GPU 第一次執行都需要初始化，時間會比較久，這是正常的
    print("🔥 正在熱身 GPU (Warm-up)...")
    for _ in range(10):
        _ = cuda_lib.preprocess(img, 640, 640)
    
    # --- 階段 2: 效能測試 (Benchmark) ---
    test_count = 1000
    print(f"🚀 開始執行 {test_count} 次極速測試...")
    
    start_time = time.time()
    for _ in range(test_count):
        # 這就是之後你要接 YOLO 前真正會跑的函式
        result = cuda_lib.preprocess(img, 640, 640)
    end_time = time.time()
    
    # --- 計算結果 ---
    total_time = end_time - start_time
    avg_time = (total_time / test_count) * 1000 # 轉成毫秒
    fps = test_count / total_time
    
    print("\n" + "="*40)
    print(f" 測試結果 (RTX 3070)")
    print(f"========================================")
    print(f" 平均延遲: {avg_time:.4f} ms")
    print(f" 處理速度: {fps:.2f} FPS")
    print(f"========================================")
    print(" (這僅包含 Host->Device, Resize, Normalize, Device->Host)")

if __name__ == "__main__":
    main()
