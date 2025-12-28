# Smart PPE Autocar 🚧🚗

智慧工安自走車系統：使用 YOLO 偵測工程帽 (Hard Hat)，
當發現未佩戴安全帽時，自走車上會觸發 LED 與蜂鳴器警示，
並且透過 Flask 後端記錄違規事件、產生統計圖表，以及推播 LINE 通知。

系統分成兩個主要部分：

1. **偵測端 (Raspberry Pi + 自走車)**  
   - USB 攝影機 + YOLO 模型偵測工程帽  
   - 馬達控制（WASD 鍵控制小車前進/後退/轉彎）  
   - LED / 蜂鳴器 / 按鈕 警示系統  
   - 偵測到違規事件時，自動呼叫 Flask `/api/events` 上報

2. **後端伺服器 (Flask Web + API)**  
   - `/api/events`：接收自走車上報的違規事件並寫入 SQLite / CSV  
   - `/Dashboard`：今日違規次數 + 最新事件）  
   - `/events`：違規事件列表 + 截圖  
   - `/stats`：統計頁面，包含：
     - 今日各時段違規次數
     - 歷史每日違規次數圖，支援縮放 / 平移（Chart.js + chartjs-plugin-zoom）  
   - `/download_csv`：下載 CSV 報表  
   - 會員系統（Flask-Login）：管理員登入後可新增 / 刪除 / 清空事件  
   - 整合 LINE Messaging API 推播違規事件（文字＋截圖）

---

## 專案結構

```text
AIoT/
  README.md
  requirements.txt
  data/
    violations.db        # SQLite 資料庫（後端啟動時自動建立）
    violations.csv       # 事件 CSV（新增/刪除/清空時自動同步）
  server/
    app.py               # Flask Web + API + 會員系統 + 統計圖
    line_notify.py       # 封裝 LINE Messaging API 推播
    templates/
      base.html          # 共用 layout + navbar (Login/Logout)
      dashboard.html     # 首頁 Dashboard
      events.html        # 事件列表（含管理按鈕）
      stats.html         # 統計頁，含 Chart.js + zoom plugin
      login.html         # 登入頁
      event_new.html     # 管理員手動新增事件表單
    static/
      css/style.css      # 前端樣式
      violations/        # 存放違規截圖（由偵測端寫入）
  detector/
    __init__.py          # 將 detector 標記為 Python package 
    car_main.py          # 主程式：WASD 控車 + YOLO 偵測 + 上報（單一流程，可關閉手動控制）
    motor_controller.py  # 馬達控制 (L298N + DC Motors, 使用 BCM 腳位)
    hardware.py          # LED / Buzzer / Button 控制（RPi.GPIO）
    ppe_detector.py      # YOLO 工程帽偵測 + 截圖 + 呼叫 /api/events
    config.py            # SERVER_URL, MODEL_PATH, IMG_SAVE_DIR, GPIO 腳位等
    models/
      best.pt            # 訓練好的 YOLO 安全帽模型（唯一使用的模型放這裡）
  cuda_kernels/          # CUDA 前/後處理模組（單一路徑集中 build，不再放模型檔）
    CMakeLists.txt       # 以 pybind11 建出 Python 模組 cuda_lib
    src/                 # preprocess/postprocess CUDA 與綁定
    tests/               # CUDA 單元測試 (C++)
    build/               # CMake 產物 (保留一份，其他重複 build 已移除)
    benchmark.py         # Python 端跑 cuda_lib.preprocess 效能測試
    test_run.py          # 簡易連續推論前處理迴圈
    tools.py             # fix/test/benchmark 統一入口


config.py設定腳位
cd AIoT
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

Server side
cd server
python app.py

admin / admin123

Raspberry PI
cd detector
python car_main.py

## 偵測模式與手動控制

- 檢查 CUDA 模組 (只檢查後退出)：  
  `python car_main.py --diagnose`

- 偵測 + 手動控制 (預設同時開)：  
  `python car_main.py --source 0`  
  `--source` 可用攝影機索引或影片路徑，`--model` 可覆寫模型路徑（預設 config.MODEL_PATH，找不到會自動 fallback 到 `detector/models/best.pt`）。

- WASD 操控（與偵測同時運行）：`w` 前進、`s` 後退、`a` 左轉、`d` 右轉、`space` 停止、`q` 離開。若只想關閉手動控制，加入 `--no-manual`。
- 需終端支援即時按鍵輸入；GPIO 控制請在樹莓派環境執行。
