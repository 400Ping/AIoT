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
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

## 當天 Demo 操作流程（一步一步）

1) 啟動虛擬環境與依賴（首次）
- `cd AIoT`
- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

2) 啟動後端 Flask + LINE 通知
- 進入 server：`cd server`
- 設定 LINE env（若要推播）：`export LINE_CHANNEL_ACCESS_TOKEN=...`、`export LINE_USER_ID=...`
- 啟動：`python app.py`
- 後台登入帳密：`admin / admin123`

3) 準備模型與路徑
- 預設模型：`detector/models/best.pt`，或在 `detector/config.py` 的 `MODEL_PATH` 改成你的實際路徑。
- 違規截圖輸出目錄：`config.IMG_SAVE_DIR`（若在筆電/桌機，可改成 repo 相對路徑 `server/static/violations` 的絕對路徑），請確保目錄存在且 Flask 靜態檔路徑一致。

4) 檢查 CUDA 模組（可選）
- `cd detector`
- `python car_main.py --diagnose` 或 `python cuda_demo.py --diagnose`

5) Demo（不帶自走車，僅鏡頭 + 事件 + LED/蜂鳴器）
- 建議用筆電/Jetson：`python cuda_demo.py --source 0`
- 參數：`--source` 攝影機索引或影片路徑；`--unsafe-threshold` 連續 unsafe 秒數才算違規（預設 3 秒）。
- 當畫面連續判定 unsafe：會觸發紅燈/蜂鳴器（若 GPIO 可用）、存截圖、POST 到 Flask，若有 LINE env 則推播。

6) Demo（帶自走車 + WASD）
- `python car_main.py --source 0`
- `--no-manual` 可關閉手動控制；`--model` 可自訂模型路徑。
- 控制鍵：`w` 前進、`s` 後退、`a` 左轉、`d` 右轉、`space` 停止、`q` 離開。

7) 預期畫面與觀察點
- OpenCV 視窗顯示原始畫面 + heatmap 疊圖。
- 終端會印出違規事件觸發，Flask 後台事件列表應同步新增；若 LINE 設好，會收到文字/圖片通知。
