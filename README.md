# Smart Personal Protective Equipment System

智慧工安系統：使用 YOLO 偵測工程帽 (Hard Hat)，
當發現未佩戴安全帽時，會觸發 LED / 蜂鳴器警示，並透過 Flask 後端記錄違規事件、
產生統計圖表，以及推播 LINE 通知。偵測端支援 CUDA kernel 加速（`cuda_kernels` + `cuda_runtime`），
可在 Raspberry Pi 5（CPU）/ 筆電 GPU 直接跑，並提供無自走車版 demo。

系統分成兩個主要部分：

1. **偵測端 (Raspberry Pi 5 / 筆電 GPU + 自走車可選)**  
   - USB 攝影機 + YOLO 模型偵測工程帽  
   - CUDA 前處理/後處理 kernel（pybind11 `cuda_lib`）加速 resize/heatmap（GPU 平台適用）  
   - 馬達控制（WASD 鍵控制小車前進/後退/轉彎，可選）  
   - LED / 蜂鳴器 / 按鈕 警示系統  
   - 偵測到違規事件時，自動呼叫 Flask `/api/events` 上報（含截圖，並可推 LINE）

2. **後端伺服器 (Flask Web + API)**  
   - `/api/events`：接收上報並寫入 SQLite / CSV  
   - `/`：今日違規次數 + 最新事件  
   - `/events`：違規事件列表 + 截圖  
   - `/stats`：統計頁面（今日各時段違規數、歷史每日違規數，支援縮放/平移）  
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
    cuda_runtime.py      # 封裝 cuda_lib 載入、GPU 前處理/後處理、YOLO 推論
    cuda_demo.py         # 舊入口（alias）：等同 `helmet_cam.py --mode cuda`
    car_main.py          # 小車手動控制（W/A/S/D）
    helmet_cam.py        # 鏡頭偵測（CPU/CUDA）+ 事件上報 + LED/蜂鳴器
    manual_control.py    # 只做馬達手動控制（WASD），不含偵測
    motor_controller.py  # 馬達控制 (L298N + DC Motors, 使用 BCM 腳位)
    hardware.py          # LED / Buzzer / Button 控制（RPi.GPIO / rpi-lgpio）
    ppe_detector.py      # YOLO 工程帽偵測 + 截圖 + 呼叫 /api/events
    config.py            # SERVER_URL, MODEL_PATH, IMG_SAVE_DIR, GPIO 腳位等
    models/
      best.pt            # 訓練好的 YOLO 安全帽模型（唯一使用的模型放這裡）
  cuda_kernels/          # CUDA 前/後處理模組（單一路徑集中 build，不再放模型檔）
    CMakeLists.txt       # 以 pybind11 建出 Python 模組 cuda_lib
    src/                 # preprocess/postprocess CUDA 與綁定 (pybind11 -> cuda_lib)
    tests/               # CUDA 單元測試 (C++)
    build/               # CMake 產物 (預期放置 cuda_lib.so/pyd)
    benchmark.py         # Python 端跑 cuda_lib.preprocess 效能測試
    test_run.py          # 簡易連續推論前處理迴圈
    tools.py             # fix/test/benchmark 統一入口（檢查 kernel/效能）


## CUDA kernel 編譯與檢查（可選但建議）

CUDA 路徑會去 `cuda_kernels/build` 或 `cuda_kernels/build/Release` 尋找 `cuda_lib`，
若沒有先編譯，`detector` 的 CUDA demo 會載入失敗。

1) 編譯 `cuda_lib`

```bash
cd cuda_kernels
mkdir -p build && cd build
cmake ..
cmake --build . --parallel
```

Windows 可加：`cmake --build . --config Release --parallel`

2) 用 `tools.py` 檢查/效能測試

```bash
cd cuda_kernels
python tools.py test --width 1920 --height 1080 --dst 640 --count 100
python tools.py benchmark
python tools.py fix
```

3) 也可用偵測端內建檢查

```bash
cd detector
python helmet_cam.py --diagnose
```

或：`python cuda_demo.py --diagnose`

## 環境設定

`detector/config.py` 設定連線/路徑/腳位：
- `SERVER_URL`：偵測端上報的 Flask 位址（預設為 `http://127.0.0.1:5001`，可用環境變數覆寫）
- `MODEL_PATH`：YOLO 模型路徑（預設 `AIoT/models/best.pt`，可用環境變數覆寫）
- `IMG_SAVE_DIR`：違規截圖存放資料夾（可用環境變數覆寫）
- GPIO 腳位：`RED_LED_PIN / GREEN_LED_PIN / BUZZER_PIN / BUTTON_PIN`

`server/app.py` 預設跑在 `5001`，若沒有改程式，請把 `SERVER_URL` 調成 `http://127.0.0.1:5001`。
若要 LINE 圖片顯示完整網址，需額外設定 `BASE_URL`（例如 ngrok 網址）。

```bash
cd AIoT
python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Raspberry Pi 5 建議先用系統套件裝 GPIO / OpenCV（避免 pip 編譯）：
- `sudo apt-get install python3-opencv python3-rpi.gpio`
- 若 RPi.GPIO 無法使用，可改裝 `rpi-lgpio`（提供 RPi.GPIO 相容 API）
- 建議用 venv 時加 `--system-site-packages`，可讀到系統 OpenCV

## Python 3.12 注意事項

- `ultralytics` 只在 Python >= 3.12 安裝（`requirements.txt` 已加條件）。
- 請先準備 YOLOv5 repo（本機資料夾）並設定環境變數：
  - `export YOLOV5_REPO=/path/to/yolov5`
  - 或直接放在 `AIoT/yolov5`
- 需自行安裝對應版本的 PyTorch（Raspberry Pi 5 建議使用官方 CPU wheel，需支援 Python 3.12）。

## 當天 Demo 操作流程（一步一步）

當天建議開 3 個 terminal：`server`、`car_main.py`（控車）、`helmet_cam.py`（偵測）。

1) 啟動虛擬環境與依賴（首次）

```bash
cd AIoT
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) 啟動後端 Flask + LINE 通知

```bash
cd server
export LINE_CHANNEL_ACCESS_TOKEN=...
export LINE_USER_ID=...
export BASE_URL=https://xxxx.ngrok-free.app
python app.py
```

後台登入帳密：`admin / admin123`

3) 準備模型與路徑
- 預設模型：`detector/models/best.pt`，或在 `detector/config.py` 的 `MODEL_PATH` 改成你的實際路徑。
- 違規截圖輸出目錄：`config.IMG_SAVE_DIR`（若在筆電/桌機，可改成 repo 相對路徑 `server/static/violations` 的絕對路徑），請確保目錄存在且 Flask 靜態檔路徑一致。

4) 檢查 CUDA 模組（可選，GPU 平台）

```bash
cd detector
python helmet_cam.py --diagnose
```

或：`python cuda_demo.py --diagnose`

5) 偵測端（鏡頭 + 事件 + LED/蜂鳴器）

GPU 平台：

```bash
python helmet_cam.py --mode cuda --source 0
```

參數：`--source` 攝影機索引或影片路徑；`--unsafe-threshold` 連續 unsafe 秒數才算違規（預設 10 秒）。
當畫面連續判定 unsafe：會觸發紅燈/蜂鳴器（若 GPIO 可用）、存截圖、POST 到 Flask，若有 LINE env 則推播。

6) 偵測端（CPU 版，不需 CUDA，Raspberry Pi 5 建議）

```bash
python helmet_cam.py --mode cpu
```

固定用攝影機 0，安全帽偵測 + 事件上報 + LED/蜂鳴器。

7) 小車控制（WASD）

```bash
python car_main.py
```

控制鍵：`w` 前進、`s` 後退、`a` 左轉、`d` 右轉、`space` 停止、`q` 離開。

8) 手動馬達測試（無偵測）

```bash
python manual_control.py
```

W/A/S/D 控制小車，Space 停止，Q 離開。

9) 預期畫面與觀察點
- CUDA 模式會顯示原始畫面 + heatmap 疊圖，CPU 模式會疊狀態文字。
- 終端會印出違規事件觸發，Flask 後台事件列表應同步新增；若 LINE 設好，會收到文字/圖片通知。
