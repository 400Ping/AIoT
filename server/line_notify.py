# server/line_notify.py
import os
import requests

# 這個 token 用的是 LINE Messaging API 的 channel access token
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID")  # 你自己的 userId 或群組Id

def send_line_message(text, image_url=None):
    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_USER_ID:
        print("[WARN] LINE env vars not set, skip notify")
        return

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
    }

    messages = [{"type": "text", "text": text}]
    if image_url:
        messages.append(
            {
                "type": "image",
                "originalContentUrl": image_url,
                "previewImageUrl": image_url,
            }
        )

    body = {
        "to": LINE_USER_ID,
        "messages": messages,
    }

    resp = requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers=headers,
        json=body,
        timeout=3.0,
    )
    print("LINE status:", resp.status_code, resp.text)
