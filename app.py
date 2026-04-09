"""
파우더브로우 LINE 자동답장 서버 v4
- Q&A 매칭 (무료) + Claude API 폴백 (~3원)
- 이미지 분석: OpenCV 실측 비교 (무료)
"""

import os
import json
import hashlib
import hmac
import base64
import time
import httpx
from flask import Flask, request, abort

from image_analyzer import analyze_student_brow

app = Flask(__name__)

LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")

response_cache = {}
CACHE_TTL = 3600


def verify_signature(body, signature):
    hash_val = hmac.new(LINE_CHANNEL_SECRET.encode(), body, hashlib.sha256).digest()
    return signature == base64.b64encode(hash_val).decode()


def reply_message(reply_token, messages):
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    body = {"replyToken": reply_token, "messages": messages[:5]}
    try:
        httpx.post(url, headers=headers, json=body, timeout=30)
    except Exception as e:
        print(f"Reply error: {e}")


def get_line_image(message_id):
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    try:
        resp = httpx.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.content
    except Exception as e:
        print(f"Image download error: {e}")
    return None


def get_user_name(user_id):
    url = f"https://api.line.me/v2/bot/profile/{user_id}"
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    try:
        resp = httpx.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("displayName", "")
    except:
        pass
    return ""


@app.route("/webhook", methods=["POST"])
def webhook():
    body = request.get_data()
    signature = request.headers.get("X-Line-Signature", "")

    if LINE_CHANNEL_SECRET and not verify_signature(body, signature):
        abort(403)

    data = json.loads(body)

    for event in data.get("events", []):
        if event["type"] != "message":
            continue

        reply_token = event["replyToken"]
        user_id = event["source"].get("userId", "")
        msg = event["message"]

        if msg["type"] == "image":
            handle_image(reply_token, user_id, msg)
        elif msg["type"] == "text":
            handle_text(reply_token, user_id, msg["text"])

    return "OK"


def handle_image(reply_token, user_id, msg):
    """학생 이미지 → OpenCV 분석 → 피드백 전송"""
    image_bytes = get_line_image(msg["id"])
    if not image_bytes:
        reply_message(reply_token, [{
            "type": "text",
            "text": "画像の取得に失敗しました。もう一度送ってください🙏"
        }])
        return

    # OpenCV 분석 (무료)
    result = analyze_student_brow(image_bytes)

    if "error" in result:
        reply_message(reply_token, [{
            "type": "text",
            "text": result["error"]
        }])
        return

    # 학생에게 일본어 피드백 전송
    reply_message(reply_token, [{
        "type": "text",
        "text": result["message_ja"],
    }])


def handle_text(reply_token, user_id, text):
    """텍스트 Q&A 처리"""
    # 캐시
    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in response_cache:
        cached = response_cache[cache_key]
        if time.time() - cached["time"] < CACHE_TTL:
            reply_message(reply_token, [{"type": "text", "text": cached["response"]}])
            return

    # 이름 호출
    user_name = get_user_name(user_id)
    name_prefix = f"{user_name}さん、" if user_name else ""

    # Q&A 매칭
    try:
        from qa_engine import find_best_qa_match
        matched = find_best_qa_match(text)
        if matched:
            resp = name_prefix + matched
            response_cache[cache_key] = {"response": resp, "time": time.time()}
            reply_message(reply_token, [{"type": "text", "text": resp}])
            return
    except ImportError:
        pass

    # Claude API 폴백 (~3원)
    if CLAUDE_API_KEY:
        claude_resp = call_claude(text)
        if claude_resp:
            resp = name_prefix + claude_resp
            response_cache[cache_key] = {"response": resp, "time": time.time()}
            reply_message(reply_token, [{"type": "text", "text": resp}])
            return

    reply_message(reply_token, [{
        "type": "text",
        "text": f"{name_prefix}ご質問ありがとうございます！先生に確認して改めてお返事しますね☺️"
    }])


def call_claude(text):
    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "system": (
                    "あなたはパウダーブロウ（半永久眉毛）の専門講師です。"
                    "11歳でも理解できる親切な日本語で答えてください。"
                    "授業で使う用語：眉頭、眉上（オレンジゾーン）、眉尾、レイヤリング、"
                    "グラデーション、デザインライン、ブレンディング。"
                ),
                "messages": [{"role": "user", "content": text}],
            },
            timeout=25,
        )
        if resp.status_code == 200:
            return resp.json()["content"][0]["text"]
    except Exception as e:
        print(f"Claude error: {e}")
    return None


@app.route("/health", methods=["GET"])
def health():
    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
