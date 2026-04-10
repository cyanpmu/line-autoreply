"""
파우더브로우 LINE 자동답장 서버 v5
- 1:1 채팅: Q&A 매칭 + Claude API 폴백 + 이미지 분석
- 그룹 채팅: 양방향 자동 번역 (일본어↔한국어) + 이미지 분석
"""

import os
import json
import hashlib
import hmac
import base64
import time
import re
import httpx
from flask import Flask, request, abort
from qa_engine import find_best_qa_match, get_system_prompt
from image_analyzer import analyze_image_bytes, format_line_message

app = Flask(__name__)

LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
IGNORE_NAMES = set(filter(None, os.environ.get("IGNORE_NAMES", "").split(",")))

response_cache = {}
CACHE_TTL = 3600


# ═══════════════════════════════════════
# LINE API 유틸
# ═══════════════════════════════════════

def verify_signature(body, signature):
    hash_val = hmac.new(LINE_CHANNEL_SECRET.encode(), body, hashlib.sha256).digest()
    return signature == base64.b64encode(hash_val).decode()


def reply_message(reply_token, messages):
    try:
        httpx.post(
            "https://api.line.me/v2/bot/message/reply",
            headers={"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}", "Content-Type": "application/json"},
            json={"replyToken": reply_token, "messages": messages[:5]},
            timeout=30,
        )
    except Exception as e:
        print(f"Reply error: {e}")


def get_line_image(message_id):
    try:
        resp = httpx.get(
            f"https://api-data.line.me/v2/bot/message/{message_id}/content",
            headers={"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.content
    except Exception as e:
        print(f"Image download error: {e}")
    return None


def get_user_name(user_id):
    try:
        resp = httpx.get(
            f"https://api.line.me/v2/bot/profile/{user_id}",
            headers={"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("displayName", "")
    except:
        pass
    return ""


def get_group_member_name(group_id, user_id):
    try:
        resp = httpx.get(
            f"https://api.line.me/v2/bot/group/{group_id}/member/{user_id}/profile",
            headers={"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("displayName", "")
    except:
        pass
    return ""


# ═══════════════════════════════════════
# 언어 감지 + 번역
# ═══════════════════════════════════════

def detect_language(text):
    korean = len(re.findall(r'[\uac00-\ud7af\u3131-\u3163\u1100-\u11ff]', text))
    japanese = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
    if korean > 0 and korean >= japanese:
        return "ko"
    elif japanese > 0:
        return "ja"
    return "unknown"


def translate_text(text, from_lang, to_lang):
    if not CLAUDE_API_KEY:
        return None

    instructions = {
        ("ko", "ja"): "韓国語を自然な日本語に翻訳。パウダーブロウ専門用語はそのまま。翻訳だけ出力。",
        ("ja", "ko"): "日本語を自然な韓国語に翻訳。パウダーブロウ専門用語はそのまま。翻訳だけ出力。",
    }
    instruction = instructions.get((from_lang, to_lang))
    if not instruction:
        return None

    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": CLAUDE_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 500,
                  "messages": [{"role": "user", "content": f"{instruction}\n\n{text}"}]},
            timeout=20,
        )
        if resp.status_code == 200:
            return resp.json()["content"][0]["text"]
    except Exception as e:
        print(f"Translate error: {e}")
    return None


# ═══════════════════════════════════════
# 웹훅
# ═══════════════════════════════════════

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
        source_type = event["source"]["type"]
        msg = event["message"]

        if source_type in ("group", "room"):
            group_id = event["source"].get("groupId") or event["source"].get("roomId", "")
            if msg["type"] == "text":
                handle_group_text(reply_token, group_id, user_id, msg["text"])
            elif msg["type"] == "image":
                handle_group_image(reply_token, group_id, user_id, msg)

        elif source_type == "user":
            if msg["type"] == "image":
                handle_image(reply_token, msg)
            elif msg["type"] == "text":
                if detect_language(msg["text"]) == "ko":
                    continue
                handle_text(reply_token, user_id, msg["text"])

    return "OK"


# ═══════════════════════════════════════
# 그룹: 자동 번역 + 이미지 분석
# ═══════════════════════════════════════

def handle_group_text(reply_token, group_id, user_id, text):
    if len(text.strip()) < 3:
        return

    sender_name = get_group_member_name(group_id, user_id)
    is_staff = sender_name in IGNORE_NAMES
    lang = detect_language(text)

    if lang == "ko":
        translated = translate_text(text, "ko", "ja")
        if translated:
            reply_message(reply_token, [{"type": "text", "text": f"🇯🇵 {translated}"}])
    elif lang == "ja" and not is_staff:
        translated = translate_text(text, "ja", "ko")
        if translated:
            reply_message(reply_token, [{"type": "text", "text": f"🇰🇷 {translated}"}])


def handle_group_image(reply_token, group_id, user_id, msg):
    image_bytes = get_line_image(msg["id"])
    if not image_bytes:
        return

    result = analyze_image_bytes(image_bytes)
    if "error" in result:
        return

    if not result.get("results"):
        return

    # 일본어 피드백
    ja_msg = format_line_message(result, "ja")
    messages = [{"type": "text", "text": ja_msg}]

    # 선생님용 한국어 요약
    first = result["results"][0]
    score = first["analysis"]["score"]
    prob_areas = [f["area"] for f in first["analysis"]["feedback"] if f["type"] in ("critical", "warning")]
    ko = f"🇰🇷 [분석] {score}점"
    if prob_areas:
        ko += f" | {', '.join(prob_areas[:5])}"
    messages.append({"type": "text", "text": ko})

    reply_message(reply_token, messages)


# ═══════════════════════════════════════
# 1:1: Q&A + 이미지
# ═══════════════════════════════════════

def handle_image(reply_token, msg):
    image_bytes = get_line_image(msg["id"])
    if not image_bytes:
        reply_message(reply_token, [{"type": "text", "text": "画像の取得に失敗しました。もう一度送ってください🙏"}])
        return

    result = analyze_image_bytes(image_bytes)

    if "error" in result:
        reply_message(reply_token, [{"type": "text", "text": result["error"]}])
        return

    if not result.get("results"):
        reply_message(reply_token, [{"type": "text", "text": "眉毛を検出できませんでした。もう少し鮮明な写真で再度お送りください🙏"}])
        return

    ja_msg = format_line_message(result, "ja")
    reply_message(reply_token, [{"type": "text", "text": ja_msg}])


def handle_text(reply_token, user_id, text):
    if text.strip().lower() in ("myid", "id"):
        reply_message(reply_token, [{"type": "text", "text": f"あなたのUser ID:\n{user_id}"}])
        return

    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in response_cache:
        cached = response_cache[cache_key]
        if time.time() - cached["time"] < CACHE_TTL:
            reply_message(reply_token, [{"type": "text", "text": cached["response"]}])
            return

    user_name = get_user_name(user_id)
    name_prefix = f"{user_name}さん、" if user_name else ""

    matched = find_best_qa_match(text)
    if matched:
        resp = name_prefix + matched
        response_cache[cache_key] = {"response": resp, "time": time.time()}
        reply_message(reply_token, [{"type": "text", "text": resp}])
        return

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
            headers={"x-api-key": CLAUDE_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "system": get_system_prompt(),
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
