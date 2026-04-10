"""
파우더브로우 LINE 자동답장 서버 v6.1
- 1:1: 사진+이름/패턴 큐잉 (순서 무관, 10분 타임아웃)
- 1:1: Q&A 매칭 + Claude 폴백
- 그룹: 양방향 번역 + 이미지 분석
- v6.1: 유저 지정 패턴을 채점에 반영, 모델명 업데이트
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

# 유저별 임시 저장소 (사진/텍스트 큐잉)
pending_submissions = {}
PENDING_TTL = 600  # 10분

# ────────────────────────────────────────────
# 모델명 설정 (중앙 관리)
# ────────────────────────────────────────────
MODEL_SMART = "claude-sonnet-4-6"            # Q&A 폴백, 메인 응답
MODEL_FAST  = "claude-haiku-4-5-20251001"    # 번역 (짧은 텍스트, 저렴)


# ═══════════════════════════════════════
# LINE API
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


def push_message(user_id, messages):
    """reply_token 없이 보내기 (비동기 응답용)"""
    try:
        httpx.post(
            "https://api.line.me/v2/bot/message/push",
            headers={"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}", "Content-Type": "application/json"},
            json={"to": user_id, "messages": messages[:5]},
            timeout=30,
        )
    except Exception as e:
        print(f"Push error: {e}")


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
    except Exception:
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
    except Exception:
        pass
    return ""


# ═══════════════════════════════════════
# 언어 감지 + 번역
# ═══════════════════════════════════════

def detect_language(text):
    korean  = len(re.findall(r'[\uac00-\ud7af\u3131-\u3163\u1100-\u11ff]', text))
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
            json={
                "model": MODEL_FAST,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": f"{instruction}\n\n{text}"}],
            },
            timeout=20,
        )
        if resp.status_code == 200:
            return resp.json()["content"][0]["text"]
        else:
            print(f"Translate API error: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        print(f"Translate error: {e}")
    return None


# ═══════════════════════════════════════
# 유연한 텍스트 파싱
# ═══════════════════════════════════════

def parse_submission_text(text):
    """
    유연하게 파싱. 있는 정보만 추출. 없으면 None.
    "田中・SOFT" → {pattern: "SOFT", name: "田中"}
    "ソフト 3回目 眉頭むずい" → {pattern: "SOFT", practice: 3, difficulty: "眉頭むずい"}
    "こんにちは" → {pattern: None} (일반 질문)
    """
    info = {
        "name": None, "pattern": None, "technique": None,
        "layering": None, "needle": None, "practice": None,
        "difficulty": None, "improvement": None,
    }

    t = text.strip()

    # 패턴 (없으면 제출 아님)
    pattern_map = {
        "SOFT": "SOFT", "ソフト": "SOFT", "そふと": "SOFT",
        "NATURAL": "NATURAL", "ナチュラル": "NATURAL",
        "MIX": "MIX", "ミックス": "MIX", "ミクス": "MIX",
    }
    for key, val in pattern_map.items():
        if key.upper() in t.upper() or key in t:
            info["pattern"] = val
            break

    if not info["pattern"]:
        return info

    # 기법
    if re.search(r'テボリ|手彫り|てぼり|hand', t, re.IGNORECASE):
        info["technique"] = "テボリ"
    elif re.search(r'マシン|マシーン|machine', t, re.IGNORECASE):
        info["technique"] = "マシン"

    # 레이어링 횟수
    m = re.search(r'レイヤリング\s*(\d+)\s*回|(\d+)\s*回\s*レイヤ|(\d+)\s*回レイヤ', t)
    if m:
        info["layering"] = int(m.group(1) or m.group(2) or m.group(3))
    else:
        m = re.search(r'回数[：:]\s*(\d+)', t)
        if m:
            info["layering"] = int(m.group(1))

    # 니들
    needles = re.findall(r'\d+(?:RL|RS|P)\b', t, re.IGNORECASE)
    if needles:
        info["needle"] = " + ".join(needles)

    # 연습 횟수
    m = re.search(r'練習\s*(\d+)\s*回目|(\d+)\s*回目', t)
    if m:
        info["practice"] = int(m.group(1) or m.group(2))

    # 이름
    m = re.search(r'名前[：:]\s*(.+?)(?:\n|$|技法|パターン|レイヤ)', t)
    if m:
        info["name"] = m.group(1).strip()
    else:
        for key in pattern_map:
            if key.upper() in t.upper() or key in t:
                before = re.split(re.escape(key), t, flags=re.IGNORECASE)[0]
                before = re.sub(r'[・/\-\s【】課題提出]+', ' ', before).strip()
                if before and len(before) <= 20:
                    info["name"] = before
                break

    # 어려운 점
    m = re.search(r'難しかった点[：:]\s*(.+?)(?:\n|改善|$)', t, re.DOTALL)
    if m:
        info["difficulty"] = m.group(1).strip()
    else:
        m = re.search(r'(?:難し|むずかし|できな|苦手).*?(?:$|\n)', t)
        if m:
            info["difficulty"] = m.group(0).strip()

    # 개선된 점
    m = re.search(r'改善できた点[：:]\s*(.+?)(?:\n|$)', t, re.DOTALL)
    if m:
        info["improvement"] = m.group(1).strip()
    else:
        m = re.search(r'(?:改善|よくなった|できるようになった).*?(?:$|\n)', t)
        if m:
            info["improvement"] = m.group(0).strip()

    return info


def clean_pending(user_id):
    """만료된 pending 제거"""
    if user_id in pending_submissions:
        if time.time() - pending_submissions[user_id].get("time", 0) > PENDING_TTL:
            del pending_submissions[user_id]


def try_analyze(user_id):
    """사진 + 패턴 둘 다 있으면 분석 실행"""
    if user_id not in pending_submissions:
        return False

    p = pending_submissions[user_id]
    if not (p.get("photo") and p.get("pattern")):
        return False

    # ★ v6.1: 유저 지정 패턴을 analyze에 전달
    result = analyze_image_bytes(p["photo"], pattern=p.get("pattern"))

    if "error" in result:
        msg = result["error"]
    else:
        context = {
            "name":        p.get("name"),
            "pattern":     p.get("pattern"),
            "technique":   p.get("technique"),
            "layering":    p.get("layering"),
            "needle":      p.get("needle"),
            "practice":    p.get("practice"),
            "difficulty":  p.get("difficulty"),
            "improvement": p.get("improvement"),
        }
        msg = format_line_message(result, context)

    push_message(user_id, [{"type": "text", "text": msg}])
    del pending_submissions[user_id]
    return True


# ═══════════════════════════════════════
# 웹훅
# ═══════════════════════════════════════

@app.route("/webhook", methods=["POST"])
def webhook():
    body = request.get_data()
    signature = request.headers.get("X-Line-Signature", "")
    if LINE_CHANNEL_SECRET and not verify_signature(body, signature):
        abort(403)

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        abort(400)

    for event in data.get("events", []):
        if event.get("type") != "message":
            continue

        reply_token = event.get("replyToken", "")
        user_id = event["source"].get("userId", "")
        source_type = event["source"]["type"]
        msg = event["message"]

        if source_type in ("group", "room"):
            group_id = event["source"].get("groupId") or event["source"].get("roomId", "")
            if msg["type"] == "text":
                handle_group_text(reply_token, group_id, user_id, msg["text"])
            elif msg["type"] == "image":
                handle_group_image(reply_token, msg)
        elif source_type == "user":
            if msg["type"] == "image":
                handle_image(reply_token, user_id, msg)
            elif msg["type"] == "text":
                if detect_language(msg["text"]) == "ko":
                    continue  # 한국어(선생님) 무시
                handle_text(reply_token, user_id, msg["text"])

    return "OK"


# ═══════════════════════════════════════
# 그룹
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


def handle_group_image(reply_token, msg):
    image_bytes = get_line_image(msg["id"])
    if not image_bytes:
        return
    result = analyze_image_bytes(image_bytes)
    if "error" in result:
        return
    ja_msg = format_line_message(result)
    score = result["results"][0]["score"] if result.get("results") else "?"
    ko_summary = f"🇰🇷 [분석] {score}점"
    reply_message(reply_token, [
        {"type": "text", "text": ja_msg},
        {"type": "text", "text": ko_summary},
    ])


# ═══════════════════════════════════════
# 1:1: 큐잉 시스템
# ═══════════════════════════════════════

def handle_image(reply_token, user_id, msg):
    """사진 수신 → 큐에 저장"""
    image_bytes = get_line_image(msg["id"])
    if not image_bytes:
        reply_message(reply_token, [{"type": "text", "text": "画像の取得に失敗しました。もう一度送ってください🙏"}])
        return

    clean_pending(user_id)

    # 이미 패턴이 저장되어 있으면 → 바로 분석
    if user_id in pending_submissions and pending_submissions[user_id].get("pattern"):
        pending_submissions[user_id]["photo"] = image_bytes
        pending_submissions[user_id]["time"] = time.time()
        reply_message(reply_token, [{"type": "text", "text": "📸 写真を受け取りました！分析中です...少々お待ちください☺️"}])
        try_analyze(user_id)
        return

    # 패턴 없으면 → 사진만 저장하고 질문
    pending_submissions[user_id] = {
        "photo": image_bytes,
        "time": time.time(),
    }
    reply_message(reply_token, [{
        "type": "text",
        "text": "📸 写真を受け取りました！\n\nお名前とパターン名を教えてください☺️\n（例：田中・SOFT）\n（例：NATURAL 3回目）",
    }])


def handle_text(reply_token, user_id, text):
    # ID 확인 커맨드
    if text.strip().lower() in ("myid", "id"):
        reply_message(reply_token, [{"type": "text", "text": f"あなたのUser ID:\n{user_id}"}])
        return

    clean_pending(user_id)

    # 제출 텍스트인지 파싱
    info = parse_submission_text(text)

    if info["pattern"]:
        # 패턴 감지됨 → 제출로 처리
        if user_id not in pending_submissions:
            pending_submissions[user_id] = {"time": time.time()}

        for key, val in info.items():
            if val is not None:
                pending_submissions[user_id][key] = val
        pending_submissions[user_id]["time"] = time.time()

        # 사진이 이미 있으면 → 바로 분석
        if pending_submissions[user_id].get("photo"):
            reply_message(reply_token, [{"type": "text", "text": f"✅ {info['pattern']}で分析します。少々お待ちください☺️"}])
            try_analyze(user_id)
            return

        # 사진 없으면 → 요청
        name_msg = f"{info['name']}さん、" if info.get("name") else ""
        reply_message(reply_token, [{
            "type": "text",
            "text": f"{name_msg}✅ {info['pattern']}パターンですね！\n課題の写真を送ってください📸",
        }])
        return

    # 일반 Q&A
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

    reply_message(reply_token, [{"type": "text", "text": f"{name_prefix}ご質問ありがとうございます！先生に確認して改めてお返事しますね☺️"}])


def call_claude(text):
    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": CLAUDE_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={
                "model": MODEL_SMART,
                "max_tokens": 500,
                "system": get_system_prompt(),
                "messages": [{"role": "user", "content": text}],
            },
            timeout=25,
        )
        if resp.status_code == 200:
            return resp.json()["content"][0]["text"]
        else:
            print(f"Claude API error: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        print(f"Claude error: {e}")
    return None


@app.route("/health", methods=["GET"])
def health():
    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
