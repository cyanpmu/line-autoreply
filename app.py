# -*- coding: utf-8 -*-
"""
파우더브로우 LINE 자동답장 서버
LINE Messaging API + Claude API + Q&A 매칭
"""

import os
import json
import hashlib
import hmac
import base64
from datetime import datetime
from pathlib import Path

from flask import Flask, request, abort
import httpx

from qa_engine import get_best_reply, match_question

app = Flask(__name__)

# ── 환경변수 ──
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
REPLY_LANG = os.environ.get("REPLY_LANG", "ja")  # ja 또는 ko
MIN_MATCH_SCORE = int(os.environ.get("MIN_MATCH_SCORE", "8"))

# ── 캐시 ──
CACHE_DIR = Path(__file__).parent / "data"
CACHE_DIR.mkdir(exist_ok=True)
TEXT_CACHE = CACHE_DIR / "text_cache.json"
IMAGE_CACHE = CACHE_DIR / "image_cache.json"

# ── Claude 시스템 프롬프트 ──
SYSTEM_PROMPT = """あなたは「シアン先生」のアシスタントです。パウダーブロウ（粉黛眉）アートメイクスクールで学生の質問に日本語で答えます。

ルール：
1. 温かく励ましながら正確な技術指導
2. シアン先生の教え方に従う：
   - 薄い明暗を重ねて完成させる（力で調節しない）
   - オレンジゾーン（中央の一番濃い部分）からグラデーション
   - マシンは最大限90度維持、前方に傾け❌、自分に向かって10度OK
   - デザインラインを含めてタッチ
   - 1・2・3法則で点と点をつなげる
3. 絵文字を適度に使って親しみやすく
4. 回答は日本語で"""

IMAGE_PROMPT = """この画像はパウダーブロウの練習作品です。分析してフィードバック：
1. グラデーション：オレンジゾーンから眉頭・眉尻への明暗の流れ
2. デザインライン：下段の鮮明さと連続性
3. 均一性：点の間隔、深さのムラ
4. 左右対称性
5. 全体的な濃さのバランス
良い点を先に褒めて、改善点を具体的に。最後に「次はこうしてみて」とアドバイス。"""


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ── 캐시 관리 ──
def load_cache(path):
    if path.exists():
        try:
            return json.loads(path.read_text("utf-8"))
        except:
            pass
    return {}

def save_cache(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


# ── LINE API ──
def verify_signature(body, signature):
    """LINE 서명 검증"""
    h = hmac.new(LINE_CHANNEL_SECRET.encode(), body, hashlib.sha256).digest()
    return hmac.compare_digest(signature, base64.b64encode(h).decode())


def reply_to_line(reply_token, text):
    """LINE에 답장 전송"""
    # 2000자 제한 처리
    if len(text) > 2000:
        text = text[:1997] + "..."

    try:
        r = httpx.post(
            "https://api.line.me/v2/bot/message/reply",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
            },
            json={
                "replyToken": reply_token,
                "messages": [{"type": "text", "text": text}],
            },
            timeout=10,
        )
        log(f"  LINE 답장 전송: {r.status_code}")
    except Exception as e:
        log(f"  LINE 답장 실패: {e}")


def get_image_from_line(message_id):
    """LINE에서 이미지 다운로드"""
    try:
        r = httpx.get(
            f"https://api-data.line.me/v2/bot/message/{message_id}/content",
            headers={"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"},
            timeout=30,
        )
        if r.status_code == 200:
            return r.content
    except Exception as e:
        log(f"  이미지 다운로드 실패: {e}")
    return None


# ── Claude API ──
def ask_claude(question):
    """텍스트 질문 → Claude 답변 (캐시 있으면 캐시)"""
    cache = load_cache(TEXT_CACHE)
    key = hashlib.md5(question.strip().lower().encode()).hexdigest()

    if key in cache:
        log(f"  💾 텍스트 캐시 히트!")
        return cache[key]["answer"]

    if not CLAUDE_API_KEY:
        return None

    try:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": question}],
            },
            timeout=30,
        )
        r.raise_for_status()
        answer = r.json()["content"][0]["text"]

        cache[key] = {"question": question, "answer": answer, "ts": datetime.now().isoformat()}
        save_cache(TEXT_CACHE, cache)
        log(f"  🤖 Claude 답변 생성 + 캐시 저장")
        return answer

    except Exception as e:
        log(f"  Claude API 에러: {e}")
        return None


def analyze_image_with_claude(image_bytes):
    """이미지 → Claude Vision 분석 (캐시 있으면 캐시)"""
    cache = load_cache(IMAGE_CACHE)
    key = hashlib.sha256(image_bytes).hexdigest()[:16]

    if key in cache:
        log(f"  💾 이미지 캐시 히트!")
        return cache[key]["feedback"]

    if not CLAUDE_API_KEY:
        return None

    b64 = base64.b64encode(image_bytes).decode()

    try:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1500,
                "system": SYSTEM_PROMPT,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": IMAGE_PROMPT},
                    ],
                }],
            },
            timeout=60,
        )
        r.raise_for_status()
        feedback = r.json()["content"][0]["text"]

        cache[key] = {"feedback": feedback, "ts": datetime.now().isoformat()}
        save_cache(IMAGE_CACHE, cache)
        log(f"  🔍 이미지 분석 완료 + 캐시 저장 (총 {len(cache)}건)")
        return feedback

    except Exception as e:
        log(f"  이미지 분석 에러: {e}")
        return None


# ── 메시지 처리 ──
def handle_text(reply_token, text, user_name=""):
    """텍스트 메시지 처리"""
    log(f"  💬 텍스트: \"{text[:50]}\" (from {user_name})")

    # 1단계: Q&A 매칭 (무료)
    answer, score = get_best_reply(text, REPLY_LANG)
    if answer and score >= MIN_MATCH_SCORE:
        log(f"  🎯 Q&A 매칭! (점수: {score})")
        reply_to_line(reply_token, answer)
        return

    # 2단계: Claude API (유료, 캐시)
    answer = ask_claude(text)
    if answer:
        reply_to_line(reply_token, answer)
        return

    # 3단계: 둘 다 안 되면
    log(f"  ⚠️ 답변 불가")


def handle_image(reply_token, message_id, user_name=""):
    """이미지 메시지 처리"""
    log(f"  📷 이미지 수신 (from {user_name})")

    image_bytes = get_image_from_line(message_id)
    if not image_bytes:
        reply_to_line(reply_token, "画像を受け取りましたが、読み込めませんでした。もう一度送っていただけますか？🙇")
        return

    feedback = analyze_image_with_claude(image_bytes)
    if feedback:
        reply_to_line(reply_token, feedback)
    else:
        reply_to_line(reply_token, "画像を確認しました！先生に詳しいフィードバックをお願いしますね。少々お待ちください🙇")


# ── Flask 라우트 ──
@app.route("/", methods=["GET"])
def index():
    stats_img = len(load_cache(IMAGE_CACHE))
    stats_txt = len(load_cache(TEXT_CACHE))
    return f"""
    <h2>✨ 파우더브로우 LINE 자동답장 서버</h2>
    <p>상태: 🟢 작동 중</p>
    <p>Q&A: 20개 | 이미지캐시: {stats_img}건 | 텍스트캐시: {stats_txt}건</p>
    <p>LINE: {"✅ 연결됨" if LINE_CHANNEL_ACCESS_TOKEN else "❌ 토큰 없음"}</p>
    <p>Claude: {"✅ 연결됨" if CLAUDE_API_KEY else "❌ 키 없음"}</p>
    """


@app.route("/webhook", methods=["POST"])
def webhook():
    """LINE Webhook 수신"""
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data()

    if LINE_CHANNEL_SECRET and not verify_signature(body, signature):
        log("❌ 서명 검증 실패")
        abort(403)

    data = json.loads(body)

    for event in data.get("events", []):
        if event.get("type") != "message":
            continue

        reply_token = event["replyToken"]
        msg = event["message"]
        msg_type = msg.get("type")

        # 보낸 사람 이름 (가능하면)
        user_id = event.get("source", {}).get("userId", "unknown")
        user_name = user_id[:8]

        log(f"\n{'='*40}")
        log(f"📩 새 메시지 (type: {msg_type})")

        if msg_type == "text":
            handle_text(reply_token, msg["text"], user_name)
        elif msg_type == "image":
            handle_image(reply_token, msg["id"], user_name)
        else:
            log(f"  ⏭ 지원 안 하는 타입: {msg_type}")

    return "OK", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log("=" * 50)
    log("✨ 파우더브로우 LINE 자동답장 서버 시작!")
    log(f"  포트: {port}")
    log(f"  LINE: {'✅' if LINE_CHANNEL_ACCESS_TOKEN else '❌'}")
    log(f"  Claude: {'✅' if CLAUDE_API_KEY else '❌'}")
    log("=" * 50)
    app.run(host="0.0.0.0", port=port)
