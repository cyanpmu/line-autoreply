"""
파우더브로우 LINE 자동답장 서버 v2
- Q&A 35개 매칭 + Claude API 폴백
- 🆕 이미지 분석: 5구역 수치 비교 + 문제 위치 표시 이미지 + 상세 피드백
"""

import os
import json
import hashlib
import time
import httpx
from flask import Flask, request, abort

from image_analyzer import (
    full_analysis, detect_pattern_type, REFERENCES
)

app = Flask(__name__)

LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")

# ── 캐시 ──
response_cache = {}
CACHE_TTL = 3600

# ── Q&A 엔진 (기존 qa_engine.py에서 import하거나 여기에 포함) ──
# from qa_engine import find_best_qa_match
# 여기서는 간략히 표시. 실제로는 qa_engine.py 그대로 사용.


def verify_signature(body, signature):
    """LINE webhook 서명 검증"""
    import hmac, hashlib, base64
    hash_val = hmac.new(
        LINE_CHANNEL_SECRET.encode(), body, hashlib.sha256
    ).digest()
    return signature == base64.b64encode(hash_val).decode()


def get_line_image(message_id):
    """LINE 서버에서 이미지 바이너리 다운로드"""
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    resp = httpx.get(url, headers=headers, timeout=30)
    if resp.status_code == 200:
        return resp.content
    return None


def upload_image_to_line(image_bytes):
    """
    분석 결과 이미지를 LINE으로 보내기 위해 임시 URL 생성
    방법 1: Render 서버에 임시 저장 후 public URL 제공
    방법 2: Imgur 등 외부 서비스 업로드
    여기서는 방법 1 (서버 내 임시 저장)
    """
    filename = f"analysis_{int(time.time())}.jpg"
    filepath = os.path.join("/tmp", filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    
    # Render 배포 시 서버 URL
    server_url = os.environ.get("RENDER_EXTERNAL_URL", "https://line-autoreply-zwvw.onrender.com")
    return f"{server_url}/static/analysis/{filename}", filepath


def reply_message(reply_token, messages):
    """LINE reply API - 텍스트 + 이미지 동시 전송 가능"""
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    body = {
        "replyToken": reply_token,
        "messages": messages[:5],  # LINE은 최대 5개 메시지
    }
    httpx.post(url, headers=headers, json=body, timeout=30)


def push_message(user_id, messages):
    """LINE push API - reply_token 없이 보내기"""
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    body = {
        "to": user_id,
        "messages": messages[:5],
    }
    httpx.post(url, headers=headers, json=body, timeout=30)


def claude_vision_analysis(image_bytes, student_data):
    """
    Claude Vision API로 정성적 분석 추가
    - 그라데이션 부드러움
    - 얼룩 여부
    - 전체 완성도 코멘트
    """
    if not CLAUDE_API_KEY:
        return None
    
    import base64
    img_b64 = base64.b64encode(image_bytes).decode()
    
    # 수치 데이터를 프롬프트에 포함
    zone_summary = "\n".join([
        f"Z{i+1} {z['zone_ko']}: 명암={z['student_dark']:.0f}(정답{z['ref_dark']}, 차이{z['diff_dark']:+.0f}), "
        f"밀도={z['student_dens']:.0f}%(정답{z['ref_dens']}%)"
        for i, z in enumerate(student_data["results"])
    ])
    
    breaks_text = ""
    if student_data.get("gradient_breaks"):
        breaks_text = "그라데이션 끊김 감지 위치:\n" + "\n".join([
            f"- {b['between_names_ko']}: 차이 {b['diff']:.0f}"
            for b in student_data["gradient_breaks"]
        ])
    
    prompt = f"""당신은 파우더브로우(반영구 눈썹) 전문 강사입니다.
학생이 제출한 연습 사진을 OpenCV로 분석한 수치 데이터가 있습니다.

【수치 분석 결과 (vs {student_data['ref_name']} 정답)】
{zone_summary}

{breaks_text}

이 사진을 직접 보고, 수치로는 잡히지 않는 다음 사항을 일본어로 간결하게 평가해주세요:

1. 그라데이션 부드러움: 구역 경계가 자연스럽게 이어지는지, 끊김이 보이는지
2. 얼룩/뭉침: 특정 부분이 뭉쳐 보이거나 불균일한 곳이 있는지
3. 앞머리(미두) 처리: "사라지듯" 연하게 마감되었는지
4. 꼬리(미미) 처리: 페이드아웃이 자연스러운지
5. 전체 형태: 디자인 라인이 깔끔한지

형식: 각 항목을 ✅(양호) 또는 💡(개선점)으로 시작하여 한 줄씩.
마지막에 종합 한 줄 코멘트.
11살도 이해할 수 있는 친절한 말투로."""

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
                "max_tokens": 600,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            },
            timeout=30,
        )
        
        if resp.status_code == 200:
            data = resp.json()
            return data["content"][0]["text"]
    except Exception as e:
        print(f"Claude Vision error: {e}")
    
    return None


# ── 이미지 서빙용 라우트 ──
@app.route("/static/analysis/<filename>")
def serve_analysis_image(filename):
    """분석 결과 이미지를 서빙"""
    from flask import send_from_directory
    return send_from_directory("/tmp", filename, mimetype="image/jpeg")


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
        
        # ── 이미지 메시지 처리 🆕 ──
def handle_image_message(reply_token, user_id, msg):
    message_id = msg["id"]
    
    image_bytes = get_line_image(message_id)
    if not image_bytes:
        reply_message(reply_token, [{
            "type": "text",
            "text": "画像の取得に失敗しました。もう一度送ってください🙏"
        }])
        return
    
    # "분석 중" 메시지 보내지 말고 바로 분석
    result = full_analysis(image_bytes, ref_name="SOFT")
    
    if "error" in result:
        reply_message(reply_token, [{
            "type": "text",
            "text": "分析エラーです。もう一度お送りください🙏"
        }])
        return
    
    detected = detect_pattern_type(result["student_zones"])
    if detected != "SOFT":
        result = full_analysis(image_bytes, ref_name=detected)
    
    # 이미지 없이 텍스트만 reply_token으로 전송
    reply_message(reply_token, [{
        "type": "text",
        "text": result["message_ja"],
    }])
        
        # ── 텍스트 메시지 처리 (기존 로직) ──
        elif msg["type"] == "text":
            handle_text_message(reply_token, user_id, msg["text"])
    
    return "OK"


def handle_image_message(reply_token, user_id, msg):
    """
    학생 이미지 수신 → 분석 → 표시 이미지 + 피드백 전송
    """
    message_id = msg["id"]
    
    # 1. 이미지 다운로드
    image_bytes = get_line_image(message_id)
    if not image_bytes:
        reply_message(reply_token, [{
            "type": "text",
            "text": "画像の取得に失敗しました。もう一度送ってください🙏"
        }])
        return
    
    # 2. 먼저 "분석 중" 안내
    reply_message(reply_token, [{
        "type": "text",
        "text": "📸 写真を受け取りました！\n分析中です...少々お待ちください🔍"
    }])
    
    # 3. OpenCV 분석 (패턴 자동 감지)
    # 기본은 SOFT로 분석, 추후 학생별 설정 가능
    result = full_analysis(image_bytes, ref_name="SOFT")
    
    if "error" in result:
        push_message(user_id, [{
            "type": "text",
            "text": f"分析エラー: {result['error']}\nもう一度お送りください🙏"
        }])
        return
    
    # 자동 패턴 감지로 재분석
    detected = detect_pattern_type(result["student_zones"])
    if detected != "SOFT":
        result = full_analysis(image_bytes, ref_name=detected)
    
    # 4. Claude Vision 정성 분석 (비용 ~20원)
    vision_comment = claude_vision_analysis(image_bytes, result)
    
    # 5. 분석 이미지 URL 생성
    img_url, _ = upload_image_to_line(result["annotated_image_bytes"])
    
    # 6. 메시지 조합
    messages = []
    
    # 표시 이미지
    messages.append({
        "type": "image",
        "originalContentUrl": img_url,
        "previewImageUrl": img_url,
    })
    
    # 수치 분석 메시지
    messages.append({
        "type": "text",
        "text": result["message_ja"],
    })
    
    # Claude Vision 정성 분석 (있으면)
    if vision_comment:
        messages.append({
            "type": "text",
            "text": f"🔍 詳細観察：\n{vision_comment}",
        })
    
    # push로 전송 (reply_token은 이미 사용됨)
    push_message(user_id, messages)


def handle_text_message(reply_token, user_id, text):
    """기존 텍스트 Q&A 처리 (qa_engine.py 사용)"""
    # 캐시 체크
    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in response_cache:
        cached = response_cache[cache_key]
        if time.time() - cached["time"] < CACHE_TTL:
            reply_message(reply_token, [{
                "type": "text",
                "text": cached["response"],
            }])
            return
    
    # Q&A 매칭 (qa_engine.py에서 import)
    try:
        from qa_engine import find_best_qa_match
        matched = find_best_qa_match(text)
        if matched:
            response_cache[cache_key] = {"response": matched, "time": time.time()}
            reply_message(reply_token, [{"type": "text", "text": matched}])
            return
    except ImportError:
        pass
    
    # Claude API 폴백
    if CLAUDE_API_KEY:
        claude_response = call_claude_text(text)
        if claude_response:
            response_cache[cache_key] = {"response": claude_response, "time": time.time()}
            reply_message(reply_token, [{"type": "text", "text": claude_response}])
            return
    
    # 기본 응답
    reply_message(reply_token, [{
        "type": "text",
        "text": "ご質問ありがとうございます！\n先生に確認して改めてお返事しますね☺️",
    }])


def call_claude_text(text):
    """Claude API로 텍스트 질문 답변"""
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
                "system": "당신은 파우더브로우(반영구 눈썹) 전문 강사입니다. 11살도 이해할 수 있는 친절한 일본어로 답변하세요.",
                "messages": [{"role": "user", "content": text}],
            },
            timeout=25,
        )
        if resp.status_code == 200:
            return resp.json()["content"][0]["text"]
    except Exception as e:
        print(f"Claude text error: {e}")
    return None


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
