# -*- coding: utf-8 -*-
"""
파우더브로우 LINE 챗봇 - 24시간 완전 자동화 (API 방식)
"""

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import threading

# 선생님이 기존에 만드신 Q&A 엔진 불러오기
from qa_engine import get_best_reply

app = Flask(__name__)

# ⚠️ 여기에 LINE Developers에서 발급받을 키를 넣어야 합니다.
CHANNEL_ACCESS_TOKEN = '+BsJhkBP3WK5asR+aW95mFETj9PJkLjjDnXr36V6gQWhLhSU1NlN5oUG5bd3QW+3TGbNgSwoskXY+GmPIaizFJCvQhSH///hh9rgXLi1hjFndShWFZwZWtv/t5sOy9HzrW8RzeD7TpUsM587XFBkZwdB04t89/1O/w1cDnyilFU='
CHANNEL_SECRET = '24e3d508f9ac6632039f9dd9e767d466'

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

@app.route("/callback", methods=['POST'])
def callback():
    """LINE 서버가 새 메시지를 보내주는 창구(엔드포인트)입니다."""
    # LINE에서 온 요청이 맞는지 보안 서명 확인
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    
    print(f"📩 [알림] 새 메시지 수신: {body}")

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("❌ 보안 서명 오류: LINE 서버에서 온 요청이 아닙니다.")
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    """실제 텍스트 메시지를 처리하고 답장을 보내는 부분입니다."""
    user_message = event.message.text
    print(f"💬 학생 질문: {user_message}")

    # 1. Q&A 엔진에서 답변 찾기
    answer, score = get_best_reply(user_message)

    # 2. 답변 전송 로직 (임계점 8점 이상일 때만 자동 답변)
    if answer and score >= 8:
        reply_text = answer
        print(f"🎯 Q&A 매칭 성공 (점수: {score}) -> 자동 답장 전송")
    else:
        # 매칭 실패 시 기본 안내 메시지 (또는 Claude API 연결)
        reply_text = "질문을 확인했습니다! 시안 선생님이 확인 후 자세히 답변해 주실 예정이니 조금만 기다려주세요 😊"
        print("⚠️ 매칭 실패 -> 대기 메시지 전송")

    # 3. LINE 서버로 답장 쏘기 (0.1초 만에 전송됨)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

if __name__ == "__main__":
    print("🚀 LINE 챗봇 서버가 시작되었습니다! (Port: 5000)")
    app.run(port=5000)