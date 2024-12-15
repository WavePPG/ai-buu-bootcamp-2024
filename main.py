import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent,
    MemberJoinedEvent
)
from linebot.v3.exceptions import InvalidSignatureError

app = FastAPI()

# ข้อมูล token และ channel secret สำหรับ LINE
ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "RMuXBCLD7tGSbkGgdELH7Vz9+Qz0YhqCIeKBhpMdKvOVii7W2L9rNpAHjYGigFN4ORLknMxhuWJYKIX3uLrY1BUg7E3Bk0v3Fmc5ZIC53d8fOdvIMyZQ6EdaOS0a6kejeqcX/dRFI/JfiFJr5mdwZgdB04t89/1O/w1cDnyilFU=")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "175149695b4d312eabb9df4b7e3e7a95")

# การเชื่อมต่อ และตั้งค่าข้อมูลเพื่อเรียกใช้งาน LINE Messaging API
configuration = Configuration(access_token=ACCESS_TOKEN)
handler = WebhookHandler(channel_secret=CHANNEL_SECRET)

@app.post('/message')
async def message(request: Request):
    # การตรวจสอบ headers จากการขอเรียกใช้บริการว่ามาจากทาง LINE Platform จริง
    signature = request.headers.get('X-Line-Signature')
    if not signature:
        raise HTTPException(
            status_code=400, detail="X-Line-Signature header is missing"
        )

    # ข้อมูลที่ส่งมาจาก LINE Platform
    body = await request.body()

    try:
        # เรียกใช้งาน Handler เพื่อจัดข้อความจาก LINE Platform
        handler.handle(body.decode("UTF-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

# Handler สำหรับข้อความทั่วไป (สามารถลบได้ถ้าไม่ต้องการใช้งาน)
@handler.add(MessageEvent, message=(TextMessageContent, ImageMessageContent))
def handle_message(event: MessageEvent):
    # ตัวอย่างการตอบกลับข้อความทั่วไป (สามารถปรับแต่งหรือลบได้)
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        reply_text = "ขอบคุณสำหรับข้อความของคุณ!"
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                replyToken=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )

# Handler สำหรับการเข้าร่วมสมาชิกใหม่ในกลุ่ม
@handler.add(MemberJoinedEvent)
def handle_member_joined(event: MemberJoinedEvent):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        # ข้อความที่ต้องการส่งเมื่อมีสมาชิกใหม่เข้าร่วมกลุ่ม
        welcome_message = TextMessage(text="https://aprlabtop.com/Honey_test/map_1.php")
        # ส่งข้อความตอบกลับไปยังกลุ่ม
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                replyToken=event.reply_token,
                messages=[welcome_message]
            )
        )

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, host="0.0.0.0")
