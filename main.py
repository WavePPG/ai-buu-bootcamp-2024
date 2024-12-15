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
    FollowEvent
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

# Handler สำหรับข้อความทั่วไป
@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    # ตรวจสอบข้อความที่ส่งมา
    user_message = event.message.text.lower()
    # กำหนดคำที่บอทจะตอบกลับเมื่อพบคำขอลิงก์
    keywords = ["ขอลิ้ง", "ส่งลิงก์", "ขอแหล่งข้อมูล", "ขอลิงค์"]

    if any(keyword in user_message for keyword in keywords):
        response_text = "นี่คือลิงก์ที่คุณขอ: https://aprlabtop.com/Honey_test/map_1.php"
    else:
        # ถ้าไม่ใช่คำที่ต้องการ บอทสามารถตอบกลับข้อความทั่วไป หรือไม่ตอบก็ได้
        response_text = "ถ้าคุณต้องการลิงก์ กรุณาพิมพ์คำว่า 'ขอลิ้ง' หรือ 'ส่งลิงก์'"

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                replyToken=event.reply_token,
                messages=[TextMessage(text=response_text)]
            )
        )

# Handler สำหรับการเพิ่มเพื่อน OA
@handler.add(FollowEvent)
def handle_follow(event: FollowEvent):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        # ข้อความที่ต้องการส่งเมื่อมีผู้ใช้เพิ่ม OA เป็นเพื่อน
        follow_message = TextMessage(text="ขอบคุณที่เพิ่มเรามาเป็นเพื่อน! คุณสามารถเข้าถึงข้อมูลเพิ่มเติมได้ที่: https://aprlabtop.com/Honey_test/map_1.php")
        # ส่งข้อความตอบกลับไปยังผู้ใช้
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                replyToken=event.reply_token,
                messages=[follow_message]
            )
        )

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, host="0.0.0.0")
