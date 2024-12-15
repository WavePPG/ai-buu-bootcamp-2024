import requests
from fastapi import FastAPI, Request, HTTPException
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, FlexMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.exceptions import InvalidSignatureError
import os

app = FastAPI()

# ข้อมูล token และ channel secret สำหรับ LINE
ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "RMuXBCLD7tGSbkGgdELH7Vz9+Qz0YhqCIeKBhpMdKvOVii7W2L9rNpAHjYGigFN4ORLknMxhuWJYKIX3uLrY1BUg7E3Bk0v3Fmc5ZIC53d8fOdvIMyZQ6EdaOS0a6kejeqcX/dRFI/JfiFJr5mdwZgdB04t89/1O/w1cDnyilFU=")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "175149695b4d312eabb9df4b7e3e7a95")

# LINE API Configuration
configuration = Configuration(access_token=ACCESS_TOKEN)
handler = WebhookHandler(channel_secret=CHANNEL_SECRET)


@app.post("/webhook")
async def webhook(request: Request):
    signature = request.headers.get("X-Line-Signature")
    if not signature:
        raise HTTPException(status_code=400, detail="X-Line-Signature header is missing")
    body = await request.body()
    try:
        handler.handle(body.decode("UTF-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return {"message": "OK"}


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    user_message = event.message.text.strip().lower()

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        # Flex Message JSON Structure
        flex_message_content = {
            "type": "bubble",
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": "เลือกตัวเลือกที่ต้องการ:",
                        "weight": "bold",
                        "size": "md",
                        "margin": "md"
                    },
                    {
                        "type": "button",
                        "style": "primary",
                        "action": {
                            "type": "message",
                            "label": "ช้าง 🐘",
                            "text": "ช้าง"
                        },
                        "color": "#1E90FF",
                        "margin": "md"
                    },
                    {
                        "type": "button",
                        "style": "primary",
                        "action": {
                            "type": "message",
                            "label": "แผนที่ 🗺️",
                            "text": "แผนที่"
                        },
                        "color": "#32CD32",
                        "margin": "md"
                    }
                ]
            }
        }

        flex_message = FlexMessage(
            alt_text="เลือกตัวเลือก", contents=flex_message_content
        )

        # Response based on user's input
        if user_message == "ช้าง":
            reply_message = "คุณเลือกช้าง 🐘! นี่คือข้อมูลเกี่ยวกับช้าง..."
        elif user_message == "แผนที่":
            reply_message = "คุณเลือกแผนที่ 🗺️! นี่คือแผนที่..."
        else:
            reply_message = "กรุณาเลือกจากตัวเลือกด้านล่าง!"

        # Reply Message with Flex Message
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                replyToken=event.reply_token,
                messages=[
                    {"type": "text", "text": reply_message},  # Normal Text Reply
                    flex_message.dict()  # Flex Message Reply
                ]
            )
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
