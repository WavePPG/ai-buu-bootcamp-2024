import requests
from io import BytesIO
from PIL import Image
import uvicorn
import json
import numpy as np
import os
import faiss

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (Configuration,
                                  ApiClient,
                                  MessagingApi,
                                  ReplyMessageRequest,
                                  TextMessage,
                                  ButtonsTemplate,
                                  TemplateAction)
from linebot.v3.webhooks import (MessageEvent,
                                 TextMessageContent,
                                 ImageMessageContent)
from linebot.v3.exceptions import InvalidSignatureError
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from typing import Dict
from contextlib import asynccontextmanager


app = FastAPI()

# ข้อมูล token และ channel secret สำหรับ LINE
ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "RMuXBCLD7tGSbkGgdELH7Vz9+Qz0YhqCIeKBhpMdKvOVii7W2L9rNpAHjYGigFN4ORLknMxhuWJYKIX3uLrY1BUg7E3Bk0v3Fmc5ZIC53d8fOdvIMyZQ6EdaOS0a6kejeqcX/dRFI/JfiFJr5mdwZgdB04t89/1O/w1cDnyilFU=")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "175149695b4d312eabb9df4b7e3e7a95")

# ข้อมูล Gemini api key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyD2YLUIU0yDcp7Y7fKaFtlBwvvK9u5RgYI")

# การเชื่อมต่อ และตั้งค่าข้อมูลเพื่อเรียกใช้งาน LINE Messaging API
configuration = Configuration(access_token=ACCESS_TOKEN)
handler = WebhookHandler(channel_secret=CHANNEL_SECRET)

# ... (ส่วนของคลาส GeminiRAGSystem ยังคงเหมือนเดิม)

# สร้าง Object สำหรับใช้งาน Gemini
gemini = GeminiRAGSystem(
    json_db_path="gemini_rag_database.json", 
    gemini_api_key=GEMINI_API_KEY
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ข้อมูลตัวอย่างที่ใช้สำหรับ Gemini
    sample_documents = [
        "ลิ้ง คือ เช็คก่อนเพื่อความปลอดภัย https://aprlabtop.com/Honey_test/map_1.php",
        "ช้าง คือ เช็คก่อนเพื่อความปลอดภัย https://aprlabtop.com/Honey_test/map_1.php",
        "ข้อมูล คือ เช็คก่อนเพื่อความปลอดภัย https://aprlabtop.com/Honey_test/map_1.php",
        "ดูช้าง คือ เช็คก่อนเพื่อความปลอดภัย https://aprlabtop.com/Honey_test/map_1.php",
    ]
    
    # เพิ่มข้อมูลตัวอย่างลงใน Gemini
    for doc in sample_documents:
        gemini.add_document(doc)
        
    yield

    # ลบข้อมูลที่ใช้ในการทดสอบออกจาก Gemini
    gemini.clear_database()

app = FastAPI(lifespan=lifespan)

# Endpoint สำหรับการสร้าง Webhook
@app.post('/message')
async def message(request: Request):
    # การตรวจสอบ headers จากการขอเรียกใช้บริการว่ามาจากทาง LINE Platform จริง
    signature = request.headers.get('X-Line-Signature')
    if not signature:
        raise HTTPException(
            status_code=400, detail="X-Line-Signature header is missing")

    # ข้อมูลที่ส่งมาจาก LINE Platform
    body = await request.body()

    try:
        # เรียกใช้งาน Handler เพื่อจัดข้อความจาก LINE Platform
        handler.handle(body.decode("UTF-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

# Function สำหรับจัดการข้อมูลที่ส่งมากจาก LINE Platform
@handler.add(MessageEvent, message=(TextMessageContent, ImageMessageContent))
def handle_message(event: MessageEvent):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        # ตรวจสอบ Message ว่าเป็นประเภทข้อความ Text
        if isinstance(event.message, TextMessageContent):
            user_text = event.message.text.strip().lower()

            # ตรวจสอบว่าผู้ใช้ส่งข้อความที่เป็นตัวเลือกหรือไม่
            if user_text in ["ช้าง", "แผนที่"]:
                if user_text == "ช้าง":
                    response_text = "คุณเลือกช้าง! 🐘 นี่คือข้อมูลเกี่ยวกับช้าง..."
                elif user_text == "แผนที่":
                    response_text = "คุณเลือกแผนที่! 🗺️ นี่คือแผนที่ที่คุณต้องการดู..."
                else:
                    response_text = "ขออภัยครับ ไม่สามารถเข้าใจคำสั่งของคุณ"
            else:
                # นำข้อมูลส่งไปยัง Gemini เพื่อทำการประมวลผล และสร้างคำตอบ และส่งตอบกลับมา
                gemini_response, prompt = gemini.generate_response(event.message.text)
                response_text = gemini_response

            # สร้างปุ่มตัวเลือก
            buttons_template = ButtonsTemplate(
                text="เลือกตัวเลือกที่ต้องการ:",
                actions=[
                    TemplateAction(label="ช้าง 🐘", text="ช้าง"),
                    TemplateAction(label="แผนที่ 🗺️", text="แผนที่")
                ]
            )

            # Reply ข้อมูลกลับไปยัง LINE พร้อมปุ่ม
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[
                        TextMessage(text=response_text),
                        buttons_template
                    ]
                )
            )

        # ตรวจสอบ Message ว่าเป็นประเภทข้อความ Image
        elif isinstance(event.message, ImageMessageContent):
            # การขอข้อมูลภาพจาก LINE Service
            headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
            url = f"https://api-data.line.me/v2/bot/message/{event.message.id}/content"
            try:
                response = requests.get(url, headers=headers, stream=True)
                response.raise_for_status()
                image_data = BytesIO(response.content)
                image = Image.open(image_data)
            except Exception as e:
                line_bot_api.reply_message_with_http_info(
                    ReplyMessageRequest(
                        replyToken=event.reply_token,
                        messages=[TextMessage(text="เกิดข้อผิดพลาด, กรุณาลองใหม่อีกครั้ง🙏🏻")]
                    )
                )
                return
            
            if image.size[0] * image.size[1] > 1024 * 1024:
                line_bot_api.reply_message_with_http_info(
                    ReplyMessageRequest(
                        replyToken=event.reply_token,
                        messages=[TextMessage(text="ขอโทษครับ ภาพมีขนาดใหญ่เกินไป กรุณาลดขนาดภาพและลองใหม่อีกครั้ง")]
                    )
                )

            try:
                # ส่งข้อมูลภาพไปยัง Gemini เพื่อทำการประมวลผล
                gemini_response = gemini.process_image_query(response.content,
                                                             query="อธิบายภาพนี้ให้ละเอียด", 
                                                             use_rag=True)
                # นำข้อมูลที่ได้จาก Gemini มาใช้งาน
                response_text = gemini_response['final_response']
            except Exception as e:
                response_text = f"เกิดข้อผิดพลาด, ไม่สามารถประมวลผลรูปภาพได้"

            # สร้างปุ่มตัวเลือกหลังจากตอบกลับภาพ
            buttons_template = ButtonsTemplate(
                text="ต้องการทำอะไรต่อไป?",
                actions=[
                    TemplateAction(label="ช้าง 🐘", text="ช้าง"),
                    TemplateAction(label="แผนที่ 🗺️", text="แผนที่")
                ]
            )

            # Reply ข้อมูลกลับไปยัง LINE พร้อมปุ่ม
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    replyToken=event.reply_token, 
                    messages=[
                        TextMessage(text=response_text),
                        buttons_template
                    ]
                )
            )

# Endpoint สำหรับทดสอบ Gemini ด้วยข้อความ
@app.get('/test-message')
async def test_message_gemini(text: str):
    """
    Debug message from Gemini
    """
    response, prompt = gemini.generate_response(text)

    return {
        "gemini_answer": response,
        "full_prompt": prompt
    }

# Endpoint สำหรับทดสอบ Gemini ด้วยรูปภาพ
@app.post('/image-query')
async def image_query(
    file: UploadFile = File(...), 
    query: str = Form("อธิบายภาพนี้ให้ละเอียด"),
    use_rag: bool = Form(True)
):
    if file.size > 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image size too large")

    # อ่านข้อมูลภาพจากไฟล์ที่ส่งมา
    contents = await file.read()

    # ส่งข้อมูลภาพไปยัง Gemini เพื่อทำการประมวลผล
    image_response = gemini.process_image_query(
        image_content=contents,
        query=query,
        use_rag=use_rag
    )
    
    return image_response

if __name__ == "__main__":
    uvicorn.run("main:app",
                port=8000,
                host="0.0.0.0") 
