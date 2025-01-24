import requests
from io import BytesIO
from PIL import Image
import uvicorn
import numpy as np
import os
import faiss
from dotenv import load_dotenv
import logging

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent,
    TextMessage,
    ImageMessage,
    FlexSendMessage,
    BubbleContainer,
    BoxComponent,
    TextComponent,
    ButtonComponent,
    URIAction,
    CarouselContainer,  # เพิ่มการนำเข้า CarouselContainer
)
from linebot.exceptions import InvalidSignatureError
from sentence_transformers import SentenceTransformer
from typing import Dict
from contextlib import asynccontextmanager

# โหลดตัวแปรสิ่งแวดล้อมจากไฟล์ .env
load_dotenv()

# กำหนดค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ข้อมูล token และ channel secret สำหรับ LINE
ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
Gemini_API_Key = os.getenv("GEMINI_API_KEY")
Gemini_Endpoint_URL = os.getenv("GEMINI_ENDPOINT_URL")

line_bot_api = LineBotApi(ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

class RAGSystem:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.database = {
            'documents': [],
            'embeddings': [],
            'metadata': []
        }
        self.create_faiss_index()

    def add_document(self, text: str, metadata: dict = None):
        embedding = self.embedding_model.encode([text])[0]
        self.database['documents'].append(text)
        self.database['embeddings'].append(embedding.tolist())
        self.database['metadata'].append(metadata or {})
        self.create_faiss_index()

    def create_faiss_index(self):
        if not self.database['embeddings']:
            self.index = None
            return
        embeddings = np.array(self.database['embeddings'], dtype='float32')
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve_documents(self, query: str, top_k: int = 3):
        if not self.database['embeddings'] or self.index is None:
            return []
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        D, I = self.index.search(query_embedding, top_k)
        return [self.database['documents'][i] for i in I[0] if i < len(self.database['documents'])]

    def clear_database(self):
        self.database = {
            'documents': [],
            'embeddings': [],
            'metadata': []
        }
        self.index = None

rag = RAGSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    sample_documents = []
    for doc in sample_documents:
        rag.add_document(doc)
    yield
    rag.clear_database()

app = FastAPI(lifespan=lifespan)

# ข้อความคู่มือต่างๆ - เวอร์ชันปรับปรุง
EMERGENCY_MANUAL = """
📱 คู่มือการใช้งาน WildSafe 🦮

🔸 วิธีใช้งานฉุกเฉิน
   • กดปุ่ม "Emergency" ทันทีเมื่อต้องการความช่วยเหลือ
   • รับคำแนะนำที่เป็นประโยชน์สำหรับสถานการณ์ฉุกเฉิน

🔸 สอบถามข้อมูลด่วน
   • พิมพ์คำถามที่ต้องการความช่วยเหลือ
   • เช่น "พบช้างบนถนนต้องทำอย่างไร?"
   • รับคำตอบและวิธีแก้ไขสถานการณ์ทันที
"""

WATCH_ELEPHANT_MANUAL = """
🐘 แนวทางปฏิบัติเมื่อพบช้างในระยะใกล้ 🚨

1. 😌 รักษาสติ - ควบคุมอารมณ์ให้สงบ ไม่ตื่นตระหนก

2. 👀 หลีกเลี่ยงการสบตา - อย่าจ้องมองช้างโดยตรง

3. 🚶‍♂️ ถอยออกอย่างช้าๆ - เคลื่อนที่เงียบๆ สร้างระยะห่างที่ปลอดภัย

4. 🌳 หาที่กำบัง - มองหาต้นไม้ใหญ่หรือสิ่งกีดขวางที่แข็งแรง

5. ☎️ แจ้งเจ้าหน้าที่ทันที
   📞 โทร: 086-092-6529 (ศูนย์บริการนักท่องเที่ยว 24 ชม.)
"""

CHECK_ELEPHANT_MANUAL = """
🔍 ตรวจสอบเส้นทางก่อนเดินทาง!

🐘 เช็คพื้นที่พบช้างป่าล่าสุด
👉 คลิกเพื่อดูแผนที่: https://aprlabtop.com/Honey_test/chang_v3.php

⚠️ เพื่อความปลอดภัยของคุณและช้างป่า
"""

OFFICER_MANUAL = """
📞 ติดต่อขอความช่วยเหลือ 🆘

🚑 เหตุฉุกเฉิน 24 ชม.: 1669

🏕️ ติดต่อเจ้าหน้าที่พื้นที่:
• ศูนย์บริการนักท่องเที่ยว: 086-092-6529
• อุทยานแห่งชาติเขาใหญ่: 086-092-6527

⏰ พร้อมให้บริการตลอด 24 ชั่วโมง
"""

def get_manual_response(user_message: str) -> str:
    user_message = user_message.strip().lower()
    if user_message in ["emergency", "คู่มือการใช้งาน"]:
        return EMERGENCY_MANUAL
    elif user_message in ["emergency เกิดเหตุฉุกเฉินทำยังไง", "มีเหตุร้ายใกล้ตัว"]:
        return WATCH_ELEPHANT_MANUAL
    elif user_message == "ตรวจสอบช้างก่อนเดินทาง":
        return CHECK_ELEPHANT_MANUAL
    elif user_message in ["ติดต่อเจ้าหน้าที่", "contact officer"]:
        return OFFICER_MANUAL
    else:
        return None

def get_gemini_response(query: str, api_key: str, endpoint_url: str):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {"query": query}
    
    try:
        response = requests.post(endpoint_url, headers=headers, json=data, timeout=10)
        response.raise_for_status()  # ยกข้อผิดพลาดถ้ามีสถานะไม่ใช่ 200
        response_data = response.json()
        return response_data.get("response", "ขอโทษค่ะ ฉันไม่สามารถช่วยได้ในขณะนี้")
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err} - Response: {response.text}")
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Connection error occurred: {conn_err}")
        return "เกิดข้อผิดพลาดในการเชื่อมต่อกับ Gemini"
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Timeout error occurred: {timeout_err}")
        return "การเชื่อมต่อกับ Gemini ใช้เวลานานเกินไป"
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred: {req_err}")
        return "เกิดข้อผิดพลาดในการเชื่อมต่อกับ Gemini"

def create_bubble_container(text: str) -> BubbleContainer:
    return BubbleContainer(
        header=BoxComponent(
            layout='vertical',
            contents=[TextComponent(
                text="WildSafe",
                weight='bold',
                align='center',
                color='#FFFFFF',
                size='xl'
            )],
            background_color='#27AE60'
        ),
        body=BoxComponent(
            layout='vertical',
            contents=[TextComponent(
                text=text,
                wrap=True,
                size='sm'
            )]
        ),
        footer=BoxComponent(
            layout='vertical',
            contents=[ButtonComponent(
                style='primary',
                action=URIAction(
                    label='GO MAP',
                    uri='https://aprlabtop.com/Honey_test/chang_v3.php'
                )
            )]
        )
    )

def create_flex_message(text: str) -> FlexSendMessage:
    bubble = create_bubble_container(text)
    return FlexSendMessage(alt_text="WildSafe Message", contents=bubble)

def create_carousel_message(texts: list) -> FlexSendMessage:
    bubbles = [create_bubble_container(text) for text in texts]
    carousel = CarouselContainer(contents=bubbles)
    return FlexSendMessage(alt_text="WildSafe Carousel", contents=carousel)

@app.post('/message')
async def message(request: Request):
    signature = request.headers.get('X-Line-Signature')
    if not signature:
        raise HTTPException(status_code=400, detail="X-Line-Signature header is missing")
    
    body = await request.body()
    try:
        handler.handle(body.decode("UTF-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

@handler.add(MessageEvent, message=(TextMessage, ImageMessage))
def handle_message(event: MessageEvent):
    if isinstance(event.message, TextMessage):
        user_message = event.message.text
        manual_response = get_manual_response(user_message)
        
        if not manual_response:
            manual_response = get_gemini_response(user_message, Gemini_API_Key, Gemini_Endpoint_URL)
        
        reply = create_flex_message(manual_response)
        line_bot_api.reply_message(event.reply_token, [reply])

    elif isinstance(event.message, ImageMessage):
        try:
            headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
            url = f"https://api-data.line.me/v2/bot/message/{event.message.id}/content"
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            image_data = BytesIO(response.content)
            image = Image.open(image_data)
            
            if image.size[0] * image.size[1] > 1024 * 1024:
                message = "ขอโทษครับ ภาพมีขนาดใหญ่เกินไป กรุณาลดขนาดภาพและลองใหม่อีกครั้ง"
            else:
                message = "ขณะนี้ระบบไม่สามารถประมวลผลรูปภาพได้ กรุณาสอบถามด้วยข้อความแทนค่ะ 🙏🏻"
                
        except Exception:
            message = "เกิดข้อผิดพลาด, กรุณาลองใหม่อีกครั้ง🙏🏻"
            
        reply = create_flex_message(message)
        line_bot_api.reply_message(event.reply_token, [reply])

@app.get('/test-message')
async def test_message_rag(text: str):
    retrieved_docs = rag.retrieve_documents(text, top_k=1)
    reply = retrieved_docs[0] if retrieved_docs else "ขออภัย ฉันไม่เข้าใจคำถามของคุณ กรุณาลองใหม่อีกครั้ง"
    return {"answer": reply}

@app.post('/image-query')
async def image_query(
    file: UploadFile = File(...), 
    query: str = Form("อธิบายภาพนี้ให้ละเอียด"),
    use_rag: bool = Form(True)
):
    if file.size > 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image size too large")
    
    contents = await file.read()
    return {
        "message": "ขณะนี้ระบบไม่สามารถประมวลผลรูปภาพได้ กรุณาสอบถามด้วยข้อความแทนค่ะ 🙏🏻"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, host="0.0.0.0")
