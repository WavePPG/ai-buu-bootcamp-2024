import requests
from io import BytesIO
from PIL import Image
import uvicorn
import numpy as np
import os
import faiss

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent,
    TextMessage,
    ImageMessage,
    FlexSendMessage,
    BubbleContainer,
    CarouselContainer,
    BoxComponent,
    TextComponent,
    ButtonComponent,
    URIAction,
)
from linebot.exceptions import InvalidSignatureError
from sentence_transformers import SentenceTransformer
from typing import Dict
from contextlib import asynccontextmanager

app = FastAPI()

# ข้อมูล token และ channel secret สำหรับ LINE
ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "YOUR_LINE_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "YOUR_LINE_CHANNEL_SECRET")

# การเชื่อมต่อ และตั้งค่าข้อมูลเพื่อเรียกใช้งาน LINE Messaging API
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

# สร้าง Object สำหรับใช้งาน RAG
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

# ฟังก์ชันสำหรับตรวจสอบและตอบกลับข้อความคู่มือ
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

# ฟังก์ชันสำหรับสร้าง Flex Message แบบกล่องเดียวที่มีปุ่ม GO MAP และ Header สีเขียว
def create_flex_message(text: str) -> FlexSendMessage:
    # แยกข้อความเป็นบรรทัดเพื่อจัดการกับลิงก์
    lines = text.strip().split('\n')
    contents = []
    for line in lines:
        if "คลิกเพื่อดูแผนที่" in line:
            # แยกข้อความก่อนและหลังลิงก์
            before, link_part = line.split("คลิกเพื่อดูแผนที่:")
            contents.append(TextComponent(text=before.strip(), wrap=True, size='md'))
            # สร้าง TextComponent สำหรับลิงก์
            contents.append(TextComponent(
                text="คลิกเพื่อดูแผนที่",
                size='md',
                color='#1E90FF',  # สีลิงก์
                action=URIAction(
                    label='คลิกเพื่อดูแผนที่',
                    uri='https://aprlabtop.com/Honey_test/chang_v3.php'
                )
            ))
        else:
            contents.append(TextComponent(text=line.strip(), wrap=True, size='md'))

    bubble = BubbleContainer(
        header=BoxComponent(
            layout='vertical',
            backgroundColor='#27AE60',  # สีเขียว
            paddingAll='10px',
            contents=[
                TextComponent(
                    text="WildSafe",
                    weight='bold',
                    color='#FFFFFF',
                    size='xl',
                    align='center'
                )
            ]
        ),
        body=BoxComponent(
            layout='vertical',
            contents=contents
        ),
        footer=BoxComponent(
            layout='horizontal',
            contents=[
                ButtonComponent(
                    style='primary',
                    action=URIAction(
                        label='GO MAP',
                        uri='https://aprlabtop.com/Honey_test/chang_v3.php'
                    )
                )
            ],
            justifyContent='center'  # จัดกึ่งกลางปุ่ม
        )
    )
    return FlexSendMessage(alt_text="WildSafe Message", contents=bubble)

# ฟังก์ชันสำหรับสร้าง Carousel Flex Message ที่มีปุ่ม GO MAP และ Header สีเขียวในแต่ละบับเบิล
def create_carousel_message(texts: list) -> FlexSendMessage:
    bubbles = []
    for text in texts:
        # แยกข้อความเป็นบรรทัดเพื่อจัดการกับลิงก์
        lines = text.strip().split('\n')
        contents = []
        for line in lines:
            if "คลิกเพื่อดูแผนที่" in line:
                # แยกข้อความก่อนและหลังลิงก์
                before, link_part = line.split("คลิกเพื่อดูแผนที่:")
                contents.append(TextComponent(text=before.strip(), wrap=True, size='md'))
                # สร้าง TextComponent สำหรับลิงก์
                contents.append(TextComponent(
                    text="คลิกเพื่อดูแผนที่",
                    size='md',
                    color='#1E90FF',  # สีลิงก์
                    action=URIAction(
                        label='คลิกเพื่อดูแผนที่',
                        uri='https://aprlabtop.com/Honey_test/chang_v3.php'
                    )
                ))
            else:
                contents.append(TextComponent(text=line.strip(), wrap=True, size='md'))

        bubble = BubbleContainer(
            header=BoxComponent(
                layout='vertical',
                backgroundColor='#27AE60',  # สีเขียว
                paddingAll='10px',
                contents=[
                    TextComponent(
                        text="WildSafe",
                        weight='bold',
                        color='#FFFFFF',
                        size='xl',
                        align='center'
                    )
                ]
            ),
            body=BoxComponent(
                layout='vertical',
                contents=contents
            ),
            footer=BoxComponent(
                layout='horizontal',
                contents=[
                    ButtonComponent(
                        style='primary',
                        action=URIAction(
                            label='GO MAP',
                            uri='https://aprlabtop.com/Honey_test/chang_v3.php'
                        )
                    )
                ],
                justifyContent='center'  # จัดกึ่งกลางปุ่ม
            )
        )
        bubbles.append(bubble)
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
        
        if manual_response:
            reply = create_flex_message(manual_response)
        else:
            retrieved_docs = rag.retrieve_documents(user_message, top_k=3)

            if retrieved_docs:
                texts = ["ดูข้อมูลเพิ่มเติมที่นี่" if "http" in doc else doc for doc in retrieved_docs]
                reply = create_carousel_message(texts)
            else:
                # ข้อความเมื่อไม่พบข้อมูลที่ต้องการ
                error_message = """
❓ ไม่พบข้อมูลที่คุณต้องการ

• ลองถามใหม่อีกครั้ง
• เลือกหัวข้อที่ต้องการความช่วยเหลือ
• กด GO MAP เพื่อดูแผนที่
"""
                reply = create_flex_message(error_message)

        line_bot_api.reply_message(
            event.reply_token,
            [reply]
        )

    elif isinstance(event.message, ImageMessage):
        try:
            headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
            url = f"https://api-data.line.me/v2/bot/message/{event.message.id}/content"
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            image_data = BytesIO(response.content)
            image = Image.open(image_data)
            
            if image.size[0] * image.size[1] > 1024 * 1024:
                message = """🙏 ขออภัย รูปภาพมีขนาดใหญ่เกินไป

• กรุณาลดขนาดรูปภาพ
• ลองส่งใหม่อีกครั้ง"""
            else:
                message = """🔄 ระบบกำลังพัฒนา

• ไม่สามารถประมวลผลรูปภาพได้
• กรุณาสอบถามด้วยข้อความ"""
                
        except Exception:
            message = """⚠️ เกิดข้อผิดพลาด

• กรุณาลองใหม่อีกครั้ง
• หรือติดต่อเจ้าหน้าที่"""
            
        reply = create_flex_message(message)
        line_bot_api.reply_message(event.reply_token, [reply])

@app.get('/test-message')
async def test_message_rag(text: str):
    """
    Debug message from RAG
    """
    retrieved_docs = rag.retrieve_documents(text, top_k=1)
    if retrieved_docs:
        reply = retrieved_docs[0]
    else:
        reply = "ขออภัย ฉันไม่เข้าใจคำถามของคุณ กรุณาลองใหม่อีกครั้ง"

    return {
        "answer": reply
    }

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

    # เนื่องจากเราไม่ใช้ Gemini ในการประมวลผลภาพ
    # คุณอาจต้องการเพิ่มฟังก์ชันการประมวลผลภาพเองที่นี่
    return {
        "message": "ขณะนี้ระบบไม่สามารถประมวลผลรูปภาพได้ กรุณาสอบถามด้วยข้อความแทนค่ะ 🙏🏻"
    }

if __name__ == "__main__":
    uvicorn.run("main:app",
                port=8000,
                host="0.0.0.0")
