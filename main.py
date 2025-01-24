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

ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "RMuXBCLD7tGSbkGgdELH7Vz9+Qz0YhqCIeKBhpMdKvOVii7W2L9rNpAHjYGigFN4ORLknMxhuWJYKIX3uLrY1BUg7E3Bk0v3Fmc5ZIC53d8fOdvIMyZQ6EdaOS0a6kejeqcX/dRFI/JfiFJr5mdwZgdB04t89/1O/w1cDnyilFU=")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "175149695b4d312eabb9df4b7e3e7a95")

line_bot_api = LineBotApi(ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

EMERGENCY_MANUAL = """🚨 คู่มือฉุกเฉิน WildSafe

📱 ขอความช่วยเหลือ
• กดปุ่ม Emergency ทันที
• รับคำแนะนำเร่งด่วน

💬 สอบถามข้อมูล
• พิมพ์คำถามที่ต้องการความช่วยเหลือ
• เช่น "เจอช้างป่าทำอย่างไร?"

⚡️ พร้อมช่วยเหลือตลอด 24 ชม."""

WATCH_ELEPHANT_MANUAL = """🐘 พบช้างป่า - คู่มือความปลอดภัย

1. 😌 รักษาสติ ไม่ตื่นตระหนก
2. 👀 ห้ามสบตาช้าง
3. 🚶 ถอยห่างช้าๆ อย่างนิ่มนวล
4. 🌳 หลบหลังต้นไม้/กำแพง
5. 📞 โทรด่วน: 086-092-6529

⚠️ อย่าวิ่ง อย่าส่งเสียงดัง!"""

CHECK_ELEPHANT_MANUAL = """🔍 ตรวจสอบเส้นทางปลอดภัย

• เช็คจุดพบช้างป่าล่าสุด
• วางแผนเส้นทางปลอดภัย

📍 กดปุ่ม GO MAP เพื่อดูแผนที่"""

OFFICER_MANUAL = """📞 ติดต่อฉุกเฉิน 24 ชม.

🚑 แจ้งเหตุด่วน: 1669
🏕️ ศูนย์บริการ: 086-092-6529
🌲 อุทยานฯ: 086-092-6527"""

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

def create_bubble_container(text: str) -> BubbleContainer:
    return BubbleContainer(
        size='mega',
        header=BoxComponent(
            layout='vertical',
            backgroundColor='#27AE60',
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
            paddingAll='12px',
            paddingBottom='0px',
            contents=[
                TextComponent(
                    text=text,
                    wrap=True,
                    size='sm'
                )
            ]
        ),
        footer=BoxComponent(
            layout='vertical',
            paddingAll='12px',
            paddingTop='8px',
            contents=[
                ButtonComponent(
                    style='link',
                    height='sm',
                    adjustMode='shrink-to-fit',
                    action=URIAction(
                        label='GO MAP',
                        uri='https://aprlabtop.com/Honey_test/chang_v3.php'
                    ),
                    color='#FFFFFF',
                    backgroundColor='#27AE60'
                )
            ]
        ),
        styles={
            "header": {"backgroundColor": "#27AE60"},
            "body": {"separator": False},
            "footer": {"separator": False}
        }
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
        
        if manual_response:
            reply = create_flex_message(manual_response)
        else:
            retrieved_docs = rag.retrieve_documents(user_message, top_k=3)
            
            if retrieved_docs:
                texts = ["ดูข้อมูลเพิ่มเติมที่นี่" if "http" in doc else doc for doc in retrieved_docs]
                reply = create_carousel_message(texts)
            else:
                error_message = """❓ ไม่พบข้อมูลที่คุณต้องการ

• ลองถามใหม่อีกครั้ง
• เลือกหัวข้อที่ต้องการความช่วยเหลือ
• กด GO MAP เพื่อดูแผนที่"""
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
    retrieved_docs = rag.retrieve_documents(text, top_k=1)
    reply = retrieved_docs[0] if retrieved_docs else "ขออภัย ไม่พบข้อมูลที่คุณต้องการ"
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
        "message": "🔄 ระบบกำลังพัฒนา ไม่สามารถประมวลผลรูปภาพได้"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, host="0.0.0.0")
