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

# Constants
ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "RMuXBCLD7tGSbkGgdELH7Vz9+Qz0YhqCIeKBhpMdKvOVii7W2L9rNpAHjYGigFN4ORLknMxhuWJYKIX3uLrY1BUg7E3Bk0v3Fmc5ZIC53d8fOdvIMyZQ6EdaOS0a6kejeqcX/dRFI/JfiFJr5mdwZgdB04t89/1O/w1cDnyilFU=")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "175149695b4d312eabb9df4b7e3e7a95")

# LINE API Setup
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

# Manuals
EMERGENCY_MANUAL = """
📱 คู่มือการใช้งาน WILDSAFE 🦮

🔸 วิธีใช้งานฉุกเฉิน
   • กดปุ่ม "Emergency" ทันทีเมื่อต้องการความช่วยเหลือ
   • รับคำแนะนำที่เป็นประโยชน์สำหรับสถานการณ์ฉุกเฉิน
"""

WATCH_ELEPHANT_MANUAL = """
🐘 แนวทางปฏิบัติเมื่อพบช้างในระยะใกล้ 🚨

1. 😌 รักษาสติ - ควบคุมอารมณ์ให้สงบ ไม่ตื่นตระหนก
2. 👀 หลีกเลี่ยงการสบตา - อย่าจ้องมองช้างโดยตรง
3. 🚶‍♂️ ถอยออกอย่างช้าๆ - เคลื่อนที่เงียบๆ สร้างระยะห่างที่ปลอดภัย
"""

CHECK_ELEPHANT_MANUAL = """
🔍 ตรวจสอบเส้นทางก่อนเดินทาง!

🐘 เช็คพื้นที่พบช้างป่าล่าสุด
👉 คลิกเพื่อดูแผนที่: https://aprlabtop.com/Honey_test/chang_v3.php
"""

OFFICER_MANUAL = """
📞 ติดต่อขอความช่วยเหลือ 🆘

🚑 เหตุฉุกเฉิน 24 ชม.: 1669
🏕️ ศูนย์บริการนักท่องเที่ยว: 086-092-6529
"""

# RAG System Initialization
rag = RAGSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Add initial documents if needed
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
    return None

def create_flex_message(text: str) -> FlexSendMessage:
    bubble = BubbleContainer(
        header=BoxComponent(
            layout='vertical',
            contents=[
                TextComponent(
                    text="WILDSAFE",
                    weight='bold',
                    size='xl',
                    color='#000000',
                    align='center'
                )
            ]
        ),
        body=BoxComponent(
            layout='vertical',
            contents=[
                TextComponent(
                    text=text,
                    wrap=True,
                    align='center',
                    margin='md'
                )
            ]
        ),
        footer=BoxComponent(
            layout='horizontal',
            contents=[
                ButtonComponent(
                    style='primary',
                    action=URIAction(
                        type='uri',
                        label='GO MAP',
                        uri='https://aprlabtop.com/Honey_test/chang_v3.php'
                    )
                )
            ]
        )
    )
    return FlexSendMessage(alt_text="WILDSAFE Message", contents=bubble)

def create_carousel_message(retrieved_docs: list) -> FlexSendMessage:
    bubbles = []
    for doc in retrieved_docs:
        bubble = BubbleContainer(
            header=BoxComponent(
                layout='vertical',
                contents=[
                    TextComponent(
                        text="WILDSAFE",
                        weight='bold',
                        size='xl',
                        color='#000000',
                        align='center'
                    )
                ]
            ),
            body=BoxComponent(
                layout='vertical',
                contents=[
                    TextComponent(
                        text=doc,
                        wrap=True,
                        align='center',
                        margin='md'
                    )
                ]
            ),
            footer=BoxComponent(
                layout='horizontal',
                contents=[
                    ButtonComponent(
                        style='primary',
                        action=URIAction(
                            type='uri',
                            label='GO MAP',
                            uri='https://aprlabtop.com/Honey_test/chang_v3.php'
                        )
                    )
                ]
            )
        )
        bubbles.append(bubble)
    
    carousel = CarouselContainer(contents=bubbles)
    return FlexSendMessage(alt_text="WILDSAFE Carousel", contents=carousel)

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
                reply = create_carousel_message(retrieved_docs)
            else:
                reply = create_flex_message("ขออภัย ฉันไม่เข้าใจคำถามของคุณ กรุณาลองใหม่อีกครั้ง")

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
                reply = create_flex_message("ขอโทษครับ ภาพมีขนาดใหญ่เกินไป กรุณาลดขนาดภาพและลองใหม่อีกครั้ง")
            else:
                reply = create_flex_message("ขณะนี้ระบบไม่สามารถประมวลผลรูปภาพได้ กรุณาสอบถามด้วยข้อความแทนค่ะ 🙏🏻")
                
        except Exception as e:
            reply = create_flex_message("เกิดข้อผิดพลาด, กรุณาลองใหม่อีกครั้ง🙏🏻")
            
        line_bot_api.reply_message(
            event.reply_token,
            [reply]
        )

@app.get('/test-message')
async def test_message_rag(text: str):
    retrieved_docs = rag.retrieve_documents(text, top_k=1)
    if retrieved_docs:
        reply = retrieved_docs[0]
    else:
        reply = "ขออภัย ฉันไม่เข้าใจคำถามของคุณ กรุณาลองใหม่อีกครั้ง"

    return {"answer": reply}

@app.post('/image-query')
async def image_query(
    file: UploadFile = File(...), 
    query: str = Form("อธิบายภาพนี้ให้ละเอียด"),
    use_rag: bool = Form(True)
):
    if file.size > 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image size too large")

    return {
        "message": "ขณะนี้ระบบไม่สามารถประมวลผลรูปภาพได้ กรุณาสอบถามด้วยข้อความแทนค่ะ 🙏🏻"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, host="0.0.0.0")
