import os
import requests
from io import BytesIO
from PIL import Image
import uvicorn
import numpy as np
import faiss

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (Configuration,
                                  ApiClient,
                                  MessagingApi,
                                  ReplyMessageRequest,
                                  TextMessage)
from linebot.v3.webhooks import (MessageEvent,
                                 TextMessageContent,
                                 ImageMessageContent)
from linebot.v3.exceptions import InvalidSignatureError
from sentence_transformers import SentenceTransformer
from typing import Dict
from contextlib import asynccontextmanager
import google.generativeai as genai

# Initialize FastAPI app with lifespan for RAG system
app = FastAPI()

# LINE Messaging API credentials
ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "swniqM2ufZYtkLpiK5DJDvfw4ALzb4Fwf2hayI7mEuOP4GMKpMW5yhyJz9LzT6TKcZDAAXhs8DhKaGpyw8R6tgx2bhOlqfFSjmn2+KAX2+CNZRtWAMBqfSrLdRLxwauYcPwmmWVz5zw1NN1ejyFnWgdB04t89/1O/w1cDnyilFU=")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "df84a355cf9b2257b4b6793b789ffac2")

# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBfkFZ8DCBb57CwW8WIwqSbUTB3fyIfw6g")

# Configure LINE Messaging API
configuration = Configuration(access_token=ACCESS_TOKEN)
handler = WebhookHandler(channel_secret=CHANNEL_SECRET)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# RAG System Definition
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

# Initialize RAG system
rag = RAGSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Add pre-defined documents to RAG system here if needed
    sample_documents = [
        # ตัวอย่างเอกสาร
        "คู่มือการใช้งานฟีเจอร์ Emergency",
        "วิธีการปฏิบัติตนเมื่อช้างเข้ามาใกล้",
        "วิธีตรวจสอบช้างก่อนเดินทาง",
        "ข้อมูลการติดต่อเจ้าหน้าที่ในกรณีฉุกเฉิน"
    ]
    for doc in sample_documents:
        rag.add_document(doc)
    yield
    rag.clear_database()

app = FastAPI(lifespan=lifespan)

# Predefined manuals
EMERGENCY_MANUAL = """
คู่มือการใช้งานฟีเจอร์ "Emergency"
**ฟังก์ชันหลัก:**
- **คำแนะนำในกรณีฉุกเฉิน**: กดปุ่ม "Emergency" เพื่อรับคำแนะนำในสถานการณ์ฉุกเฉินต่างๆ
- **ถามตอบกับบอท**: พิมพ์คำถามเกี่ยวกับสถานการณ์ฉุกเฉิน เช่น "ช้างเหยียบรถควรทำยังไง" เพื่อรับคำตอบทันที
"""

WATCH_ELEPHANT_MANUAL = """
เมื่อช้างเข้าใกล้ในสถานการณ์ฉุกเฉิน ควรทำตามขั้นตอนดังนี้:
1. รักษาความสงบ: หลีกเลี่ยงการแสดงอาการตกใจหรือกลัว
2. หลีกเลี่ยงการสบตา: ไม่มองตาช้างโดยตรง
3. ค่อยๆ ถอยหลังออก: เคลื่อนไหวอย่างช้าๆ เพื่อสร้างระยะห่าง
4. หาที่หลบภัย: เข้าไปในที่มีอุปสรรค เช่น ต้นไม้ใหญ่หรือกำแพง
5. ติดต่อเจ้าหน้าที่: โทรขอความช่วยเหลือทันที โทร **086-092-6529** เป็นหมายเลขโทรศัพท์ของ **ศูนย์บริการนักท่องเที่ยว**
"""

CHECK_ELEPHANT_MANUAL = """
ตรวจช้างก่อนเดินทาง!** เช็คความปลอดภัยก่อนออกเดินทางที่นี่ 👉 [คลิกเลย](https://aprlabtop.com/Honey_test/chang_v3.php)
"""

OFFICER_MANUAL = """
**ติดต่อเจ้าหน้าที่**
- **หมายเลขหลัก**: 1669 (บริการฉุกเฉิน 24 ชั่วโมง)
- **ศูนย์บริการนักท่องเที่ยว**: โทร 086-092-6529
- **ที่ทำการอุทยานแห่งชาติเขาใหญ่**: โทร 086-092-6527
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

def create_bubble_container(text: str) -> dict:
    return {
        "type": "bubble",
        "header": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": "WildSafe",
                    "weight": "bold",
                    "align": "center",
                    "color": "#FFFFFF",
                    "size": "xl"
                }
            ],
            "backgroundColor": "#27AE60"
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": text,
                    "wrap": True,
                    "size": "sm"
                }
            ]
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "action": {
                        "type": "uri",
                        "label": "GO MAP",
                        "uri": "https://aprlabtop.com/Honey_test/chang_v3.php"
                    }
                }
            ]
        }
    }

def create_flex_message(text: str) -> dict:
    bubble = create_bubble_container(text)
    return {
        "type": "flex",
        "altText": "WildSafe Message",
        "contents": bubble
    }

def create_carousel_message(texts: list) -> dict:
    bubbles = [create_bubble_container(text) for text in texts]
    return {
        "type": "flex",
        "altText": "WildSafe Carousel",
        "contents": {
            "type": "carousel",
            "contents": bubbles
        }
    }

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

@handler.add(MessageEvent, message=(TextMessageContent, ImageMessageContent))
def handle_message(event: MessageEvent):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        if isinstance(event.message, TextMessageContent):
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
                    # ใช้ Gemini ตอบกลับคำถามที่ไม่มีใน RAG
                    try:
                        gemini_response = gemini_model.generate_content(user_message)
                        reply = create_flex_message(gemini_response.text)
                    except Exception as e:
                        reply = create_flex_message("เกิดข้อผิดพลาดในการประมวลผลคำถาม กรุณาลองใหม่อีกครั้ง🙏🏻")
        
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[reply]
                )
            )

        elif isinstance(event.message, ImageMessageContent):
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
                    # สามารถเพิ่มการประมวลผลรูปภาพด้วย Gemini หรือระบบอื่น ๆ ได้ที่นี่
                    message = "ขณะนี้ระบบไม่สามารถประมวลผลรูปภาพได้ กรุณาสอบถามด้วยข้อความแทนค่ะ 🙏🏻"
                    
            except Exception:
                message = "เกิดข้อผิดพลาด, กรุณาลองใหม่อีกครั้ง🙏🏻"
                
            reply = create_flex_message(message)
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[reply]
                )
            )

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
