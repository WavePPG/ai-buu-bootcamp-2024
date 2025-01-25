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

# Initialize FastAPI app
app = FastAPI()

# LINE Messaging API credentials
ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "RMuXBCLD7tGSbkGgdELH7Vz9+Qz0YhqCIeKBhpMdKvOVii7W2L9rNpAHjYGigFN4ORLknMxhuWJYKIX3uLrY1BUg7E3Bk0v3Fmc5ZIC53d8fOdvIMyZQ6EdaOS0a6kejeqcX/dRFI/JfiFJr5mdwZgdB04t89/1O/w1cDnyilFU=")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "175149695b4d312eabb9df4b7e3e7a95")
# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBfkFZ8DCBb57CwW8WIwqSbUTB3fyIfw6g")
# Configure LINE Messaging API
configuration = Configuration(access_token=ACCESS_TOKEN)
handler = WebhookHandler(channel_secret=CHANNEL_SECRET)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Predefined manuals
EMERGENCY_MANUAL = """
คู่มือการใช้งานฟีเจอร์ "Emergency"
**ฟังก์ชันหลัก:**
- **คำแนะนำในกรณีฉุกเฉิน**: กดปุ่ม "Emergency" เพื่อรับคำแนะนำในสถานการณ์ฉุกเฉินต่างๆ
- **ถามตอบกับบอท**: พิมพ์คำถามเกี่ยวกับสถานการณ์ฉุกเฉิน เพื่อรับคำตอบทันที
"""

WATCH_ELEPHANT_MANUAL = """
เมื่อช้างเข้าใกล้ในสถานการณ์ฉุกเฉิน ควรทำตามขั้นตอนดังนี้:
1. รักษาความสงบ: หลีกเลี่ยงการแสดงอาการตกใจหรือกลัว
2. หลีกเลี่ยงการสบตา: ไม่มองตาช้างโดยตรง
3. ค่อยๆ ถอยหลังออก: เคลื่อนไหวอย่างช้าๆ เพื่อสร้างระยะห่าง
4. หาที่หลบภัย: เข้าไปในที่มีอุปสรรค เช่น ต้นไม้ใหญ่หรือกำแพง
5. ติดต่อเจ้าหน้าที่: โทรขอความช่วยเหลือทันที
"""

CHECK_ELEPHANT_MANUAL = """
ตรวจช้างก่อนเดินทาง! เช็คความปลอดภัยก่อนออกเดินทางที่นี่
"""

OFFICER_MANUAL = """
**ติดต่อเจ้าหน้าที่**
- **หมายเลขหลัก**: 1669 (บริการฉุกเฉิน 24 ชั่วโมง)
- **ศูนย์บริการนักท่องเที่ยว**: โทร 086-092-6529
- **ที่ทำการอุทยานแห่งชาติเขาใหญ่**: โทร 086-092-6527
"""

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
    sample_documents = [
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

def create_bubble_container(text: str) -> dict:
    # Ensure text is not None and convert to string
    safe_text = str(text) if text is not None else "ไม่มีข้อความ"
    
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
                    "text": safe_text,
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
    # ทำให้แน่ใจว่า text ไม่เป็น None และแปลงเป็น string
    safe_text = str(text) if text is not None else "ไม่มีข้อความ"
    
    bubble = create_bubble_container(safe_text)
    
    # ทำให้แน่ใจว่า altText ไม่เป็นค่าว่าง
    alt_text = "WildSafe: " + (safe_text[:50] + "...") if safe_text.strip() else "WildSafe: ไม่มีข้อความ"
    
    # ทำให้แน่ใจว่า contents มีค่าที่ถูกต้อง
    if not isinstance(bubble, dict) or 'contents' not in bubble:
        bubble = create_bubble_container("ขออภัย เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง")
    
    return {
        "type": "flex",
        "altText": alt_text,
        "contents": bubble
    }


def create_carousel_message(texts: list) -> dict:
    # Ensure each text is valid
    safe_texts = [str(text) if text is not None else "ไม่มีข้อความ" for text in texts]
    bubbles = [create_bubble_container(text) for text in safe_texts]
    
    # Create a summary of all texts for altText
    summary = " | ".join(safe_texts)[:50] + "..."
    
    return {
        "type": "flex",
        "altText": "WildSafe Carousel: " + summary,
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
    return {"message": "OK"}

@handler.add(MessageEvent, message=(TextMessageContent, ImageMessageContent))
def handle_message(event: MessageEvent):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        if isinstance(event.message, TextMessageContent):
            try:
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
                        try:
                            gemini_response = gemini_model.generate_content(user_message)
                            response_text = gemini_response.text if gemini_response.text else "ขออภัย ไม่สามารถประมวลผลคำตอบได้"
                            reply = create_flex_message(response_text)
                        except Exception as e:
                            reply = create_flex_message("ขออภัย ไม่สามารถประมวลผลคำตอบได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง")

                # Validate Flex message before sending
                if (not reply.get('contents') or 
                    not reply.get('altText') or 
                    reply['altText'].strip() == '' or 
                    not isinstance(reply['contents'], dict)):
                    reply = create_flex_message("ขออภัย เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง")

                line_bot_api.reply_message_with_http_info(
                    ReplyMessageRequest(
                        replyToken=event.reply_token,
                        messages=[reply]
                    )
                )

            except Exception as e:
                fallback_reply = create_flex_message("ขออภัย เกิดข้อผิดพลาดในระบบ กรุณาลองใหม่อีกครั้ง")
                line_bot_api.reply_message_with_http_info(
                    ReplyMessageRequest(
                        replyToken=event.reply_token,
                        messages=[fallback_reply]
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
                    message = "ขณะนี้ระบบไม่สามารถประมวลผลรูปภาพได้ กรุณาสอบถามด้วยข้อความแทนค่ะ 🙏🏻"
                    
            except Exception:
                message = "เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้ง 🙏🏻"
                
            reply = create_flex_message(message)
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[reply]
                )
            )
@app.get('/test-message')
async def test_message_rag(text: str):
    try:
        # Try to get documents from RAG system
        retrieved_docs = rag.retrieve_documents(text, top_k=1)
        
        if retrieved_docs:
            reply = retrieved_docs[0]
        else:
            # Try Gemini as fallback
            try:
                gemini_response = gemini_model.generate_content(text)
                reply = gemini_response.text if gemini_response.text else "ขออภัย ไม่สามารถประมวลผลคำตอบได้"
            except Exception:
                reply = "ขออภัย ฉันไม่เข้าใจคำถามของคุณ กรุณาลองใหม่อีกครั้ง"
                
        return {"answer": reply}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง"
        )

@app.post('/image-query')
async def image_query(
    file: UploadFile = File(...),
    query: str = Form("อธิบายภาพนี้ให้ละเอียด"),
    use_rag: bool = Form(True)
):
    try:
        # Check file size
        contents = await file.read()
        if len(contents) > 1024 * 1024:  # 1MB limit
            raise HTTPException(
                status_code=400, 
                detail="ขนาดไฟล์ใหญ่เกินไป กรุณาลดขนาดไฟล์และลองใหม่อีกครั้ง"
            )

        # Process image and return response
        return {
            "message": "ขณะนี้ระบบไม่สามารถประมวลผลรูปภาพได้ กรุณาสอบถามด้วยข้อความแทนค่ะ 🙏🏻"
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
