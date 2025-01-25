import requests
from io import BytesIO
from PIL import Image
import uvicorn
import numpy as np
import os
import faiss
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, FlexSendMessage,
    BubbleContainer, CarouselContainer, BoxComponent,
    TextComponent, ButtonComponent, URIAction
)
from linebot.exceptions import InvalidSignatureError
from sentence_transformers import SentenceTransformer
from typing import Dict
from contextlib import asynccontextmanager

ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "RMuXBCLD7tGSbkGgdELH7Vz9+Qz0YhqCIeKBhpMdKvOVii7W2L9rNpAHjYGigFN4ORLknMxhuWJYKIX3uLrY1BUg7E3Bk0v3Fmc5ZIC53d8fOdvIMyZQ6EdaOS0a6kejeqcX/dRFI/JfiFJr5mdwZgdB04t89/1O/w1cDnyilFU=")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "175149695b4d312eabb9df4b7e3e7a95")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBfkFZ8DCBb57CwW8WIwqSbUTB3fyIfw6g")

# Setup LINE API
line_bot_api = LineBotApi(ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# Setup Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

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
    # Add your sample documents here
    sample_documents = [EMERGENCY_MANUAL, WATCH_ELEPHANT_MANUAL, CHECK_ELEPHANT_MANUAL, OFFICER_MANUAL]
    for doc in sample_documents:
        rag.add_document(doc)
    yield
    rag.clear_database()

app = FastAPI(lifespan=lifespan)

# Your manual content here
EMERGENCY_MANUAL = """..."""  # Your existing manual content
WATCH_ELEPHANT_MANUAL = """..."""
CHECK_ELEPHANT_MANUAL = """..."""
OFFICER_MANUAL = """..."""

def get_manual_response(user_message: str) -> str:
    user_message = user_message.strip().lower()
    manuals = {
        "emergency": EMERGENCY_MANUAL,
        "คู่มือการใช้งาน": EMERGENCY_MANUAL,
        "emergency เกิดเหตุฉุกเฉินทำยังไง": WATCH_ELEPHANT_MANUAL,
        "มีเหตุร้ายใกล้ตัว": WATCH_ELEPHANT_MANUAL,
        "ตรวจสอบช้างก่อนเดินทาง": CHECK_ELEPHANT_MANUAL,
        "ติดต่อเจ้าหน้าที่": OFFICER_MANUAL,
        "contact officer": OFFICER_MANUAL
    }
    return manuals.get(user_message)

def create_bubble_container(text: str) -> BubbleContainer:
    return BubbleContainer(
        header=BoxComponent(
            layout='vertical',
            contents=[
                TextComponent(
                    text="WildSafe",
                    weight='bold',
                    align='center',
                    color='#FFFFFF',
                    size='xl'
                )
            ],
            background_color='#27AE60'
        ),
        body=BoxComponent(
            layout='vertical',
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
            contents=[
                ButtonComponent(
                    style='primary',
                    action=URIAction(
                        label='GO MAP',
                        uri='https://aprlabtop.com/Honey_test/chang_v3.php'
                    )
                )
            ]
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
    return {"status": "ok"}

@handler.add(MessageEvent, message=(TextMessage, ImageMessage))
def handle_message(event: MessageEvent):
    if isinstance(event.message, TextMessage):
        user_message = event.message.text
        manual_response = get_manual_response(user_message)
        
        if manual_response:
            reply = create_flex_message(manual_response)
        else:
            # Try RAG first
            retrieved_docs = rag.retrieve_documents(user_message, top_k=3)
            
            if retrieved_docs:
                texts = ["ดูข้อมูลเพิ่มเติมที่นี่" if "http" in doc else doc for doc in retrieved_docs]
                reply = create_carousel_message(texts)
            else:
                # If RAG fails, use Gemini
                try:
                    gemini_response = model.generate_content(user_message)
                    reply = create_flex_message(gemini_response.text)
                except Exception as e:
                    reply = create_flex_message("ขออภัย เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง")

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
                message = "ขอโทษครับ ภาพมีขนาดใหญ่เกินไป กรุณาลดขนาดภาพและลองใหม่อีกครั้ง"
            else:
                try:
                    gemini_response = model.generate_content(['อธิบายรูปภาพนี้', image])
                    message = gemini_response.text
                except Exception:
                    message = "ขณะนี้ระบบไม่สามารถประมวลผลรูปภาพได้ กรุณาสอบถามด้วยข้อความแทนค่ะ 🙏🏻"
                
        except Exception:
            message = "เกิดข้อผิดพลาด, กรุณาลองใหม่อีกครั้ง🙏🏻"
            
        reply = create_flex_message(message)
        line_bot_api.reply_message(event.reply_token, [reply])

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, host="0.0.0.0")
