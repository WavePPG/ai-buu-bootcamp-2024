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
    BoxBackground,  # เพิ่มตรงนี้
)
from linebot.exceptions import InvalidSignatureError
from sentence_transformers import SentenceTransformer
from typing import Dict
from contextlib import asynccontextmanager

app = FastAPI()

# ข้อมูล token และ channel secret สำหรับ LINE
ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "RMuXBCLD7tGSbkGgdELH7Vz9+Qz0YhqCIeKBhpMdKvOVii7W2L9rNpAHjYGigFN4ORLknMxhuWJYKIX3uLrY1BUg7E3Bk0v3Fmc5ZIC53d8fOdvIMyZQ6EdaOS0a6kejeqcX/dRFI/JfiFJr5mdwZgdB04t89/1O/w1cDnyilFU=")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "175149695b4d312eabb9df4b7e3e7a95")

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

# ข้อความคู่มือต่างๆ
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

# ฟังก์ชันสำหรับสร้าง Flex Message แบบกล่องเดียว
def create_flex_message(text: str) -> FlexSendMessage:
   bubble = BubbleContainer(
       header=BoxComponent(
           layout='vertical',
           background=BoxBackground(color='#1eb900'),
           paddingAll='20px',
           contents=[
               TextComponent(
                   text="WildSafe",
                   weight='bold', 
                   size='xl',
                   color='#FFFFFF',
                   align='center'
               )
           ]
       ),
       body=BoxComponent(
           layout='horizontal',
           paddingAll='20px',
           contents=[
               TextComponent(
                   text=text,
                   wrap=True
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
   return FlexSendMessage(alt_text="WildSafe Message", contents=bubble)
def create_carousel_message() -> FlexSendMessage:
    # บับเบิลแรก
    bubble1 = BubbleContainer(
        header=BoxComponent(
            layout='vertical',
            background=BoxBackground(color='#1eb900'),
            paddingAll='20px',
            contents=[
                TextComponent(
                    text="WildSafe",
                    weight='bold',
                    size='xl',
                    color='#FFFFFF',
                    align='center'
                )
            ]
        ),
        body=BoxComponent(
            layout='horizontal',
            paddingAll='20px',
            contents=[
                TextComponent(
                    text="Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
                    wrap=True
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

    # บับเบิลที่สอง
    bubble2 = BubbleContainer(
        header=BoxComponent(
            layout='vertical',
            background=BoxBackground(color='#1eb900'),
            paddingAll='20px',
            contents=[
                TextComponent(
                    text="WildSafe",
                    weight='bold',
                    size='xl',
                    color='#FFFFFF',
                    align='center'
                )
            ]
        ),
        body=BoxComponent(
            layout='horizontal',
            paddingAll='20px',
            contents=[
                TextComponent(
                    text="Hello, World!",
                    wrap=True
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

    carousel = CarouselContainer(contents=[bubble1, bubble2])
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
                bubbles = []
                for doc in retrieved_docs:
                    text = "ดูข้อมูลเพิ่มเติมที่นี่" if "http" in doc else doc
                    bubble = BubbleContainer(
                        header=BoxComponent(
                            layout='vertical',
                            backgroundColor='#27AE60',
                            paddingAll='20px',
                            contents=[
                                TextComponent(
                                    text="WildSafe",
                                    weight='bold',
                                    size='xl',
                                    color='#FFFFFF',
                                    align='center'
                                )
                            ]
                        ),
                        body=BoxComponent(
                            layout='horizontal',
                            contents=[
                                TextComponent(
                                    text=text,
                                    wrap=True
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
                reply = FlexSendMessage(
                    alt_text="WildSafe Carousel",
                    contents=carousel
                )
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
