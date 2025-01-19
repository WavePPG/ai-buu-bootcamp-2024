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
    ImageComponent,
    Separator
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
        # โมเดลที่ใช้ในการสร้างเวกเตอร์ของข้อความ
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize in-memory database
        self.database = {
            'documents': [],
            'embeddings': [],
            'metadata': []
        }

        # สร้าง FAISS index
        self.create_faiss_index()

    def add_document(self, text: str, metadata: dict = None):
        """Add document to in-memory database with embedding"""
        # ประมวลผลข้อความเพื่อหาเวกเตอร์ของข้อความ
        embedding = self.embedding_model.encode([text])[0]

        # เพิ่มข้อมูลลงในฐานข้อมูล
        self.database['documents'].append(text)
        self.database['embeddings'].append(embedding.tolist())
        self.database['metadata'].append(metadata or {})

        # สร้าง FAISS index ใหม่หลังจากเพิ่มเอกสาร
        self.create_faiss_index()

    def create_faiss_index(self):
        """Create FAISS index for similarity search"""
        if not self.database['embeddings']:
            self.index = None
            return

        # แปลงข้อมูลเป็น numpy array
        embeddings = np.array(self.database['embeddings'], dtype='float32')
        dimension = embeddings.shape[1]

        # สร้าง FAISS index
        self.index = faiss.IndexFlatL2(dimension)

        # เพิ่มข้อมูลลงใน FAISS index
        self.index.add(embeddings)

    def retrieve_documents(self, query: str, top_k: int = 3):
        """Retrieve most relevant documents"""
        if not self.database['embeddings'] or self.index is None:
            return []

        # แปลงข้อความเป็นเวกเตอร์
        query_embedding = self.embedding_model.encode([query]).astype('float32')

        # ค้นหาเอกสารที่เกี่ยวข้องด้วย similarity search
        D, I = self.index.search(query_embedding, top_k)

        return [self.database['documents'][i] for i in I[0] if i < len(self.database['documents'])]

    def clear_database(self):
        """Clear in-memory database"""
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
    # ข้อมูลตัวอย่างที่ใช้สำหรับ RAG
    sample_documents = [
        # เพิ่มเอกสารตัวอย่างที่นี่ถ้าต้องการ
    ]
    # เพิ่มข้อมูลตัวอย่างลงใน RAG
    for doc in sample_documents:
        rag.add_document(doc)

    yield

    # ลบข้อมูลที่ใช้ในการทดสอบออกจาก RAG
    rag.clear_database()

app = FastAPI(lifespan=lifespan)

# ข้อความคู่มือสำหรับฟีเจอร์ Emergency
EMERGENCY_MANUAL = """
คู่มือการใช้งานฟีเจอร์ "Emergency"
**ฟังก์ชันหลัก:**
- **คำแนะนำในกรณีฉุกเฉิน**: กดปุ่ม "Emergency" เพื่อรับคำแนะนำในสถานการณ์ฉุกเฉินต่างๆ
- **ถามตอบกับบอท**: พิมพ์คำถามเกี่ยวกับสถานการณ์ฉุกเฉิน เช่น "ช้างเหยียบรถควรทำยังไง" เพื่อรับคำตอบทันที
"""

# ข้อความคู่มือสำหรับฟีเจอร์ ระวังช้าง
WATCH_ELEPHANT_MANUAL = """
เมื่อช้างเข้าใกล้ในสถานการณ์ฉุกเฉิน ควรทำตามขั้นตอนดังนี้:
1. รักษาความสงบ: หลีกเลี่ยงการแสดงอาการตกใจหรือกลัว
2. หลีกเลี่ยงการสบตา: ไม่มองตาช้างโดยตรง
3. ค่อยๆ ถอยหลังออก: เคลื่อนไหวอย่างช้าๆ เพื่อสร้างระยะห่าง
4. หาที่หลบภัย: เข้าไปในที่มีอุปสรรค เช่น ต้นไม้ใหญ่หรือกำแพง
5. ติดต่อเจ้าหน้าที่: โทรขอความช่วยเหลือทันที โทร **086-092-6529** เป็นหมายเลขโทรศัพท์ของ **ศูนย์บริการนักท่องเที่ยว** ที่คุณสามารถติดต่อเมื่อพบช้างบนถนนหรือมีเหตุฉุกเฉินอื่นๆ ในพื้นที่เขาใหญ่
"""

# ข้อความสำหรับ "ตรวจสอบช้างก่อนเดินทาง"
CHECK_ELEPHANT_MANUAL = """
ตรวจช้างก่อนเดินทาง!** เช็คความปลอดภัยก่อนออกเดินทางที่นี่ 👉 [คลิกเลย](https://aprlabtop.com/Honey_test/chang_v3.php)
"""

# ข้อความคู่มือสำหรับการติดต่อเจ้าหน้าที่
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

# ฟังก์ชันสำหรับสร้าง Flex Send Message ที่มี header และข้อความในกล่อง
def create_flex_message(text: str) -> FlexSendMessage:
    bubble = BubbleContainer(
        direction='ltr',
        header=BoxComponent(
            layout='horizontal',
            contents=[
                ImageComponent(
                    url='https://i.imgur.com/your-header-icon.png',  # เปลี่ยน URL เป็นไอคอนที่คุณต้องการ
                    size='xxs',
                    aspect_ratio='1:1',
                    aspect_mode='cover',
                    margin='none'
                ),
                TextComponent(
                    text='ข้อมูลสำคัญ',
                    weight='bold',
                    size='lg',
                    color='#FFFFFF',
                    margin='md',
                    flex=5
                )
            ],
            padding_all='10px',
            background_color='#1DB446'  # สีพื้นหลังของ header
        ),
        body=BoxComponent(
            layout='vertical',
            contents=[
                TextComponent(
                    text=text,
                    wrap=True,
                    weight='regular',
                    size='md',
                    color='#000000'
                )
            ],
            padding_all='10px',
            background_color='#F0F0F0',  # สีพื้นหลังของกล่องข้อความ
            border_width='1px',
            border_color='#CCCCCC',
            corner_radius='10px'
        ),
        footer=BoxComponent(
            layout='vertical',
            contents=[
                Separator(),  # เพิ่มเส้นคั่น
                ButtonComponent(
                    style='primary',
                    action=URIAction(
                        label='ดูเพิ่มเติม',
                        uri='https://example.com'  # เปลี่ยน URI เป็นลิงก์ที่ต้องการ
                    )
                )
            ],
            padding_all='10px',
            background_color='#FFFFFF'
        )
    )
    
    return FlexSendMessage(alt_text="Flex Message", contents=bubble)

# ฟังก์ชันสำหรับสร้าง Carousel Flex Send Message ที่มี header
def create_carousel_message() -> FlexSendMessage:
    # บับเบิลแรก
    bubble1 = BubbleContainer(
        direction='ltr',
        header=BoxComponent(
            layout='horizontal',
            contents=[
                ImageComponent(
                    url='https://i.imgur.com/header-icon1.png',  # เปลี่ยน URL เป็นไอคอนที่คุณต้องการ
                    size='xxs',
                    aspect_ratio='1:1',
                    aspect_mode='cover',
                    margin='none'
                ),
                TextComponent(
                    text='หัวข้อแรก',
                    weight='bold',
                    size='lg',
                    color='#FFFFFF',
                    margin='md',
                    flex=5
                )
            ],
            padding_all='10px',
            background_color='#1DB446'  # สีพื้นหลังของ header
        ),
        body=BoxComponent(
            layout='vertical',
            contents=[
                TextComponent(
                    text="Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
                    wrap=True,
                    size='md',
                    color='#000000'
                )
            ],
            padding_all='10px',
            background_color='#F0F0F0',
            border_width='1px',
            border_color='#CCCCCC',
            corner_radius='10px'
        ),
        footer=BoxComponent(
            layout='vertical',
            contents=[
                Separator(),  # เพิ่มเส้นคั่น
                ButtonComponent(
                    style='primary',
                    action=URIAction(
                        label='ไปที่เว็บไซต์',
                        uri='https://example.com'
                    )
                )
            ],
            padding_all='10px',
            background_color='#FFFFFF'
        )
    )

    # บับเบิลที่สอง
    bubble2 = BubbleContainer(
        direction='ltr',
        header=BoxComponent(
            layout='horizontal',
            contents=[
                ImageComponent(
                    url='https://i.imgur.com/header-icon2.png',  # เปลี่ยน URL เป็นไอคอนที่คุณต้องการ
                    size='xxs',
                    aspect_ratio='1:1',
                    aspect_mode='cover',
                    margin='none'
                ),
                TextComponent(
                    text='หัวข้อที่สอง',
                    weight='bold',
                    size='lg',
                    color='#FFFFFF',
                    margin='md',
                    flex=5
                )
            ],
            padding_all='10px',
            background_color='#1DB446'  # สีพื้นหลังของ header
        ),
        body=BoxComponent(
            layout='vertical',
            contents=[
                TextComponent(
                    text="Hello, World!",
                    wrap=True,
                    size='md',
                    color='#000000'
                )
            ],
            padding_all='10px',
            background_color='#F0F0F0',
            border_width='1px',
            border_color='#CCCCCC',
            corner_radius='10px'
        ),
        footer=BoxComponent(
            layout='vertical',
            contents=[
                Separator(),  # เพิ่มเส้นคั่น
                ButtonComponent(
                    style='primary',
                    action=URIAction(
                        label='ไปที่เว็บไซต์',
                        uri='https://example.com'
                    )
                )
            ],
            padding_all='10px',
            background_color='#FFFFFF'
        )
    )

    # สร้าง Carousel Container ที่บรรจุหลายบับเบิล
    carousel = CarouselContainer(
        contents=[bubble1, bubble2]
    )

    # สร้าง Flex Send Message ด้วย Carousel
    return FlexSendMessage(
        alt_text="Carousel Message",
        contents=carousel
    )

# Endpoint สำหรับการสร้าง Webhook
@app.post('/message')
async def message(request: Request):
    # การตรวจสอบ headers จากการขอเรียกใช้บริการว่ามาจากทาง LINE Platform จริง
    signature = request.headers.get('X-Line-Signature')
    if not signature:
        raise HTTPException(
            status_code=400, detail="X-Line-Signature header is missing"
        )

    # ข้อมูลที่ส่งมาจาก LINE Platform
    body = await request.body()

    try:
        # เรียกใช้งาน Handler เพื่อจัดข้อความจาก LINE Platform
        handler.handle(body.decode("UTF-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

# ปรับปรุงฟังก์ชัน handle_message เพื่อใช้ Carousel Flex Send Message
@handler.add(MessageEvent, message=(TextMessage, ImageMessage))
def handle_message(event: MessageEvent):
    # ตรวจสอบ Message ว่าเป็นประเภทข้อความ Text
    if isinstance(event.message, TextMessage):
        user_message = event.message.text
        # ตรวจสอบว่าผู้ใช้ต้องการคู่มือการใช้งานหรือไม่
        manual_response = get_manual_response(user_message)
        if manual_response:
            reply = create_flex_message(manual_response)  # ใช้ Flex Send Message แบบกล่องเดียว
        else:
            # การค้นหาข้อมูลจาก RAG
            retrieved_docs = rag.retrieve_documents(user_message, top_k=3)  # เปลี่ยน top_k เป็น 3

            if retrieved_docs:
                # สร้างบับเบิลสำหรับแต่ละเอกสารที่ค้นพบ
                bubbles = []
                for doc in retrieved_docs:
                    if "http" not in doc:
                        text = doc
                    else:
                        text = "ดูข้อมูลเพิ่มเติมที่นี่"  # หรือข้อความอื่นๆ ตามต้องการ
                    bubble = BubbleContainer(
                        direction='ltr',
                        header=BoxComponent(
                            layout='horizontal',
                            contents=[
                                ImageComponent(
                                    url='https://i.imgur.com/header-icon.png',  # เปลี่ยน URL เป็นไอคอนที่คุณต้องการ
                                    size='xxs',
                                    aspect_ratio='1:1',
                                    aspect_mode='cover',
                                    margin='none'
                                ),
                                TextComponent(
                                    text='ข้อมูลสำคัญ',
                                    weight='bold',
                                    size='lg',
                                    color='#FFFFFF',
                                    margin='md',
                                    flex=5
                                )
                            ],
                            padding_all='10px',
                            background_color='#1DB446'  # สีพื้นหลังของ header
                        ),
                        body=BoxComponent(
                            layout='vertical',
                            contents=[
                                TextComponent(
                                    text=text,
                                    wrap=True,
                                    size='md',
                                    color='#000000'
                                )
                            ],
                            padding_all='10px',
                            background_color='#F0F0F0',
                            border_width='1px',
                            border_color='#CCCCCC',
                            corner_radius='10px'
                        ),
                        footer=BoxComponent(
                            layout='vertical',
                            contents=[
                                Separator(),  # เพิ่มเส้นคั่น
                                ButtonComponent(
                                    style='primary',
                                    action=URIAction(
                                        label='ดูเพิ่มเติม',
                                        uri='https://example.com'  # เปลี่ยน URI ตามเอกสาร
                                    )
                                )
                            ],
                            padding_all='10px',
                            background_color='#FFFFFF'
                        )
                    )
                    bubbles.append(bubble)
                
                # สร้าง Carousel ด้วยบับเบิลที่สร้างขึ้น
                carousel = CarouselContainer(contents=bubbles)
                reply = FlexSendMessage(
                    alt_text="Carousel Message",
                    contents=carousel
                )
            else:
                default_text = "ขออภัย ฉันไม่เข้าใจคำถามของคุณ กรุณาลองใหม่อีกครั้ง"
                reply = create_flex_message(default_text)  # ใช้ Flex Send Message แบบกล่องเดียว

        # Reply ข้อมูลกลับไปยัง LINE
        line_bot_api.reply_message(
            event.reply_token,
            [reply]
        )

    # ตรวจสอบ Message ว่าเป็นประเภทข้อความ Image
    elif isinstance(event.message, ImageMessage):
        # การขอข้อมูลภาพจาก LINE Service
        headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
        url = f"https://api-data.line.me/v2/bot/message/{event.message.id}/content"
        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            image_data = BytesIO(response.content)
            image = Image.open(image_data)
        except Exception as e:
            error_reply = create_flex_message("เกิดข้อผิดพลาด, กรุณาลองใหม่อีกครั้ง🙏🏻")
            line_bot_api.reply_message(
                event.reply_token,
                [error_reply]
            )
            return

        if image.size[0] * image.size[1] > 1024 * 1024:
            size_error_reply = create_flex_message("ขอโทษครับ ภาพมีขนาดใหญ่เกินไป กรุณาลดขนาดภาพและลองใหม่อีกครั้ง")
            line_bot_api.reply_message(
                event.reply_token,
                [size_error_reply]
            )
            return

        # เนื่องจากเราไม่ใช้ Gemini ในการสร้างคำตอบจากรูปภาพ
        # คุณอาจต้องการเพิ่มฟังก์ชันการประมวลผลภาพเพิ่มเติมเอง
        # สำหรับตัวอย่างนี้ จะตอบกลับด้วย Flex Send Message แบบกล่องเดียว
        image_reply = create_flex_message("ขณะนี้ระบบไม่สามารถประมวลผลรูปภาพได้ กรุณาสอบถามด้วยข้อความแทนค่ะ 🙏🏻")
        line_bot_api.reply_message(
            event.reply_token,
            [image_reply]
        )

# Endpoint สำหรับทดสอบ RAG ด้วยข้อความ
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

# Endpoint สำหรับทดสอบ RAG ด้วยรูปภาพ (ยังไม่ได้ปรับปรุงเนื่องจากไม่ใช้ Gemini)
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
