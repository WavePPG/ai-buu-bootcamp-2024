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
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent
)
from linebot.v3.exceptions import InvalidSignatureError
from sentence_transformers import SentenceTransformer
from typing import Dict
from contextlib import asynccontextmanager

# ข้อมูล token และ channel secret สำหรับ LINE
ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "RMuXBCLD7tGSbkGgdELH7Vz9+Qz0YhqCIeKBhpMdKvOVii7W2L9rNpAHjYGigFN4ORLknMxhuWJYKIX3uLrY1BUg7E3Bk0v3Fmc5ZIC53d8fOdvIMyZQ6EdaOS0a6kejeqcX/dRFI/JfiFJr5mdwZgdB04t89/1O/w1cDnyilFU=")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "175149695b4d312eabb9df4b7e3e7a95")

# การเชื่อมต่อ และตั้งค่าข้อมูลเพื่อเรียกใช้งาน LINE Messaging API
configuration = Configuration(access_token=ACCESS_TOKEN)
handler = WebhookHandler(channel_secret=CHANNEL_SECRET)

app = FastAPI()

class RAGSystem:
    def __init__(self, json_db_path: str, embedding_model: str = 'all-MiniLM-L6-v2'):
        # โมเดลที่ใช้ในการสร้างเวกเตอร์ของข้อความ
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # ข้อมูล JSON ที่ใช้เก็บข้อมูล
        self.json_db_path = json_db_path
        
        # Load ฐานข้อมูลจากไฟล์ JSON
        self.load_database()
        
        # สร้าง FAISS index
        self.create_faiss_index()
    
    def load_database(self):
        """Load existing database or create new"""
        try:
            with open(self.json_db_path, 'r', encoding='utf-8') as f:
                self.database = json.load(f)
        except FileNotFoundError:
            self.database = {
                'documents': [],
                'embeddings': [],
                'metadata': []
            }
    
    def save_database(self):
        """Save database to JSON file"""
        with open(self.json_db_path, 'w', encoding='utf-8') as f:
            json.dump(self.database, f, indent=2, ensure_ascii=False)
     
    def add_document(self, text: str, metadata: dict = None):
        """Add document to database with embedding"""
        # ประมวลผลข้อความเพื่อหาเวกเตอร์ของข้อความ
        embedding = self.embedding_model.encode([text])[0]
        
        # เพิ่มข้อมูลลงในฐานข้อมูล
        self.database['documents'].append(text)
        self.database['embeddings'].append(embedding.tolist())
        self.database['metadata'].append(metadata or {})
        
        # Save ฐานข้อมูลลงในไฟล์ JSON
        self.save_database()
        self.create_faiss_index()
    
    def create_faiss_index(self):
        """Create FAISS index for similarity search"""
        if not self.database['embeddings']:
            return
        
        # แปลงข้อมูลเป็น numpy array
        embeddings = np.array(self.database['embeddings'], dtype='float32')
        dimension = embeddings.shape[1]

        # สร้าง FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        
        # เพิ่มข้อมูลลงใน FAISS index
        self.index.add(embeddings)
    
    def retrieve_documents(self, query: str, top_k: int = 1):
        """Retrieve most relevant documents"""
        if not self.database['embeddings']:
            return []
        
        # แปลงข้อความเป็นเวกเตอร์
        query_embedding = self.embedding_model.encode([query])
        
        # ค้นหาเอกสารที่เกี่ยวข้องด้วย similarity search
        D, I = self.index.search(query_embedding, top_k)
        
        return [self.database['documents'][i] for i in I[0]]
    
    def clear_database(self):
        """Clear database and save to JSON file"""
        self.database = {
            'documents': [],
            'embeddings': [],
            'metadata': []
        }
        self.save_database()

# สร้าง Object สำหรับใช้งาน RAG
rag = RAGSystem(
    json_db_path="rag_database.json"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ข้อมูลตัวอย่างที่ใช้สำหรับ RAG
    sample_documents = [
        # ข้อความโปรโมชั่นเดิม
        "**🔍 ตรวจช้างก่อนเดินทาง!** เช็คความปลอดภัยก่อนออกเดินทางที่นี่ 👉 [คลิกเลย](https://aprlabtop.com/Honey_test/chang_v3.php)",

        # ข้อความติดต่อเจ้าหน้าที่เพิ่มเติม
        """**เกิดเหตุฉุกเฉิน ช้างเข้าใกล้ควรทำยังไง** เมื่อช้างเข้าใกล้ในสถานการณ์ฉุกเฉิน ควรทำตามขั้นตอนดังนี้:

1. รักษาความสงบ: หลีกเลี่ยงการแสดงอาการตกใจหรือกลัว
2. หลีกเลี่ยงการสบตา: ไม่มองตาช้างโดยตรง
3. ค่อยๆ ถอยหลังออก: เคลื่อนไหวอย่างช้าๆ เพื่อสร้างระยะห่าง
4. หาที่หลบภัย: เข้าไปในที่มีอุปสรรค เช่น ต้นไม้ใหญ่หรือกำแพง
5. ติดต่อเจ้าหน้าที่: โทรขอความช่วยเหลือทันที โทร **086-092-6529** เป็นหมายเลขโทรศัพท์ของ **ศูนย์บริการนักท่องเที่ยว** ที่คุณสามารถติดต่อเมื่อพบช้างบนถนนหรือมีเหตุฉุกเฉินอื่นๆ ในพื้นที่เขาใหญ่""",

        # ชุดคำถามและคำตอบสำหรับกรณีฉุกเฉิน
        """**เกิดเหตุฉุกเฉิน ต้องเรียกเจ้าหน้าที่หมายเลขอะไร?**
**✅ คำตอบ:** หากเกิดเหตุฉุกเฉินทั่วไป เช่น อุบัติเหตุทางถนน หรือเหตุการณ์ที่ต้องการความช่วยเหลือทันที กรุณาเรียกหมายเลข **1669** ซึ่งเป็นหมายเลขบริการฉุกเฉินของประเทศไทย""",

        """**❓ หากช้างเหยียบรถควรทำอย่างไร?**
**✅ คำตอบ:** หากช้างเหยียบรถของคุณ ให้รักษาความสงบและอย่าพยายามเคลื่อนย้ายรถด้วยตัวเอง เพราะอาจทำให้ช้างตกใจและก่อให้เกิดอันตรายเพิ่มเติม
- ติดต่อเจ้าหน้าที่ทางตำรวจหรือหน่วยงานที่เกี่ยวข้องทันทีผ่านหมายเลข **1669** เพื่อขอความช่วยเหลือ
- หากมีผู้บาดเจ็บ ให้ให้ความช่วยเหลือเบื้องต้นและรอการมาถึงของทีมแพทย์""",
    ]
    
    # เพิ่มข้อมูลตัวอย่างลงใน RAG
    for doc in sample_documents:
        rag.add_document(doc)
        
    yield

    # ลบข้อมูลที่ใช้ในการทดสอบออกจาก RAG
    rag.clear_database()

# สร้าง FastAPI แอปพร้อมกับ lifespan
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
            user_message = event.message.text
            # การค้นหาข้อมูลจาก RAG
            retrieved_docs = rag.retrieve_documents(user_message, top_k=1)
            if retrieved_docs:
                # ตอบกลับด้วยข้อความที่ดึงมาจาก RAG
                reply = retrieved_docs[0]
            else:
                reply = "ขออภัย ฉันไม่เข้าใจคำถามของคุณ กรุณาลองใหม่อีกครั้ง"
            
            # Reply ข้อมูลกลับไปยัง LINE
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text=reply)]
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
                return

            # เนื่องจากเราไม่ใช้ Gemini ในการสร้างคำตอบจากรูปภาพ
            # คุณอาจต้องการเพิ่มฟังก์ชันการประมวลผลภาพเพิ่มเติมเอง
            # สำหรับตัวอย่างนี้ จะตอบกลับด้วยข้อความทั่วไป
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text="ขณะนี้ระบบไม่สามารถประมวลผลรูปภาพได้ กรุณาสอบถามด้วยข้อความแทนค่ะ 🙏🏻")]
                )
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
