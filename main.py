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

# ข้อมูล token และ channel secret สำหรับ LINE
ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "RMuXBCLD7tGSbkGgdELH7Vz9+Qz0YhqCIeKBhpMdKvOVii7W2L9rNpAHjYGigFN4ORLknMxhuWJYKIX3uLrY1BUg7E3Bk0v3Fmc5ZIC53d8fOdvIMyZQ6EdaOS0a6kejeqcX/dRFI/JfiFJr5mdwZgdB04t89/1O/w1cDnyilFU=")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "175149695b4d312eabb9df4b7e3e7a95")

# การเชื่อมต่อ และตั้งค่าข้อมูล
configuration = Configuration(access_token=ACCESS_TOKEN)
handler = WebhookHandler(channel_secret=CHANNEL_SECRET)

class RAGSystem:
    def __init__(self, json_db_path: str, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.json_db_path = json_db_path
        self.load_database()
        self.create_faiss_index()
    
    def load_database(self):
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
        with open(self.json_db_path, 'w', encoding='utf-8') as f:
            json.dump(self.database, f, indent=2, ensure_ascii=False)
     
    def add_document(self, text: str, metadata: dict = None):
        embedding = self.embedding_model.encode([text])[0]
        self.database['documents'].append(text)
        self.database['embeddings'].append(embedding.tolist())
        self.database['metadata'].append(metadata or {})
        self.save_database()
        self.create_faiss_index()
    
    def create_faiss_index(self):
        if not self.database['embeddings']:
            return
        embeddings = np.array(self.database['embeddings'], dtype='float32')
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
    
    def retrieve_documents(self, query: str, top_k: int = 1):
        if not self.database['embeddings']:
            return []
        query_embedding = self.embedding_model.encode([query])
        D, I = self.index.search(query_embedding, top_k)
        return [self.database['documents'][i] for i in I[0]]
    
    def clear_database(self):
        self.database = {
            'documents': [],
            'embeddings': [],
            'metadata': []
        }
        self.save_database()

# สร้าง Object สำหรับใช้งาน RAG
rag = RAGSystem(json_db_path="rag_database.json")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ข้อมูลที่ปรับปรุงใหม่สำหรับ RAG
    consolidated_documents = [
        # ข้อความโปรโมชั่น
        """**🔍 ตรวจสถานะก่อนเดินทาง!** เช็คความปลอดภัยก่อนออกเดินทางที่นี่ 👉 [คลิกเลย](https://aprlabtop.com/Honey_test/chang_v3.php)""",

        # ข้อมูลฉุกเฉินและการติดต่อ
        """**เกิดเหตุฉุกเฉิน ช้างเข้าใกล้ควรทำยังไง:**
1. รักษาความสงบ ไม่แสดงอาการตื่นตระหนก
2. หลีกเลี่ยงการสบตาช้างโดยตรง
3. เคลื่อนที่ถอยหลังอย่างช้าๆ
4. มองหาที่กำบังปลอดภัย เช่น ต้นไม้ใหญ่
5. โทรขอความช่วยเหลือทันที

**หมายเลขโทรศัพท์สำคัญ:**
- ศูนย์บริการนักท่องเที่ยวเขาใหญ่: **086-092-6529**
- หน่วยฉุกเฉิน: **1669**""",

        # วิธีสังเกตอารมณ์ช้าง
        """**ช้างอารมณ์ดี:**
- หูและหางเคลื่อนไหวเป็นจังหวะ
- งวงเคลื่อนไหวอย่างผ่อนคลาย
- กินอาหารตามปกติ
- หากถูกรบกวน จะไล่ระยะสั้นแล้วเลิก""",

        """**ช้างอารมณ์ไม่ดี:**
- หูกางออก ไม่เคลื่อนไหว
- หางนิ่ง ไม่แกว่ง
- งวงแข็ง ชี้ตรง
- จ้องมองนิ่ง
- อาจพุ่งเข้าใส่ทันที""",

        # คำแนะนำเมื่อพบช้างบนถนน
        """**สิ่งที่ควรทำ:**
1. จอดรถห่างอย่างน้อย 30 เมตร
2. ติดเครื่องยนต์ไว้เสมอ
3. เปิดไฟหน้าแบบไฟต่ำ
4. ถอยรถอย่างช้าๆ ถ้าจำเป็น
5. แจ้งเจ้าหน้าที่ทันที""",

        """**สิ่งที่ห้ามทำ:**
1. บีบแตร
2. ส่งเสียงดัง
3. ใช้แฟลชถ่ายรูป
4. เปิดไฟกะพริบ
5. จอดดูหรือถ่ายรูปใกล้ๆ""",

        # กรณีเกิดเหตุฉุกเฉิน
        """**หากช้างทำร้ายรถ:**
1. อย่าขยับรถ
2. รักษาความสงบ
3. โทรแจ้งเจ้าหน้าที่ทันที
4. รอความช่วยเหลือ
5. หากมีผู้บาดเจ็บ ให้ปฐมพยาบาลเบื้องต้น""",

        """**ข้อควรระวังเป็นพิเศษ:**
- ช้างแม่ลูกอ่อนมักก้าวร้าวกว่าปกติ
- ช้างตกใจง่ายจากเสียงและแสง
- ช้างสามารถวิ่งเร็วกว่าที่คิด
- ให้ความร่วมมือกับรถคันอื่นในการหลบหลีก"""
    ]
    
    # เพิ่มข้อมูลที่ปรับปรุงแล้วลงใน RAG
    for doc in consolidated_documents:
        rag.add_document(doc)
    
    yield
    
    # ลบข้อมูลที่ใช้ในการทดสอบออกจาก RAG
    rag.clear_database()

app = FastAPI(lifespan=lifespan)

# ข้อความคู่มือสำหรับฟีเจอร์ Emergency
EMERGENCY_MANUAL = """
# คู่มือการใช้งานฟีเจอร์ "Emergency"

**ฟังก์ชันหลัก:**
- **คำแนะนำในกรณีฉุกเฉิน**: กดปุ่ม "Emergency" เพื่อรับคำแนะนำในสถานการณ์ฉุกเฉินต่างๆ
- **ถามตอบกับบอท**: พิมพ์คำถามเกี่ยวกับสถานการณ์ฉุกเฉิน เช่น "ช้างเหยียบรถควรทำยังไง" เพื่อรับคำตอบทันที

**วิธีใช้งาน:**
1. เปิดแอป Line และไปที่แชทบอทที่เกี่ยวข้อง
2. กดปุ่ม **"Emergency"** ที่เมนูหลักหรือแถบด้านล่าง
3. เลือกสถานการณ์ฉุกเฉินที่ต้องการ หรือพิมพ์คำถามของคุณ
4. บอทจะแนะนำขั้นตอนการจัดการสถานการณ์ให้คุณ

**ตัวอย่างการใช้งาน:**
- **สถานการณ์**: ช้างเหยียบรถ
- **การดำเนินการ**: กด "Emergency" > เลือก "ช้างเหยียบรถ" หรือพิมพ์ "ช้างเหยียบรถควรทำยังไง"
- **คำตอบจากบอท**: ให้คำแนะนำเกี่ยวกับการติดต่อเจ้าหน้าที่ การตรวจสอบความเสียหาย และขั้นตอนการแก้ไข
"""

# ข้อความคู่มือสำหรับฟีเจอร์ ระวังช้าง
WATCH_ELEPHANT_MANUAL = """
# คู่มือการใช้งานฟีเจอร์ "ระวังช้าง"

**ฟังก์ชันหลัก:**
- **ตรวจสอบเส้นทางปลอดภัยจากช้าง**: กดปุ่ม "ระวังช้าง" เพื่อรับลิงก์ตรวจสอบเส้นทางที่ปลอดภัยจากช้าง

**วิธีใช้งาน:**
1. เปิดแอป Line และไปที่แชทบอทที่เกี่ยวข้อง
2. กดปุ่ม **"ระวังช้าง"** ที่เมนูหลักหรือแถบด้านล่าง
3. บอทจะส่งลิงก์ไปยังแผนที่หรือเว็บไซต์ที่ให้ข้อมูลเส้นทางที่ปลอดภัย
4. คลิกลิงก์เพื่อตรวจสอบและเลือกเส้นทางที่เหมาะสม

**ตัวอย่างการใช้งาน:**
- **สถานการณ์**: กำลังวางแผนเดินทางไปพื้นที่มีช้างบ่อย
- **การดำเนินการ**: กด "ระวังช้าง"
- **คำตอบจากบอท**: ส่งลิงก์แผนที่เส้นทางที่แนะนำให้หลีกเลี่ยงพื้นที่ช้าง เช่น Google Maps หรือเว็บไซต์ท้องถิ่น
"""

# ฟังก์ชันสำหรับตรวจสอบและตอบกลับข้อความคู่มือ
def get_manual_response(user_message: str) -> str:
    user_message = user_message.strip().lower()
    if user_message == "emergency" or user_message == "คู่มือการใช้งาน":
        return EMERGENCY_MANUAL
    elif user_message == "ระวังช้าง":
        return WATCH_ELEPHANT_MANUAL
    else:
        return None

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
            # ตรวจสอบว่าผู้ใช้ต้องการคู่มือการใช้งานหรือไม่
            manual_response = get_manual_response(user_message)
            if manual_response:
                reply = manual_response
            else:
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
