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
    sample_documents = [EMERGENCY_MANUAL, FIRST_AID_MANUAL, HEALTH_CHECK_MANUAL, MEDICAL_CONTACTS]
    for doc in sample_documents:
        rag.add_document(doc)
    yield
    rag.clear_database()

app = FastAPI(lifespan=lifespan)

usage_promotion = """แนะนำการใช้งาน
1. ใช้งานง่ายเพียงกรอกข้อมูลเกี่ยวกับอาหารที่ต้องการหรือเลือกจากตัวเลือกที่แนะนำ
2. ระบบจะวิเคราะห์ข้อมูลและให้คำแนะนำที่เหมาะสมตามโภชนาการและไลฟ์สไตล์ของคุณ
3. สามารถดูเมนูแนะนำประจำวัน รวมถึงเคล็ดลับด้านสุขภาพและการออกกำลังกายได้
4. มีระบบบันทึกเมนูอาหารที่คุณเคยเลือก เพื่อช่วยให้คุณวางแผนมื้ออาหารในอนาคต
"""

Recommended_food = """แนะนำอาหาร
1. อาหารเพื่อสุขภาพ: แนะนำเมนูที่มีคุณค่าทางโภชนาการสูง เหมาะกับการดูแลสุขภาพ
2. อาหารลดน้ำหนัก: เมนูที่มีแคลอรีต่ำแต่ยังคงให้พลังงานเพียงพอในแต่ละวัน
3. อาหารเพิ่มพลังงาน: เหมาะสำหรับผู้ที่ต้องการพลังงานเพิ่ม เช่น นักกีฬา หรือคนที่ต้องใช้แรงงานหนัก
4. อาหารสำหรับผู้สูงอายุ: เมนูที่มีสารอาหารครบถ้วนและย่อยง่าย
5. อาหารตามเทศกาล: แนะนำเมนูพิเศษตามเทศกาล เช่น ตรุษจีน สงกรานต์ หรือปีใหม่
"""

care_health = """ดูแลสุขภาพ
1. การรับประทานอาหารที่สมดุล: ควรบริโภคอาหารให้ครบ 5 หมู่ และลดอาหารที่มีไขมันสูง
2. ดื่มน้ำให้เพียงพอ: ควรดื่มน้ำวันละ 8 แก้วขึ้นไป เพื่อช่วยให้ร่างกายทำงานได้ดี
3. การนอนหลับที่เพียงพอ: ควรนอนอย่างน้อย 7-8 ชั่วโมงต่อคืน เพื่อให้ร่างกายได้พักฟื้น
4. หลีกเลี่ยงอาหารที่มีโซเดียมสูง: ลดการบริโภคอาหารแปรรูปเพื่อลดความเสี่ยงของโรคความดันโลหิตสูง
5. ออกกำลังกายสม่ำเสมอ: ควรออกกำลังกายอย่างน้อยวันละ 30 นาที เพื่อช่วยเสริมสร้างสุขภาพที่แข็งแรง
"""

exercise = """ออกกำลังกาย
1. คาร์ดิโอ: วิ่ง เดินเร็ว ปั่นจักรยาน หรือกระโดดเชือก ช่วยเผาผลาญแคลอรีและเพิ่มความแข็งแรงของหัวใจ
2. เวทเทรนนิ่ง: ยกน้ำหนักหรือใช้แรงต้านเพื่อสร้างกล้ามเนื้อและเพิ่มความแข็งแรง
3. โยคะและพิลาทิส: ช่วยเพิ่มความยืดหยุ่น ลดความเครียด และปรับสมดุลร่างกาย
4. HIIT (High-Intensity Interval Training): การออกกำลังกายแบบหนักสลับเบา ช่วยเผาผลาญไขมันได้เร็วขึ้น
5. ออกกำลังกายกลางแจ้ง: เช่น วิ่งในสวน ขี่จักรยาน หรือเล่นกีฬากลางแจ้งเพื่อรับอากาศบริสุทธิ์
"""


def get_manual_response(user_message: str) -> str:
    user_message = user_message.strip().lower()
    manuals = {
        "แนะนำการใช้งาน": usage_promotion,
        "แนะนำอาหาร": Recommended_food,
        "ดูแลสุขภาพ": care_health,
        "ออกกำลังกาย": exercise,
    }
    return manuals.get(user_message)

def create_bubble_container(text: str) -> BubbleContainer:
    return BubbleContainer(
        header=BoxComponent(
            layout='vertical',
            contents=[
                TextComponent(
                    text="HealthMate",
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
            # Check if the message matches predefined keywords exactly
            relevant_to_rag = any(user_message.strip().lower() == phrase for phrase in ['ฉุกเฉิน', 'ช้าง', 'เจ้าหน้าที่'])
            
            if relevant_to_rag:
                retrieved_docs = rag.retrieve_documents(user_message, top_k=3)
                if retrieved_docs:
                    texts = ["ดูข้อมูลเพิ่มเติมที่นี่" if "http" in doc else doc for doc in retrieved_docs]
                    reply = create_carousel_message(texts)
                else:
                    # Use Gemini if RAG has no results
                    gemini_response = model.generate_content(user_message + " ให้สรุปสั้นๆใน 2-3 บรรทัด")
                    reply = create_flex_message(gemini_response.text.strip().split("\n")[:3])
            else:
                # Use Gemini for non-matching queries
                gemini_response = model.generate_content(user_message + " ให้สรุปสั้นๆใน 2-3 บรรทัด")
                reply = create_flex_message("\n".join(gemini_response.text.strip().split("\n")[:3]))

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
                    gemini_response = model.generate_content("อธิบายรูปภาพนี้ ให้สรุปสั้นๆใน 2-3 บรรทัด")
                    message = "\n".join(gemini_response.text.strip().split("\n")[:3])
                except Exception:
                    message = "ขณะนี้ระบบไม่สามารถประมวลผลรูปภาพได้ กรุณาสอบถามด้วยข้อความแทนค่ะ 🙏🏻"
                
        except Exception:
            message = "เกิดข้อผิดพลาด, กรุณาลองใหม่อีกครั้ง🙏🏻"
            
        reply = create_flex_message(message)
        line_bot_api.reply_message(event.reply_token, [reply])



if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, host="0.0.0.0")
