import requests
from io import BytesIO
from PIL import Image
import uvicorn
import numpy as np
import faiss

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent,
    TextMessage,
    ImageMessage,
    FlexSendMessage,
    BubbleContainer,
    BoxComponent,
    TextComponent,
    ButtonComponent,
    URIAction,
    CarouselContainer
)
from linebot.exceptions import InvalidSignatureError
from sentence_transformers import SentenceTransformer
from typing import Dict
from contextlib import asynccontextmanager

# ไม่จำเป็นต้องใช้ dotenv แล้ว เพราะจะไม่โหลดจาก .env
# from dotenv import load_dotenv  

app = FastAPI()

# กำหนดค่าตัวแปรโดยตรงแทนการใช้ os.getenv
ACCESS_TOKEN = "RMuXBCLD7tGSbkGgdELH7Vz9+Qz0YhqCIeKBhpMdKvOVii7W2L9rNpAHjYGigFN4ORLknMxhuWJYKIX3uLrY1BUg7E3Bk0v3Fmc5ZIC53d8fOdvIMyZQ6EdaOS0a6kejeqcX/dRFI/JfiFJr5mdwZgdB04t89/1O/w1cDnyilFU="
CHANNEL_SECRET = "175149695b4d312eabb9df4b7e3e7a95"
GEMINI_API_KEY = "AIzaSyBfkFZ8DCBb57CwW8WIwqSbUTB3fyIfw6g"

# ตรวจสอบว่าตัวแปรถูกกำหนดค่าไว้แล้ว
if not ACCESS_TOKEN or not CHANNEL_SECRET or not GEMINI_API_KEY:
    raise ValueError("Please set the LINE_ACCESS_TOKEN, LINE_CHANNEL_SECRET, and GEMINI_API_KEY environment variables.")

line_bot_api = LineBotApi(ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

class GeminiRAGSystem:
    def __init__(self, 
                 json_db_path: str, 
                 gemini_api_key: str, 
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        # การเชื่อมต่อ และตั้งค่าข้อมูลเพื่อเรียกใช้งาน Gemini
        genai.configure(api_key=gemini_api_key)
        
        # ประกาศโมเดลที่ใช้งาน
        self.generation_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # ข้อมูล JSON ที่ใช้เก็บข้อมูล
        self.json_db_path = json_db_path
        
        # โมเดลที่ใช้ในการสร้างเวกเตอร์ของข้อความ
        self.embedding_model = SentenceTransformer(embedding_model)
        
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
    
    def generate_response(self, query: str):
        """Generate response using Gemini and retrieved documents"""
        # Retrieve ข้อมูลจากฐานข้อมูล
        retrieved_docs = self.retrieve_documents(query)
        
        # เตรียมข้อมูลเพื่อใช้ในการสร้างบริบท
        context = "\n\n".join(retrieved_docs)
        
        # สร้าง Prompt เพื่อใช้ในการสร้างคำตอบ
        full_prompt = f"""You are an AI assistant. 
Use the following context to answer the question precisely:

Context:
{context}

Question: {query}

Provide a detailed and informative response based on the context in Thai 
but if the response is not about the context just ignore and answer in a natural way."""
        
        # คำตอบจาก Gemini
        try:
            response = self.generation_model.generate_content(full_prompt)
            return response.text, full_prompt
        except Exception as e:
            return f"Error generating response: {str(e)}", str(e)
    
    def generate_concise_response(self, query: str):
        """Generate a concise response using Gemini without RAG context"""
        # สร้าง Prompt สำหรับการตอบคำถามโดยไม่ใช้บริบทจาก RAG
        concise_prompt = f"""You are an AI assistant. 
Provide a concise and to-the-point answer to the following question in Thai:

Question: {query}"""
        
        try:
            response = self.generation_model.generate_content(
                concise_prompt,
                generation_config={
                    "max_output_tokens": 100,  # จำกัดความยาวของคำตอบ
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "top_k": 40
                }
            )
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def process_image_query(self, 
                            image_content: bytes, 
                            query: str,
                            use_rag: bool = True,
                            top_k_docs: int = 3) -> Dict:
        """
        Process image-based query with optional RAG enhancement
        
        Args:
            image_content (bytes): Content of the image
            query (str): Query about the image
            use_rag (bool): Whether to use RAG for context
            top_k_docs (int): Number of documents to retrieve
        
        Returns:
            Generated response about the image
        """
        # เปิดภาพจากข้อมูลที่ส่งมา
        image = Image.open(BytesIO(image_content))

        # สร้างคำอธิบายของภาพ
        initial_description = self.generation_model.generate_content(
            ["Provide a detailed, objective description of this image", image],
            generation_config={
                "max_output_tokens": 256,
                "temperature": 0.4,
                "top_p": 0.9,
                "top_k": 8
            }
        ).text
        
        # สำหรับการใช้งาน RAG 
        context = ""
        if use_rag:
            # นำคำอธิบายภาพไปใช้ในการค้นหาเอกสารที่เกี่ยวข้องใน JSON
            retrieved_docs = self.retrieve_documents(initial_description, top_k_docs)
            
            # นำข้อมูลที่ได้จากการค้นหามาใช้ในการสร้างบริบท
            context = "\n\n".join(retrieved_docs)
        
        # สร้าง Prompt สำหรับการสร้างคำตอบ
        enhanced_prompt = f"""Image Description:
{initial_description}

Context from Knowledge Base:
{context}

User Query: {query}

Based on the image description and the contextual information from our knowledge base, 
provide a comprehensive and insightful response to the query. 
If the context does not directly relate to the image, focus on the image description 
and your visual analysis in Thai."""
        
        # สร้างคำตอบจาก Gemini
        try:
            response = self.generation_model.generate_content(
                [enhanced_prompt, image],
                generation_config={
                    "max_output_tokens": 256,
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "top_k": 8
                }
            )
            
            return {
                "final_response": response.text,
            }
        except Exception as e:
            return {
                "error": f"Error generating response: {str(e)}",
                "image_description": initial_description
            }
    
    def clear_database(self):
        """Clear database and save to JSON file"""
        self.database = {
            'documents': [],
            'embeddings': [],
            'metadata': []
        }
        self.save_database()

# สร้าง Object สำหรับใช้งาน Gemini
gemini = GeminiRAGSystem(
    json_db_path="gemini_rag_database.json", 
    gemini_api_key=GEMINI_API_KEY
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ข้อมูลตัวอย่างที่ใช้สำหรับ Gemini
    sample_documents = [
        # เพิ่มเอกสารตัวอย่างที่ต้องการ
    ]
    
    # เพิ่มข้อมูลตัวอย่างลงใน Gemini
    for doc in sample_documents:
        gemini.add_document(doc)
        
    yield

    # ลบข้อมูลที่ใช้ในการทดสอบออกจาก Gemini
    gemini.clear_database()

app = FastAPI(lifespan=lifespan)

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

def get_manual_response(user_message: str) -> str:
    user_message = user_message.strip().lower()
    
    # ตรวจสอบคำถามทั่วไป
    if user_message in ["emergency", "คู่มือการใช้งาน"]:
        return EMERGENCY_MANUAL
    elif user_message in ["emergency เกิดเหตุฉุกเฉินทำยังไง", "มีเหตุร้ายใกล้ตัว"]:
        return WATCH_ELEPHANT_MANUAL
    elif user_message == "ตรวจสอบช้างก่อนเดินทาง":
        return CHECK_ELEPHANT_MANUAL
    elif user_message in ["ติดต่อเจ้าหน้าที่", "contact officer"]:
        return OFFICER_MANUAL
    
    # เพิ่มการจับคีย์เวิร์ดสำหรับคำถามเกี่ยวกับช้าง
    elephant_keywords = ['ช้าง', 'พบช้าง', 'เจอช้าง', 'วิธีจัดการกับช้าง']
    if any(keyword in user_message for keyword in elephant_keywords):
        return WATCH_ELEPHANT_MANUAL
    
    return None

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

@handler.add(MessageEvent, message=(TextMessage, ImageMessage))
def handle_message(event: MessageEvent):
    if isinstance(event.message, TextMessage):
        user_message = event.message.text
        manual_response = get_manual_response(user_message)
        
        if manual_response:
            # ถ้าคำถามเป็นคำถามที่เตรียมไว้ล่วงหน้า ใช้ข้อความจาก manual
            reply = create_flex_message(manual_response)
        else:
            # ถ้าไม่ใช่คำถามที่เตรียมไว้ ให้ใช้ Gemini ตอบกระชับ
            gemini_response = gemini.generate_concise_response(user_message)
            
            if "Error" not in gemini_response:
                reply = create_flex_message(gemini_response)
            else:
                # หากเกิดข้อผิดพลาดในการสร้างคำตอบด้วย Gemini ให้ใช้การตอบกลับเริ่มต้น
                retrieved_docs = gemini.retrieve_documents(user_message, top_k=3)
                
                if retrieved_docs:
                    texts = ["ดูข้อมูลเพิ่มเติมที่นี่" if "http" in doc else doc for doc in retrieved_docs]
                    reply = create_carousel_message(texts)
                else:
                    reply = create_flex_message("ขออภัย ฉันไม่เข้าใจคำถามของคุณ กรุณาลองใหม่อีกครั้ง")

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
                # ส่งข้อมูลภาพไปยัง Gemini เพื่อทำการประมวลผล
                gemini_response = gemini.process_image_query(
                    image_content=response.content,
                    query="อธิบายภาพนี้ให้ละเอียด", 
                    use_rag=True
                )
                # นำข้อมูลที่ได้จาก Gemini มาใช้งาน
                response_text = gemini_response.get('final_response', "ขออภัย ฉันไม่สามารถประมวลผลรูปภาพได้ในขณะนี้")
                message = response_text
                
        except Exception:
            message = "เกิดข้อผิดพลาด, กรุณาลองใหม่อีกครั้ง🙏🏻"
            
        reply = create_flex_message(message)
    
    line_bot_api.reply_message(
        event.reply_token,
        [reply]
    )

@app.get('/test-message')
async def test_message_gemini(text: str):
    """
    Debug message from Gemini
    """
    response, prompt = gemini.generate_response(text)

    return {
        "gemini_answer": response,
        "full_prompt": prompt
    }

@app.post('/image-query')
async def image_query(
    file: UploadFile = File(...), 
    query: str = Form("อธิบายภาพนี้ให้ละเอียด"),
    use_rag: bool = Form(True)
):
    if file.size > 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image size too large")
    
    contents = await file.read()
    
    # ส่งข้อมูลภาพไปยัง Gemini เพื่อทำการประมวลผล
    image_response = gemini.process_image_query(
        image_content=contents,
        query=query,
        use_rag=use_rag
    )
    
    return image_response

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, host="0.0.0.0")
