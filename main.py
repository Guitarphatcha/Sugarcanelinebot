from linebot.models import FlexSendMessage, MessageEvent, TextSendMessage
from linebot.exceptions import InvalidSignatureError
from linebot.aiohttp_async_http_client import AiohttpAsyncHttpClient
from linebot import AsyncLineBotApi, WebhookParser
from fastapi import Request, FastAPI, HTTPException
import logging
import cv2
import re
import aiohttp
import numpy as np
import os
import sys
import tempfile
from dotenv import load_dotenv
# from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
# import tensorflow as tf
import asyncio
import tflite_runtime.interpreter as tflite
from rapidfuzz import process, fuzz
from pythainlp.tokenize import word_tokenize

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

channel_secret = os.getenv('ChannelSecret')
channel_access_token = os.getenv('ChannelAccessToken')

if not channel_secret or not channel_access_token:
    logging.error("Missing LINE channel credentials")
    sys.exit(1)

try:
#    model = load_model('model_final_1.h5')
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter .allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    sys.exit(1)

model_lock = asyncio.Lock()
# ข้อมูลโรคใบอ้อย (รวมคลาส Unknown)
disease_info = {
    "Healthy": """
🌿 อ้อยสุขภาพดี

✅ ลักษณะของอ้อยสุขภาพดี
- ใบมีสีเขียวสดใส สม่ำเสมอ  
- ลำต้นแข็งแรง ไม่มีรอยแตกหรือจุดผิดปกติ  
- การเจริญเติบโตเป็นไปตามปกติ  

🛡️ วิธีดูแลรักษา
- ให้น้ำและปุ๋ยอย่างเหมาะสม  
- ตรวจสอบแปลงอ้อยเป็นประจำ  
- กำจัดวัชพืชและศัตรูพืช  
    """,
    "Mosaic": """
🌿 โรคใบด่างอ้อย 

⚠️ อาการของโรค
- ใบอ้อยมีลายด่างสีเขียวอ่อนสลับเขียวเข้มเป็นทาง  
- ใบอาจบิดเบี้ยวหรือเจริญเติบโตช้ากว่าปกติ  
- อ้อยที่เป็นโรครุนแรงอาจแคระแกร็น และให้ผลผลิตต่ำ  

🔎 สาเหตุที่เกิดโรค 
- เกิดจาก เชื้อไวรัส Sugarcane Mosaic Virus (SCMV)
- ติดต่อผ่าน เพลี้ยอ่อน (Aphids) ที่เป็นพาหะ  
- แพร่กระจายผ่านพันธุ์อ้อยที่ติดเชื้อ  

🛡️ วิธีป้องกันโรค  
- ใช้พันธุ์อ้อยต้านทานโรค เช่น KK3, LK92-11 
- ควบคุมเพลี้ยอ่อนโดยใช้สารชีวภัณฑ์  
- หลีกเลี่ยงการใช้พันธุ์อ้อยจากแหล่งที่มีโรคระบาด  

💊 สารเคมีที่ใช้
- อิมิดาคลอพริด (Imidacloprid)
- เชื้อรา Beauveria bassiana  

🌱 ระยะแรกเริ่มของโรค
- พบอาการในช่วง ต้นอ้อยอายุ 1-2 เดือน
- ใบเริ่มมีจุดด่างเป็นแถบคล้ายลายโมเสค  
    """,
    "Rust": """
🌿 โรคราสนิม (Rust Disease)

⚠️ อาการของโรค  
- มีจุดสีน้ำตาลหรือส้มกระจายทั่วใบ  
- ใบอ้อยเหลืองและแห้งก่อนเวลา  
- พบเชื้อราละอองสีน้ำตาลคล้ายสนิม  

🔎 สาเหตุที่เกิดโรค
- เกิดจากเชื้อรา Puccinia kuehnii 
- แพร่กระจายผ่านลมและน้ำฝน 
 
🛡️ วิธีป้องกันโรค 
- ใช้พันธุ์อ้อยต้านทาน เช่น K88-92, Suphanburi 50
- หลีกเลี่ยงปลูกอ้อยแน่นเกินไป เพื่อให้อากาศถ่ายเท  

💊 สารเคมีที่ใช้ 
- ไตรฟอกซี่สโตรบิน (Trifloxystrobin) 
- โพรพิเนบ (Propineb)

🌱 ระยะแรกเริ่มของโรค
- พบอาการในช่วง อ้อยอายุ 2-3 เดือน  
- เริ่มจากจุดเล็ก ๆ ก่อนขยายเป็นแผลสีน้ำตาล  
    """,
    "RedRot": """
🌿 โรคเหี่ยวเน่าแดง (Red Rot Disease)

⚠️ อาการของโรค
- ใบเหี่ยวแห้ง และมีสีเหลือง  
- ลำต้นอ้อยมีรอยแตก และเนื้อในเป็นสีแดง-ดำ  
- มีกลิ่นเน่าเหม็น  

🔎 สาเหตุที่เกิดโรค
- เกิดจากเชื้อรา Colletotrichum falcatum 
- เชื้อแพร่กระจายในดินและเข้าทำลายทางราก  

🛡️ วิธีป้องกันโรค
- ใช้พันธุ์อ้อยทนโรค เช่น KPS94-13, U-Thong 2
- ปลูกอ้อยหมุนเวียนกับพืชอื่นเพื่อลดการสะสมของเชื้อ  

💊 สารเคมีที่ใช้  
- แมนโคเซบ (Mancozeb) 
- คาร์เบนดาซิม (Carbendazim) 

🌱 ระยะแรกเริ่มของโรค 
- พบอาการใน อ้อยอายุ 3-5 เดือน 
- ใบเริ่มเหี่ยวเฉา และลำต้นอ้อยเริ่มเป็นรอยแตก  
    """,
    "Yellow": """
🌿 โรคใบไหม้ (Leaf Scald Disease)

⚠️ อาการของโรค 
- ใบมีรอยไหม้เป็นทางยาวคล้ายถูกน้ำร้อนลวก  
- อ้อยเจริญเติบโตช้า และอาจตายต้น  

🔎 สาเหตุที่เกิดโรค
- เกิดจากเชื้อแบคทีเรีย Xanthomonas albilineans
- ติดต่อผ่านพันธุ์อ้อยและแมลงพาหะ  

🛡️ วิธีป้องกันโรค 
- ใช้พันธุ์อ้อยที่ทนทาน เช่น **Suphanburi 7**  
- แช่ท่อนพันธุ์ในน้ำร้อน 50°C นาน 30 นาที เพื่อลดเชื้อ  

💊 สารเคมีที่ใช้ 
- ไม่มีสารเคมีรักษาโดยตรง ต้องใช้การจัดการแปลงแทน  

🌱 ระยะแรกเริ่มของโรค 
- พบอาการเมื่ออ้อยเริ่มแตกใบใหม่  
- ใบมีรอยซีดขาวก่อนเปลี่ยนเป็นสีน้ำตาลไหม้  
    """,
    "Unknown": """
⚠️ ไม่สามารถระบุโรคได้
สาเหตุอาจเกิดจาก:
1. ภาพไม่ใช่ใบอ้อย
2. โรคยังไม่ถูกบันทึกในระบบ
3. ภาพถ่ายไม่ชัดเจน

กรุณาส่งภาพใบอ้อยที่มีอาการชัดเจน
ตัวอย่างภาพที่เหมาะสม:
- ภาพใบอ้อยเดี่ยวในแสงธรรมชาติ
- ภาพแสดงอาการโรคชัดเจน
- ภาพความละเอียดสูงกว่า 300x300 พิกเซล
    """
}

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"])  

disease_display_names = {
    "Healthy": "อ้อยสุขภาพดี",
    "Mosaic": "โรคใบด่าง",
    "Rust": "โรคราสนิม",
    "RedRot": "โรคเหี่ยวเน่าแดง",
    "Yellow": "โรคใบไหม้"
}
disease_keywords = {
    "Mosaic": ["ใบด่าง", "ด่าง","โรคใบด่าง","โรคใบด่าอ้อย","mosaic","Mosaic","โรคใบด่า","ใบดาง","ดาง","ด่างอ่อย", "ใบอ้อยมีลายด่างสีเขียวอ่อนสลับเขียวเข้มเป็นทาง", "ใบอาจบิดเบี้ยวหรือเจริญเติบโตช้ากว่าปกติ", "อ้อยที่เป็นโรครุนแรงอาจแคระแกร็น และให้ผลผลิตต่ำ"],
    "Rust": ["โรคราสนิม", "สนิม","ราสนิม","โรคราสนิมอ้อย","Rust","rust","ราสนิมอ้อย","สนิมอ้อย","จุดสนิม", "มีจุดสีน้ำตาลหรือส้มกระจายทั่วใบ", "ใบอ้อยเหลืองและแห้งก่อนเวลา", "พบเชื้อราละอองสีน้ำตาลคล้ายสนิม"],
    "RedRot": ["โรคเหี่ยวเน่าแดง", "เหี่ยวเน่าแดง", "เน่าแดง", "เหี่ยวเน่า", "เน่า", "เหี่ยว", "เน่าแดง", "เหี่ยวเน่าแดงอ้อย" , "เน่าอ้อย","RedRot","redrot", "ใบเหี่ยวแห้ง และมีสีเหลือง", "ลำต้นอ้อยมีรอยแตก และเนื้อในเป็นสีแดง-ดำ", "มีกลิ่นเหม็นเน่า"],
    "Yellow": ["โรคใบไหม้", "ใบไหม้", "ไหม้","โรคใบไหม้อ้อย","Yellow","yellow","ใบไหม้อ้อย","ไหม้อ้อย", "ใบมีรอยไหม้เป็นทางยาวคล้ายถูกน้ำร้อนลวก", "อ้อยเจริญเติบโตช้า และอาจตายต้น"],
    "Healthy": ["อ้อยสุขภาพดี", "สุขภาพดี", "อ้อยสุขภาพ", "Healthy","healthy","สุขภาพดีอ้อย","อ้อยสุขภาพ","ใบปกติ","อ้อยไม่เป็นโรค", "ใบมีสีเขียวสดใส สม่ำเสมอ", "ลำต้นแข็งแรง ไม่มีรอยแตกหรือจุดผิดปกติ", "การเจริญเติบโตเป็นไปตามปกติ"]
}
app = FastAPI()
session = aiohttp.ClientSession()
line_bot_api = AsyncLineBotApi(channel_access_token, AiohttpAsyncHttpClient(session))
parser = WebhookParser(channel_secret)

# ตั้งค่า LabelEncoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["Healthy", "Mosaic", "RedRot", "Rust", "Yellow", "Unknown"])

def remove_background(img):
    """ตัดพื้นหลังใบอ้อยและเปลี่ยนเป็นสีดำ"""
    try:
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # กำหนดช่วงสีเขียวของใบอ้อย
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([95, 255, 255])
        
        # สร้าง mask ด้วย inRange
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # ปรับปรุง mask ด้วย morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # หา contour ที่ใหญ่ที่สุด
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # สร้างภาพใหม่โดยเปลี่ยนพื้นหลังเป็นสีดำ
        result = img.copy()
        result[mask == 0] = [0, 0, 0]  # ตั้งค่าพื้นหลังเป็นสีดำ
        
        return result, mask
    except Exception as e:
        logging.error(f"Background removal error: {e}")
        return img, None

def check_image_characteristics(img):
    """ตรวจสอบคุณสมบัติพื้นฐานของภาพ (ปรับปรุงใหม่)"""
    try:
        # ตรวจสอบขนาดภาพ
        if img.shape[0] < 100 or img.shape[1] < 100:
            return False
        
        # ตัดพื้นหลังและรับ mask
        _, mask = remove_background(img)
        
        if mask is None:
            return False
        
        # ตรวจสอบพื้นที่สีเขียว
        green_pixels = cv2.countNonZero(mask)
        green_ratio = green_pixels / (img.shape[0] * img.shape[1])
        
        return green_ratio > 0.2
    except Exception as e:
        logging.error(f"Image check error: {e}")
        return False

def preprocess_image(image_data):
    """ปรับปรุงการพรีโพรเซสภาพ"""
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # ตรวจสอบคุณสมบัติภาพ
        if not check_image_characteristics(img):
            return None, "ภาพไม่ตรงเงื่อนไขการวิเคราะห์"

        # ปรับขนาดและสี
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        return np.expand_dims(img, axis=0), None
    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        return None, "ข้อผิดพลาดในการประมวลผลภาพ"

def calculate_ood_score(predictions):
    """คำนวณคะแนน Out-of-Distribution"""
    epsilon = 1e-10
    entropy = -np.sum(predictions * np.log(predictions + epsilon))
    return entropy

def detect_mosaic_pattern(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        abs_sobel_y = np.absolute(sobel_y)
        mean_sobel = np.mean(abs_sobel_y)
        return mean_sobel > 20 
    except Exception as e:
        logging.error(f'Mosaic pattern detection error: {e}')
        return False

async def classify_image(image_data):
    """กระบวนการจำแนกภาพแบบปรับปรุง"""
    try:
        # พรีโพรเซสและตรวจสอบภาพ
        processed_image, error_msg = preprocess_image(image_data)
        if error_msg:
            return error_msg

        async with model_lock:
            interpreter.set_tensor(input_details[0]['index'], processed_image.astype(np.float32))
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]
            #prediction = model.predict(processed_image)[0]
       
        ood_score = calculate_ood_score(prediction)
        confidence = np.max(prediction) * 100
        if ood_score > 1.2 or confidence < 66:
            return disease_info["Unknown"]
        predicted_class = np.argmax(prediction)
        disease_name = label_encoder.inverse_transform([predicted_class])[0]
        
        return (
            f"ผลการวินิจฉัย: {disease_name}\n"
            f"ความแม่นยำ: {confidence:.2f}%\n\n"
            f"{disease_info.get(disease_name, disease_info['Unknown'])}"
        )
    except Exception as e:
        logging.error(f"Classification error: {e}")
        return disease_info["Unknown"]
    
def tokenize(text):
	return ' ' .join(word_tokenize(text))

def find_matching_disease_rapidfuzz(user_input, threshold=75):
    tokenize_input = tokenize(user_input)
    best_match = None 
    best_score = 0

    for disease, keywords in disease_keywords.items():
        tokenized_keywords = [tokenize(k) for k in keywords]
        # match, score, _ = process.extractOne(
        #     tokenize_input,
        #     tokenized_keywords
        #     scorer=fuzz.partial_ratio
        # )
        # if score >= threshold and score > best_score:
        #     best_match = disease
        #     best_score = score 
        result = process.extractOne(
            tokenize_input,
            tokenized_keywords,
            scorer=fuzz.partial_ratio
        )
        if result:
            _, score, _ = result
            if score >= threshold and score > best_score:
                best_match = disease
                best_score = score
    return best_match

@app.post("/callback")
async def handle_callback(request: Request):
    signature = request.headers.get('X-Line-Signature', '')
    body = await request.body()
    
    try:
        events = parser.parse(body.decode(), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue

        if event.message.type == "text":
            msg = event.message.text.strip()
            response = None 

            if msg in ["สวัสดี","สวัสดีครับ", "สวัสดีค่ะ","ดีจ้า","ดีครับ","ดีค่ะ"]:
                response = "น้องอ้อยใจสวัสดีค่ะ 💚\nสามารถส่งรูปภาพใบอ้อยเพื่อทำนายโรคหรือส่งคำถามเกี่ยวกับโรคอ้อยมาได้เลยค่ะ"
            # if not response:
            #     matched_disease = None
            #     for disease, keywords in disease_keywords.items():
            #         if any(keyword in msg for keyword in keywords):
            #             matched_disease = disease
            #             break
                
            #     if matched_disease:
            #         response = disease_info.get(matched_disease, disease_info["Unknown"])
            if not response:
                list_keywords = [
                    "โรคอะไรบ้าง", "มีโรคอะไร", "โรคมีอะไร", "โรคอ้อย","มีโรคไรบ้าง",
                    "โรคอ้อยที่พบบ่อยในประเทศไทย", "โรคอ้อยที่พบบ่อย", "โรคอ่อย", 
                    "อ้อย", "โรค", "อ้อยที่เป็นโรค", "โรคอ้อยที่อันตราย", 
                    
                ]
                if any(kw in msg for kw in list_keywords):
                    diseases = [disease_display_names[d] for d in disease_info.keys() if d != "Unknown"]
                    disease_list = "\n- ".join(diseases)
                    response = f"📜 โรคในอ้อยที่สามารถวิเคราะห์ได้มีดังนี้:\n- {disease_list}\n\n🖼️ สามารถส่งรูปภาพใบอ้อยเพื่อวิเคราะห์โรคได้ค่ะ"

            if not response:
                if any(kw in msg for kw in ["โรค", "อ้อย"]):
                    response = "กรุณาส่งภาพใบอ้อยเพื่อวิเคราะห์โรค"

            if not response:
                response = "ระบบนี้ใช้สำหรับวิเคราะห์โรคอ้อยเท่านั้น"
            if not response:
                match_disease = find_matching_disease_rapidfuzz(msg)
                if match_disease:
                    response = f"🦠 ข้อมูลเกี่ยวกับ {disease_display_names[match_disease]}:\n{disease_info.get(match_disease, disease_info['Unknown'])}"
            await line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=response) )
        
        elif event.message.type == "image":
            try:
                message_content = await line_bot_api.get_message_content(event.message.id)
                image_data = b''
                
                async for chunk in message_content.iter_content():
                    image_data += chunk

                result = await classify_image(image_data)
                
                await line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text=result)
                )
                
            except Exception as e:
                logging.error(f"Image processing error: {e}")
                await line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text=disease_info["Unknown"])
                )

    return "OK"
