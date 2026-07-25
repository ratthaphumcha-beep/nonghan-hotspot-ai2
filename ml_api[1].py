# ก่อนรัน:
# 1) pip install fastapi uvicorn scikit-learn pandas numpy httpx python-dotenv
# 2) สร้างไฟล์ config.env (ในโฟลเดอร์เดียวกับไฟล์นี้) ด้วย Notepad ธรรมดาได้เลย แล้วใส่บรรทัดเดียว:
#    GEMINI_API_KEY=ใส่คีย์จริงของคุณตรงนี้
#    (ห้าม commit ไฟล์ config.env ขึ้น GitHub เด็ดขาด — ควรมี config.env ใน .gitignore ด้วย)

import os
import time
import json
import hashlib
import secrets
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="config.env")
except ImportError:
    pass  # ถ้าไม่ได้ลง python-dotenv ก็ยังอ่าน env var จากระบบได้ตามปกติ

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nonghan-ml-api")

# ---------------------------------------------------------
# 0. ระบบล็อกอินแอดมิน (สำหรับกรอกข้อมูลมือกรณี AI/เซนเซอร์มีปัญหา)
# ---------------------------------------------------------
# หมายเหตุความปลอดภัย: นี่คือการยืนยันตัวตนแบบง่ายที่เหมาะกับ demo แฮกกาธอน
# (เก็บรหัสผ่านแบบ hash+salt ในไฟล์ JSON บนเครื่อง, session token อยู่ใน memory)
# ถ้าจะใช้งานจริง/deploy สาธารณะ ควรย้ายไปใช้ระบบ auth มาตรฐาน (เช่น OAuth),
# เก็บข้อมูลใน database จริง, บังคับ HTTPS, และเพิ่ม rate-limit กันการเดารหัส
ADMINS_FILE = Path("admins.json")
OVERRIDES_FILE = Path("overrides.json")
SESSIONS: dict[str, dict] = {}          # token -> {"username":..., "expires_at":...}
SESSION_TTL_SECONDS = 8 * 3600          # session หมดอายุใน 8 ชม.
DEFAULT_ADMIN_USERNAME = "อ้ายมาสี่คน"
DEFAULT_ADMIN_PASSWORD = "012301230123"


def hash_password(password: str, salt: str | None = None) -> str:
    if salt is None:
        salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return f"{salt}${pwd_hash.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, hash_hex = stored.split("$")
    except ValueError:
        return False
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return secrets.compare_digest(pwd_hash.hex(), hash_hex)


def load_admins() -> dict:
    """คืนค่า dict: username -> {"hash": "...", "must_change_password": bool}
    บัญชีเริ่มต้นถูกบังคับให้เปลี่ยนรหัสผ่านตอนล็อกอินครั้งแรก เพราะรหัสผ่าน
    เริ่มต้นถูกเขียนไว้ตรงๆ ใน source code (เห็นได้ทุกคนที่เข้าถึง repo นี้)"""
    if not ADMINS_FILE.exists():
        default_admins = {
            DEFAULT_ADMIN_USERNAME: {
                "hash": hash_password(DEFAULT_ADMIN_PASSWORD),
                "must_change_password": True,
            }
        }
        save_admins(default_admins)
        logger.info("สร้างบัญชีแอดมินเริ่มต้น: %s (ต้องเปลี่ยนรหัสผ่านตอนล็อกอินครั้งแรก)", DEFAULT_ADMIN_USERNAME)
        return default_admins

    raw = json.loads(ADMINS_FILE.read_text(encoding="utf-8"))
    # normalize เผื่อไฟล์ admins.json เก่าที่เก็บเป็น username -> hash string ตรงๆ (schema เดิม)
    normalized = {}
    for uname, val in raw.items():
        if isinstance(val, str):
            normalized[uname] = {"hash": val, "must_change_password": False}
        else:
            normalized[uname] = val
    return normalized


def save_admins(admins: dict) -> None:
    ADMINS_FILE.write_text(json.dumps(admins, ensure_ascii=False, indent=2), encoding="utf-8")


def load_overrides() -> dict:
    if not OVERRIDES_FILE.exists():
        return {}
    return json.loads(OVERRIDES_FILE.read_text(encoding="utf-8"))


def save_overrides(overrides: dict) -> None:
    OVERRIDES_FILE.write_text(json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8")


def require_admin(token: str) -> str:
    session = SESSIONS.get(token)
    if not session or session["expires_at"] < time.time():
        SESSIONS.pop(token, None)
        raise HTTPException(status_code=401, detail="เซสชันหมดอายุ กรุณาเข้าสู่ระบบใหม่")
    return session["username"]


load_admins()  # bootstrap admins.json ตั้งแต่ตอนเปิดเซิร์ฟเวอร์ ไม่ต้องรอ login ครั้งแรก

print("⏳ 1. กำลังสร้าง Dataset หนองหาร และเทรนโมเดล ML...")

# ---------------------------------------------------------
# 1. สร้าง Dataset จำลองและเทรนโมเดล (Machine Learning)
# ---------------------------------------------------------
np.random.seed(42)
n_samples = 1500
do = np.random.uniform(2.0, 9.0, n_samples)
nh3 = np.random.uniform(0.1, 5.0, n_samples)
fcb = np.random.uniform(100, 50000, n_samples)
rain = np.random.exponential(20, n_samples)

# สร้างกฎความวิกฤต (Target: 0=Green, 1=Yellow, 2=Red)
status_target = []
for i in range(n_samples):
    if do[i] < 3.0 or fcb[i] > 30000 or nh3[i] > 2.5:
        label = 2 # Red (อันตราย)
    elif do[i] < 5.0 or fcb[i] > 5000 or nh3[i] > 1.0 or rain[i] > 50:
        label = 1 # Yellow (เฝ้าระวัง)
    else:
        label = 0 # Green (ปลอดภัย)
    status_target.append(label)

df = pd.DataFrame({'DO': do, 'NH3': nh3, 'FCB': fcb, 'Rain': rain, 'Status': status_target})

# หมายเหตุ: label ด้านบนสร้างจากกฎ if/else ตายตัว ถ้าเทรนโมเดลด้วย label ชุดนี้ตรงๆ
# โมเดลจะแค่ "ท่องจำ" กฎเดิม ไม่ได้เรียนรู้ pattern จริงจากข้อมูล (circular training)
# จึงเติม noise แบบสุ่มเล็กน้อย (~3% ของแถว) ให้ label คลาดเคลื่อนจากกฎ เพื่อให้การวัด
# accuracy สะท้อนความสามารถ generalize ของโมเดลจริง ไม่ใช่แค่ reproduce กฎ 100%
noise_idx = np.random.choice(df.index, size=int(0.03 * n_samples), replace=False)
df.loc[noise_idx, 'Status'] = np.random.randint(0, 3, size=len(noise_idx))

# เทรนโมเดล Random Forest พร้อมแบ่ง train/test เพื่อวัดความแม่นยำจริง
X = df[['DO', 'NH3', 'FCB', 'Rain']]
y = df['Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"✅ เทรนโมเดลเสร็จสมบูรณ์ AI พร้อมทำงาน! (Test Accuracy: {accuracy:.2%})")

if not GEMINI_API_KEY:
    print("⚠️  ยังไม่ได้ตั้งค่า GEMINI_API_KEY (ในไฟล์ config.env หรือ environment variable) — /api/chat จะใช้งานไม่ได้จนกว่าจะตั้งค่า")

# ---------------------------------------------------------
# 2. สร้าง API Server สำหรับส่งข้อมูลให้ UI
# ---------------------------------------------------------
app = FastAPI()

# ปลดล็อก CORS ให้หน้าเว็บดึงข้อมูลไปใช้ได้ — allow_origins=["*"] ครอบคลุมทั้ง localhost
# และโดเมนจริงตอน deploy ขึ้น Netlify (เช่น https://sage-manatee-7c7adc.netlify.app) ไม่ต้องแก้เพิ่ม
# หมายเหตุ: เหมาะกับ demo แฮกกาธอน ถ้า production จริงควรจำกัด origin ให้เจาะจงโดเมนของตัวเอง
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class PredictRequest(BaseModel):
    do_mgL: float
    nh3_mgL: float
    ecoli_mpn: float
    rain_mm: float


@app.post("/api/predict")
def predict_status(payload: PredictRequest):
    """ใช้โมเดล RandomForest จริง (ตัวเดียวกับ dashboard) ทำนายสถานะ
    เพื่อให้หน้า 'ห้องทดลอง AI' เรียกโมเดลจริงแทนสูตรคำนวณฝั่ง client"""
    try:
        input_features = pd.DataFrame({
            'DO': [payload.do_mgL],
            'NH3': [payload.nh3_mgL],
            'FCB': [payload.ecoli_mpn],
            'Rain': [payload.rain_mm],
        })
        prediction = int(model.predict(input_features)[0])
        probabilities = model.predict_proba(input_features)[0].tolist()
        status_map = {0: 'green', 1: 'yellow', 2: 'red'}
        return {
            "status": status_map[prediction],
            "confidence": round(max(probabilities), 3),
        }
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="ไม่สามารถประมวลผลการทำนายได้ กรุณาลองใหม่อีกครั้ง")


class ChatRequest(BaseModel):
    message: str
    context: str = ""


@app.post("/api/chat")
async def chat_with_ai(payload: ChatRequest):
    """Proxy ไปหา Gemini API โดยเก็บ API key ไว้ฝั่งเซิร์ฟเวอร์เท่านั้น
    ไม่ให้ browser เห็น key เด็ดขาด"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing GEMINI_API_KEY")

    sys_prompt = (
        "คุณคือ Nong Han Smart Bot ผู้ช่วย AI ด้านมลพิษทางน้ำของโครงการแฮกกาธอนจังหวัดสกลนคร "
        "ให้ตอบคำถามผู้ใช้แบบสั้นๆ กระชับ เป็นธรรมชาติ เป็นกันเอง (ไม่เกิน 3-4 ประโยค) และให้ข้อมูลที่แม่นยำ "
        "หากผู้ใช้ถามว่าคุณเป็น AI หรือไม่ ให้ตอบตามตรงว่าใช่ "
        f"นี่คือข้อมูลแบบ Real-time ของสถานีวัดน้ำต่างๆ ในขณะนี้: [ {payload.context} ]"
    )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [
                        {"role": "user", "parts": [{"text": sys_prompt + "\n\nคำถามจากผู้ใช้: " + payload.message}]}
                    ]
                },
            )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return {"reply": text}
    except Exception:
        logger.exception("Gemini request failed")
        raise HTTPException(status_code=502, detail="ไม่สามารถเชื่อมต่อกับผู้ช่วย AI ได้ในขณะนี้ กรุณาลองใหม่")


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/admin/login")
def admin_login(payload: LoginRequest):
    admins = load_admins()
    record = admins.get(payload.username)
    if not record or not verify_password(payload.password, record["hash"]):
        raise HTTPException(status_code=401, detail="ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = {"username": payload.username, "expires_at": time.time() + SESSION_TTL_SECONDS}
    return {
        "token": token,
        "username": payload.username,
        "must_change_password": bool(record.get("must_change_password", False)),
    }


class LogoutRequest(BaseModel):
    token: str


@app.post("/api/admin/logout")
def admin_logout(payload: LogoutRequest):
    SESSIONS.pop(payload.token, None)
    return {"ok": True}


@app.get("/api/admin/list-admins")
def list_admins(token: str):
    require_admin(token)
    return {"admins": list(load_admins().keys())}


class CreateAdminRequest(BaseModel):
    token: str
    new_username: str
    new_password: str


@app.post("/api/admin/create-admin")
def create_admin(payload: CreateAdminRequest):
    require_admin(payload.token)  # ต้องล็อกอินอยู่ก่อนถึงจะเพิ่มแอดมินคนใหม่ได้
    new_username = payload.new_username.strip()
    if not new_username:
        raise HTTPException(status_code=400, detail="กรุณาระบุชื่อผู้ใช้")
    if len(payload.new_password) < 8:
        raise HTTPException(status_code=400, detail="รหัสผ่านต้องมีอย่างน้อย 8 ตัวอักษร")
    admins = load_admins()
    if new_username in admins:
        raise HTTPException(status_code=400, detail="มีชื่อผู้ใช้นี้อยู่แล้ว")
    admins[new_username] = {"hash": hash_password(payload.new_password), "must_change_password": False}
    save_admins(admins)
    logger.info("เพิ่มแอดมินใหม่: %s", new_username)
    return {"ok": True}


class ChangePasswordRequest(BaseModel):
    token: str
    current_password: str
    new_password: str


@app.post("/api/admin/change-password")
def change_password(payload: ChangePasswordRequest):
    """ให้แอดมินเปลี่ยนรหัสผ่านของตัวเอง — ใช้บังคับเปลี่ยนรหัสผ่านเริ่มต้นตอนล็อกอินครั้งแรก
    หรือเปลี่ยนรหัสผ่านตามปกติเมื่อไหร่ก็ได้"""
    username = require_admin(payload.token)
    admins = load_admins()
    record = admins.get(username)
    if not record or not verify_password(payload.current_password, record["hash"]):
        raise HTTPException(status_code=401, detail="รหัสผ่านปัจจุบันไม่ถูกต้อง")
    if len(payload.new_password) < 8:
        raise HTTPException(status_code=400, detail="รหัสผ่านใหม่ต้องมีอย่างน้อย 8 ตัวอักษร")
    admins[username] = {"hash": hash_password(payload.new_password), "must_change_password": False}
    save_admins(admins)
    logger.info("แอดมิน %s เปลี่ยนรหัสผ่านเรียบร้อยแล้ว", username)
    return {"ok": True}


class OverrideRequest(BaseModel):
    token: str
    station_id: str
    do_mgL: float | None = None
    nh3_mgL: float | None = None
    ecoli_mpn: float | None = None
    rain_mm: float | None = None
    forced_status: str | None = None  # 'green' / 'yellow' / 'red' — ใช้เมื่อโมเดล AI เองก็ใช้งานไม่ได้
    note: str = ""


@app.post("/api/admin/override")
def set_override(payload: OverrideRequest):
    username = require_admin(payload.token)
    if payload.forced_status and payload.forced_status not in ("green", "yellow", "red"):
        raise HTTPException(status_code=400, detail="forced_status ต้องเป็น green, yellow หรือ red เท่านั้น")

    overrides = load_overrides()
    entry = overrides.get(payload.station_id, {})
    for field in ("do_mgL", "nh3_mgL", "ecoli_mpn", "rain_mm", "forced_status"):
        value = getattr(payload, field)
        if value is not None:
            entry[field] = value
    entry["note"] = payload.note
    entry["updated_by"] = username
    entry["updated_at"] = time.time()
    overrides[payload.station_id] = entry
    save_overrides(overrides)
    return {"ok": True, "override": entry}


@app.delete("/api/admin/override/{station_id}")
def clear_override(station_id: str, token: str):
    require_admin(token)
    overrides = load_overrides()
    overrides.pop(station_id, None)
    save_overrides(overrides)
    return {"ok": True}


@app.get("/api/admin/overrides")
def get_overrides(token: str):
    require_admin(token)
    return load_overrides()


@app.get("/api/dashboard-data")
def get_dashboard_data():
    # ข้อมูลดิบของ 6 สถานีปัจจุบันที่เราจะให้ AI วิเคราะห์
    stations = [
        {"id": "BNH01", "name_th": "สถานี BNH01 (กลางหนองหาร)", "checkpoint_th": "หนองหาร", "lat": 17.1850, "lng": 104.1800, "do_mgL": 7.2, "ecoli_mpn": 1200, "nh3_mgL": 0.2, "rain_mm": 0},
        {"id": "BNH02", "name_th": "สถานี BNH02 (ใกล้เทศบาลนคร)", "checkpoint_th": "หนองหาร", "lat": 17.1650, "lng": 104.1450, "do_mgL": 6.8, "ecoli_mpn": 2500, "nh3_mgL": 0.4, "rain_mm": 5},
        {"id": "BNH03", "name_th": "สถานี BNH03 (เกาะดอนสวรรค์)", "checkpoint_th": "หนองหาร", "lat": 17.1950, "lng": 104.1650, "do_mgL": 7.0, "ecoli_mpn": 800, "nh3_mgL": 0.1, "rain_mm": 0},
        {"id": "TRIB01", "name_th": "ลำห้วยน้ำก่ำ (TRIB01)", "checkpoint_th": "ลำห้วย", "lat": 17.1100, "lng": 104.2200, "do_mgL": 4.1, "ecoli_mpn": 15000, "nh3_mgL": 2.1, "rain_mm": 20},
        {"id": "TRIB02", "name_th": "ลำห้วยทราย (TRIB02)", "checkpoint_th": "ลำห้วย", "lat": 17.2000, "lng": 104.1100, "do_mgL": 6.5, "ecoli_mpn": 3000, "nh3_mgL": 0.5, "rain_mm": 10},
        {"id": "TRIB03", "name_th": "ลำห้วยเชียงเครือ (TRIB03)", "checkpoint_th": "ลำห้วย", "lat": 17.2500, "lng": 104.1500, "do_mgL": 2.5, "ecoli_mpn": 35000, "nh3_mgL": 0.8, "rain_mm": 60}
    ]

    overrides = load_overrides()

    try:
        # วนลูปให้โมเดล AI ทำนายสถานะของแต่ละสถานี
        for st in stations:
            ov = overrides.get(st["id"])
            if ov:
                # แอดมินกรอกค่าด้วยมือ (กรณีเซนเซอร์/ระบบ AI มีปัญหา) ใช้ค่านี้แทนค่าเดิม
                for field in ("do_mgL", "nh3_mgL", "ecoli_mpn", "rain_mm"):
                    if field in ov:
                        st[field] = ov[field]
                st["manual_override"] = True
                st["override_note"] = ov.get("note", "")
                st["override_by"] = ov.get("updated_by", "")

                if ov.get("forced_status") in ("green", "yellow", "red"):
                    # แอดมินบังคับสถานะเองตรงๆ (ข้ามโมเดล AI ไปเลย เผื่อโมเดลเองก็ใช้งานไม่ได้)
                    st["status"] = ov["forced_status"]
                    continue

            input_features = pd.DataFrame({'DO': [st['do_mgL']], 'NH3': [st['nh3_mgL']], 'FCB': [st['ecoli_mpn']], 'Rain': [st['rain_mm']]})

            prediction = model.predict(input_features)[0] # ผลลัพธ์ 0, 1, 2

            # แปลงตัวเลขจาก ML กลับเป็นสีให้หน้าเว็บ HTML เข้าใจ
            if prediction == 2: st['status'] = 'red'
            elif prediction == 1: st['status'] = 'yellow'
            else: st['status'] = 'green'
    except Exception:
        logger.exception("Model inference failed")
        raise HTTPException(status_code=500, detail="ไม่สามารถประมวลผลข้อมูลสถานีได้ในขณะนี้")

    # ส่ง JSON กลับไปให้ UI (รูปแบบเดียวกับที่หน้าเว็บต้องการเป๊ะๆ)
    return {
        "summary": { "avg_wqi": 68.40, "class_ab_pct": "50.0%", "class_c_pct": "33.3%", "class_d_pct": "16.7%" },
        "historical_stats": {
            "wqi": [85.87, 85.5, 85.46, 76.47, 76.11, 75.89, 66.84, 64.32, 62.69, 51.41, 42.93],
            "months": ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."],
            "month_wqi": [69.66, 69.15, 68.8, 69.11, 68.0, 74.12, 72.07, 75.07, 73.18, 73.66, 66.7, 67.9],
            "month_do": [5.58, 5.55, 5.54, 5.51, 5.46, 6.44, 6.23, 6.53, 6.35, 6.4, 5.31, 5.47],
            "years": ["2564", "2565", "2566", "2567", "2568"],
            "year_wqi": [68.5, 69.2, 70.1, 71.5, 70.68],
            "year_do": [5.2, 5.4, 5.8, 6.1, 5.9],
            "class_labels": ["A (ดีมาก)", "B (ดี)", "C (พอใช้)", "D (เสื่อมโทรม)"],
            "class_counts": [3, 0, 2, 1]
        },
        "locations": stations
    }

if __name__ == "__main__":
    # ตอนรันบนเครื่องตัวเอง (ไม่มี env var PORT) จะใช้พอร์ต 8000 เหมือนเดิม
    # ตอน deploy บน Render/Railway/Fly.io ระบบจะกำหนด PORT มาให้เอง และต้อง bind ที่ 0.0.0.0
    # (ไม่ใช่ 127.0.0.1) ไม่งั้นเซิร์ฟเวอร์จะรับ request จากภายนอกไม่ได้ deploy แล้วก็ยังเข้าไม่ได้
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 เปิดเซิร์ฟเวอร์สำเร็จที่พอร์ต {port}! ย่อหน้าต่างนี้ไว้ แล้วไปเปิดไฟล์ hackathon.html ได้เลย")
    uvicorn.run(app, host="0.0.0.0", port=port)