# NongHan Hotspot AI — Smart Water Warning System

ระบบแจ้งเตือนคุณภาพน้ำหนองหาร จ.สกลนคร ด้วย AI (RandomForest) พร้อมแผงควบคุมสำหรับแอดมินกรอกข้อมูลมือกรณีเซนเซอร์/AI มีปัญหา

พัฒนาโดย **กลุ่มอ้ายมาสี่คน** — คณะวิทยาศาสตร์และวิศวกรรมศาสตร์ มหาวิทยาลัยเกษตรศาสตร์ วิทยาเขตเฉลิมพระเกียรติ จังหวัดสกลนคร

## โครงสร้างไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `hackathon.html` | หน้าเว็บทั้งหมด (public dashboard, แผนที่, สถิติ, ห้องทดลอง AI, แผงแอดมิน) |
| `ml_api.py` | เซิร์ฟเวอร์ FastAPI: เทรน/รันโมเดล RandomForest, ระบบล็อกอินแอดมิน, proxy ไปหา Gemini |
| `requirements.txt` | รายชื่อ Python package ที่ต้องติดตั้ง — บริการ deploy อย่าง Render จะอ่านไฟล์นี้อัตโนมัติ |
| `config.env.example` | ตัวอย่างไฟล์ตั้งค่า — คัดลอกเป็น `config.env` แล้วใส่คีย์จริง |

## วิธีรัน

```bash
pip install -r requirements.txt
cp config.env.example config.env   # แล้วแก้ GEMINI_API_KEY ในไฟล์นี้ให้เป็นคีย์จริงของคุณ
python ml_api.py
```

จากนั้นเปิดไฟล์ `hackathon.html` ด้วยเบราว์เซอร์ (เซิร์ฟเวอร์ `ml_api.py` ต้องรันค้างไว้ที่พอร์ต 8000)

> ถ้าไม่ได้รัน `ml_api.py` หน้าเว็บจะสลับไปใช้ข้อมูลจำลอง (Demo Data) อัตโนมัติ ทำให้ยังกดดูแผนที่/กราฟ/ห้องทดลอง AI ได้ตามปกติ — เฉพาะหน้าแอดมินและแชทบอทที่ต้องมีเซิร์ฟเวอร์รันอยู่จริง

## Deploy ขึ้นสาธารณะ (เช่น Netlify)

โปรเจกต์นี้มี 2 ส่วนที่ต้อง deploy แยกกัน เพราะ Netlify เป็น **static hosting** รันได้แค่ `hackathon.html` แต่รันเซิร์ฟเวอร์ Python (`ml_api.py`) ค้างไว้ให้ไม่ได้:

1. **Frontend (`hackathon.html`)** → deploy บน Netlify ได้ตามปกติ (ลาก-วางไฟล์ หรือเชื่อม repo นี้)
2. **Backend (`ml_api.py`)** → deploy แยกไปที่บริการที่รัน Python server ค้างไว้ได้ เช่น Render (ขั้นตอนละเอียดด้านล่าง)

### วิธี deploy `ml_api.py` บน Render (ฟรี)

1. ไปที่ [render.com](https://render.com) แล้วสมัคร/ล็อกอินด้วยบัญชี GitHub
2. กด **New +** → **Web Service**
3. เลือก repo `nonghan-hotspot-ai` นี้ แล้วกด Connect
4. ตั้งค่าตามนี้:
   - **Name**: ตั้งชื่ออะไรก็ได้ เช่น `nonghan-ml-api`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python ml_api.py`
   - **Instance Type**: Free
5. เลื่อนลงมาที่ **Environment Variables** → กด **Add Environment Variable**
   - Key: `GEMINI_API_KEY`
   - Value: ใส่คีย์จริงของคุณ (**ห้าม** commit `config.env` ตัวจริงขึ้น GitHub เด็ดขาด)
6. กด **Create Web Service** แล้วรอ build (ประมาณ 2-5 นาที) ดู log จนขึ้นว่า `🚀 เปิดเซิร์ฟเวอร์สำเร็จ`
7. ได้ URL ประมาณ `https://nonghan-ml-api.onrender.com` — คัดลอกเก็บไว้

> หมายเหตุ: แผนฟรีของ Render จะ "หลับ" หลังไม่มีคนเรียกใช้ ~15 นาที รีเควสต์แรกหลังตื่นอาจช้าไปสักครึ่งนาที เป็นเรื่องปกติของแผนฟรี ไม่ใช่บั๊ก

### เชื่อม URL backend เข้ากับหน้าเว็บ

เปิดไฟล์ `hackathon.html` (แก้ในเครื่อง หรือกด "Edit" ตรงๆ บนหน้า GitHub ก็ได้) หา (Ctrl+F) คำว่า `PROD_API_BASE` แล้วแก้บรรทัดนี้:

```js
const PROD_API_BASE = 'https://YOUR-ML-API-SERVICE.onrender.com'; // TODO: แก้เป็น URL จริงหลัง deploy ml_api.py
```

เปลี่ยน `https://YOUR-ML-API-SERVICE.onrender.com` เป็น URL จริงจากขั้นตอนที่ 7 ด้านบน เช่น:

```js
const PROD_API_BASE = 'https://nonghan-ml-api.onrender.com';
```

จากนั้น commit + push ขึ้น GitHub (หรือกด "Commit changes" ถ้าแก้บนหน้าเว็บ GitHub โดยตรง) → Netlify จะ build ให้ใหม่อัตโนมัติภายในไม่กี่นาที เปิดเว็บอีกครั้ง badge ควรขึ้นเขียว "เชื่อมต่อสำเร็จ · ข้อมูลจริงจาก AI"

**ถ้ายังไม่ได้ deploy backend:** เว็บจะขึ้น badge "โหมดสาธิต · ข้อมูลจำลอง" (สีเหลือง) ให้เอง — เป็นพฤติกรรมที่ตั้งใจออกแบบไว้กันหน้าเว็บพัง ไม่ใช่บั๊ก แต่หน้าแอดมินและแชทบอทจะใช้งานไม่ได้จนกว่าจะเชื่อม backend จริง

## เข้าสู่ระบบแอดมิน

กดปุ่ม "ผู้ดูแลระบบ" ที่ท้ายหน้าเว็บ

- ชื่อผู้ใช้เริ่มต้น: `อ้ายมาสี่คน`
- รหัสผ่านเริ่มต้น: `012301230123`

ระบบจะบังคับให้เปลี่ยนรหัสผ่านทันทีที่ล็อกอินครั้งแรก (เพราะรหัสผ่านเริ่มต้นนี้ปรากฏอยู่ใน source code ที่ทุกคนเข้าถึงได้)

## ความปลอดภัย

- ห้าม commit ไฟล์ `config.env` ตัวจริงขึ้น GitHub (มี `.gitignore` กันไว้ให้แล้ว)
- รหัสผ่านแอดมินถูกเก็บแบบ hash (PBKDF2) ไม่ใช่ plain text
- นี่คือระบบยืนยันตัวตนแบบง่ายที่เหมาะกับ demo แฮกกาธอน หากจะ deploy ใช้งานจริงสาธารณะ ควรทำ HTTPS, ระบบ auth มาตรฐาน, database จริง และ rate-limit เพิ่มเติม
