# Data Scientist / AI Engineer - Coding Test

โปรเจค Coding Test สำหรับตำแหน่ง Data Scientist / AI Engineer

## Overview

โปรเจคนี้ประกอบด้วย 3 โจทย์หลัก:

| Problem | Notebook | Description |
|---------|----------|-------------|
| 1 | `01-web-scraping.ipynb` | Web Scraping จาก Thailand Yellow Pages |
| 2 | `02-sentiment-analysis.ipynb` | Sentiment Analysis (Thai Text) |
| 3| `03-chatbot-langchain-sqlite.ipynb` | Chatbot with LangChain + SQLite |

## Requirements

```bash
pip install requests beautifulsoup4 pandas
pip install datasets transformers torch scikit-learn seaborn matplotlib
pip install google-generativeai python-dotenv
pip install langchain langchain-google-genai langchain-community
```

## Setup

1. Clone หรือ download โปรเจค
2. สร้างไฟล์ `.env` และใส่ API Key:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```
3. เปิด Jupyter Notebook และ run แต่ละไฟล์

---

## Problem 1: Web Scraping

**File:** `01-web-scraping.ipynb`

**Objective:** ดึงข้อมูลธุรกิจจาก [Thailand Yellow Pages](https://www.yellowpages.co.th/)

**Features:**
- ใช้ `requests` + `BeautifulSoup` สำหรับ scraping
- ดึงข้อมูล: ชื่อร้าน, ที่อยู่, รายละเอียด, เว็บไซต์, URL
- Export เป็น CSV (`clinic_listings_yellowpages.csv`)

**Output:**
```
| name | address | description | website | category | profile_url |
|------|---------|-------------|---------|----------|-------------|
| คลินิก A | กรุงเทพ | ... | ... | คลินิก | ... |
```

---

## Problem 2: Sentiment Analysis

**File:** `02-sentiment-analysis.ipynb`

**Objective:** วิเคราะห์ความรู้สึกของข้อความภาษาไทย (Positive / Neutral / Negative)

**Features:**
- ใช้ Hugging Face `transformers`
- Dataset: `wisesight_sentiment` (Thai social media)
- Model: `poom-sci/WangchanBERTa-finetuned-sentiment`

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix

**Results:**
- Accuracy: 93.33% (on clear test examples)

---

## Problem 3: Chatbot with Memory

**File:** `03-chatbot-with-memory.ipynb`

**Objective:** สร้าง Chatbot ที่จดจำบทสนทนา 3 รอบล่าสุด

**Features:**
- ใช้ Google Gemini API (`gemini-2.0-flash`)
- In-memory sliding window (3 turns = 6 messages)
- Context-aware responses

**Architecture:**
```
User Input → Memory Manager → Gemini API → Response
                  ↑                           |
                  └───────────────────────────┘
                     (Last 3 conversations)
```

**Demo:**
```
👤 USER: ผมชื่อสมชาย
🤖 BOT: สวัสดีครับคุณสมชาย...

👤 USER: ผมชื่ออะไรนะ?
🤖 BOT: คุณชื่อสมชายครับ  ← จำได้!
```

---

## Problem 3 (Bonus): LangChain + SQLite

**File:** `04-chatbot-langchain-sqlite.ipynb`

**Objective:** Enhanced chatbot with persistent memory

**Features:**
- ใช้ **LangChain** framework
- **SQLite** persistent storage (`chat_memory.db`)
- Multi-session support
- Production-ready architecture

**Advantages over In-Memory:**
- Persistent: ปิด notebook แล้วเปิดใหม่ยังจำได้
- Scalable: รองรับ history ขนาดใหญ่
- Multi-user: แยก session ได้

**Key Components:**
```python
# Custom sliding window with SQLite
class SlidingWindowSQLChatHistory(SQLChatMessageHistory):
    def __init__(self, session_id, db_path, max_turns=3):
        ...

# LangChain integration
chatbot = RunnableWithMessageHistory(chain, get_session_history, ...)
```

---

## Project Structure

```
test-ai-developer/
├── .env                              # API Keys (not committed)
├── README.md                         # This file
├── 01-web-scraping.ipynb            # Problem 1
├── 02-sentiment-analysis.ipynb      # Problem 2
├── 03-chatbot-with-memory.ipynb     # Problem 3
├── 04-chatbot-langchain-sqlite.ipynb # Problem 3 (Bonus)
├── clinic_listings_yellowpages.csv  # Output from Problem 1
├── confusion_matrix_sentiment.png   # Output from Problem 2
└── chat_memory.db                   # SQLite DB (created at runtime)
```

---

## Technologies Used

| Category | Technologies |
|----------|--------------|
| Web Scraping | `requests`, `BeautifulSoup`, `pandas` |
| NLP/ML | `transformers`, `datasets`, `scikit-learn` |
| LLM | Google Gemini API, LangChain |
| Storage | SQLite, CSV |
| Visualization | `matplotlib`, `seaborn` |


