# 💊 MediSnap（智藥快搜）

> AI 驅動的藥品辨識與用藥衛教系統

---

## 📌 專案簡介

MediSnap 是一套結合電腦視覺與自然語言處理的藥品衛教系統。使用者拍攝藥品照片後，系統會自動辨識藥品並以繁體中文提供適應症、用法用量、副作用、禁忌症等衛教資訊。

---

## 🏗️ 系統架構

```
使用者上傳藥品照片
        ↓
   YOLO 藥品辨識
        ↓
  RAG Pipeline 查詢知識庫
  (MedBERT 嵌入 + ChromaDB)
        ↓
   LLM 生成衛教回覆
   (Ollama / Groq API)
        ↓
  FastAPI 回傳結果給前端
```

---

## ✨ 主要功能

- 📸 **藥品影像辨識** — 使用 YOLOv8 辨識藥品，支援 15 種常見藥物
- 💬 **智慧問答** — 使用者可針對藥品追問，支援多輪對話
- 🔍 **藥品搜尋** — 支援中文藥名、品牌名、學名搜尋
- 📚 **知識庫查詢** — 基於台灣藥品仿單建立的 RAG 知識庫

---

## 🛠️ 技術棧

| 類別 | 技術 |
|------|------|
| 藥品辨識 | YOLOv8 (Ultralytics) |
| 文字嵌入 | MedBERT (medbert-pharma-v5-contrastive) |
| 向量資料庫 | ChromaDB |
| LLM | Ollama (qwen2.5:1.5b) / Groq API (llama-3.1-8b) |
| 後端 API | FastAPI + Uvicorn |
| 部署 | Google Colab + ngrok |
| 語言 | Python |

---

## 🚀 快速開始

### 前置需求

1. Google Colab（建議使用 T4 GPU）
2. Google Drive 掛載，並確認以下路徑存在：
   - `MediSnap_files/models/ragllm/medbert-pharma-v5-contrastive`
   - `MediSnap_files/models/ragllm/chromadb_store_v5`
   - `MediSnap_files/models/detection_yolo26/weights/best.pt`
3. [ngrok](https://ngrok.com) 帳號與 authtoken
4. [Groq API Key](https://console.groq.com)（若使用 Groq 版本）

### 執行步驟

1. 開啟 `medisnap_backend.py`，依照 Cell 0 → Cell 9 順序執行
2. Cell 0 執行完後**重啟 session**，再從 Cell 1 繼續
3. 在 Cell 9 填入你的 ngrok token：
   ```python
   NGROK_TOKEN = "YOUR_NGROK_TOKEN"
   ```
4. 執行完成後會顯示公開 URL，用瀏覽器開啟即可使用

---

## 📁 檔案結構

```
medisnap/
├── medisnap_backend.py   # 主要後端 Colab Notebook（含 inference + API）
└── README.md
```

---

## 🔌 API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/health` | 確認服務狀態 |
| POST | `/analyze` | 上傳藥品圖片，取得辨識結果與衛教資訊 |
| POST | `/chat` | 針對已辨識藥品追問 |
| POST | `/search` | 以關鍵字搜尋藥品 |

---

## ⚠️ 注意事項

- 本系統資訊**僅供參考**，不構成醫療診斷建議
- 實際用藥請諮詢醫師或藥師
- ngrok 免費版 URL 每次重啟後會變更

---

## 👤 作者

**陳思翰 (Hank Chen)**
- GitHub: [@hankchen0629](https://github.com/hankchen0629)
- Email: hank880629@gmail.com

---

## 📄 授權

本專案為學術用途，僅供學習與展示使用。
