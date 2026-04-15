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
- ⚖️ **LLM vs RAG 比較** — 視覺化比較有無 RAG 的回答差異

---

## 🎬 Demo 影片

### 完整系統 Demo
[![MediSnap Demo](https://img.youtube.com/vi/AKphOy9sc90/0.jpg)](https://youtube.com/shorts/AKphOy9sc90)

### LLM Only vs RAG 比較
[![LLM vs RAG](https://img.youtube.com/vi/7rEnCLQzLPY/0.jpg)](https://youtu.be/7rEnCLQzLPY)

---

## 🛠️ 技術棧

| 類別 | 技術 |
|------|------|
| 藥品辨識 | YOLOv8 (Ultralytics) |
| 文字嵌入 | MedBERT (medbert-pharma-v5-contrastive) |
| 向量資料庫 | ChromaDB |
| LLM | Ollama (qwen2.5:1.5b / 7b) / Groq API (llama-3.1-8b) |
| 後端 API | FastAPI + Uvicorn |
| 比較介面 | Gradio |
| 部署 | Google Colab + ngrok |
| 語言 | Python |

---

## 📁 檔案結構

```
MediSnap/
├── README.md
├── requirements.txt
│
├── training/                          # 模型訓練流程（依序執行）
│   ├── step1_mlm_finetune.py          # MedBERT MLM Fine-tuning (v4c)
│   ├── step2_contrastive.py           # Contrastive Learning (v5)
│   ├── step3_build_chromadb.py        # 建立 ChromaDB 向量資料庫
│   └── step4_compare.py              # LLM Only vs RAG 比較介面
│
└── inference/                         # 推理與部署
    ├── inference.py                   # RAG 核心邏輯
    ├── main.py                        # FastAPI 路由
    └── MediSnap_Backend.ipynb         # Colab 部署 Notebook
```

---

## 🔬 模型訓練流程

MediSnap 的 RAG 系統使用自訓練的 MedBERT 模型，從原版模型開始經過兩階段訓練：

```
trueto/medbert-base-wwm-chinese（基礎醫療中文 BERT）
        ↓ step1_mlm_finetune.py
medbert-pharma-v4c（MLM Fine-tuning，學習藥品仿單語言）
        ↓ step2_contrastive.py
medbert-pharma-v5-contrastive（Contrastive Learning，優化向量空間）
        ↓ step3_build_chromadb.py
chromadb_store_v5（向量資料庫，供 RAG 查詢用）
```

### Step 1：MLM Fine-tuning
- 用台灣藥品仿單句子對 MedBERT 做 Masked Language Model 訓練
- 讓模型學會藥品領域的專業語彙
- 訓練參數：80 epochs、batch size 24、MLM probability 0.20

### Step 2：Contrastive Learning
- 用 Triplet Loss 繼續訓練
- 讓相同 section（適應症/副作用/禁忌症等）的句子向量靠近
- 讓不同 section 的句子向量推遠
- 直接優化向量空間，提升 RAG 搜尋準確度

### Step 3：建庫
- 用 v5 模型把所有藥品 chunk 轉成 CLS token 向量
- 批次寫入 ChromaDB，建立向量索引

### Step 4：LLM vs RAG 比較
- 用 Gradio 建立比較介面
- 左欄：純 LLM 回答，右欄：RAG 增強回答
- 內建測試題庫，涵蓋劑量、交互作用、警語等難題

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

### 訓練模型（如需重新訓練）

依序執行 training/ 資料夾下的四個腳本：

```bash
# Step 1：MLM Fine-tuning（約需 1-2 小時，建議使用 GPU）
python training/step1_mlm_finetune.py

# Step 2：Contrastive Learning（約需 30 分鐘）
python training/step2_contrastive.py

# Step 3：建立向量資料庫
python training/step3_build_chromadb.py

# Step 4：啟動比較介面（可選）
python training/step4_compare.py
```

### 部署後端（Colab）

1. 開啟 `inference/MediSnap_Backend.ipynb`
2. Cell 0 執行完後**重啟 session**，再從 Cell 1 繼續
3. 在 Cell 9 填入你的 ngrok token：
   ```python
   NGROK_TOKEN = "YOUR_NGROK_TOKEN"
   ```
4. 執行完成後會顯示公開 URL，用瀏覽器開啟即可使用

### 模型下載

以下檔案因體積較大，不存放於 GitHub，請從 Google Drive 下載：

| 檔案 | 說明 |
|------|------|
| medbert-pharma-v5-contrastive | 自訓練的 MedBERT embedding 模型 |
| chromadb_store_v5 | 預建的向量資料庫 |
| best.pt | YOLOv8 藥品辨識模型 |

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
- 模型和資料庫檔案不包含在此 repo 中，需另行下載

---

## 👤 作者

**陳思翰 (Hank Chen)**
- GitHub: [@hankchen0629](https://github.com/hankchen0629)
- Email: hank880629@gmail.com

---

## 📄 授權

本專案為學術用途，僅供學習與展示使用。
