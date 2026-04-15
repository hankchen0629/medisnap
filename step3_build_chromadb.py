# -*- coding: utf-8 -*-
"""
MediSnap - Step 3: 建立 ChromaDB 向量資料庫
============================================
用 v5 Contrastive 模型把所有藥品 chunks 轉成向量
存入 ChromaDB 建立向量索引

這是 RAG 系統的「建庫」階段，只需要跑一次
建好的 chromadb_store_v5 之後直接給 inference.py 使用

流程：
    載入 v5 模型
    → 把每個藥品 chunk 轉成 CLS token 向量
    → 批次寫入 ChromaDB
    → 驗證檢索結果

環境：本機（建議使用 GPU）或 Google Colab T4
作者：MediSnap Team
"""

import json
import os
import torch
import torch.nn.functional as F
import chromadb
from transformers import BertTokenizer, BertModel

# ==================== 設定區（依環境修改這裡）====================
MODEL_PATH      = os.path.join('models', 'medbert-pharma-v5-contrastive')  # Step 2 輸出
CHUNKS_FOLDER   = os.path.join('data', 'RAGchunks')                         # 藥品 JSON 資料夾
CHROMA_DB_DIR   = os.path.join('models', 'chromadb_store_v5')               # ChromaDB 輸出
COLLECTION_NAME = 'pharma_chunks'
MAX_LENGTH      = 256
BATCH_SIZE      = 32   # 每批處理幾個 chunk
# =================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用裝置：{device}")


# ─────────────────────────────────────────
# Step 1：載入 v5 模型
# ─────────────────────────────────────────
print("=" * 60)
print("Step 1：載入 v5 模型")
print("=" * 60)

tokenizer   = BertTokenizer.from_pretrained(MODEL_PATH)
embed_model = BertModel.from_pretrained(MODEL_PATH).to(device)
embed_model.eval()

print(f"✅ 模型載入完成：{MODEL_PATH}")


# ─────────────────────────────────────────
# Step 2：定義 embedding 函式
# ─────────────────────────────────────────
def get_embedding(text):
    """
    把一段文字轉成 768 維向量

    流程：
        文字 → tokenize → MedBERT v5
        → 取 CLS token（第 0 個位置）
        → L2 normalize
        → 回傳 Python list（ChromaDB 格式）
    """
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length'
    ).to(device)

    with torch.no_grad():
        outputs = embed_model(**inputs)

    cls_emb = outputs.last_hidden_state[:, 0, :]
    cls_emb = F.normalize(cls_emb, p=2, dim=1)

    return cls_emb.squeeze().cpu().numpy().tolist()


# ─────────────────────────────────────────
# Step 3：載入所有藥品 chunks
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3：載入所有藥品 chunks")
print("=" * 60)

all_chunks = []
for filename in os.listdir(CHUNKS_FOLDER):
    if not filename.endswith('.json'):
        continue
    with open(os.path.join(CHUNKS_FOLDER, filename), 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        all_chunks.extend(chunks)
    print(f"✅ 載入：{filename}，共 {len(chunks)} 個 chunks")

print(f"\n總共 {len(all_chunks)} 個 chunks 需要建立 embedding")


# ─────────────────────────────────────────
# Step 4：建立 ChromaDB
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4：建立 ChromaDB")
print("=" * 60)

os.makedirs(CHROMA_DB_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# 如果已存在就清掉重建
try:
    chroma_client.delete_collection(COLLECTION_NAME)
    print("🗑️  舊 collection 已清除")
except:
    pass

# 建立新的 collection，使用 cosine 距離
collection = chroma_client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
print(f"✅ 新 collection 建立完成：{COLLECTION_NAME}")


# ─────────────────────────────────────────
# Step 5：批次寫入
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5：批次寫入向量")
print("=" * 60)

total = len(all_chunks)

for i in range(0, total, BATCH_SIZE):
    batch = all_chunks[i:i + BATCH_SIZE]

    ids        = []
    embeddings = []
    documents  = []
    metadatas  = []

    for j, chunk in enumerate(batch):
        chunk_id = f"chunk_{i+j:04d}"
        content  = chunk.get('content', '').strip()
        metadata = chunk.get('metadata', {})

        # 相容不同 JSON 格式
        if not metadata:
            metadata = {
                'drug_name':  chunk.get('drug_name', '未知'),
                'brand_name': chunk.get('brand_name', ''),
                'section':    chunk.get('section', '未知'),
                'source':     chunk.get('source', '未知'),
            }

        if not content:
            continue

        emb = get_embedding(content)

        ids.append(chunk_id)
        embeddings.append(emb)
        documents.append(content)
        metadatas.append(metadata)

    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    progress = min(i + BATCH_SIZE, total)
    print(f"  進度：{progress}/{total} ({progress/total*100:.1f}%)")

print(f"\n✅ ChromaDB 建立完成！共 {collection.count()} 筆")
print(f"📁 儲存位置：{CHROMA_DB_DIR}")


# ─────────────────────────────────────────
# Step 6：驗證檢索結果
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 6：驗證檢索結果")
print("=" * 60)

test_queries = [
    ("癲妥錠的禁忌症",     "contraindications"),
    ("服藥後出現頭暈嘔吐", "adverse_effects"),
    ("每日建議劑量",       "dosage"),
]

for query, expected_section in test_queries:
    query_emb = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=3,
        include=['documents', 'metadatas', 'distances']
    )

    print(f"\n查詢：「{query}」（期望 section：{expected_section}）")
    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        hit = "✅" if meta.get('section') == expected_section else "❌"
        print(f"  {hit} [{meta.get('drug_name')}][{meta.get('section')}] "
              f"dist={dist:.4f} | {doc[:50]}...")

print("\n✅ ChromaDB 建立完成！可以開始使用 inference.py")
