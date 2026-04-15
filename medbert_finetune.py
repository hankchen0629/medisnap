# -*- coding: utf-8 -*-
"""
MediSnap - MedBERT Fine-tuning Script (v4)
==========================================
基於 trueto/medbert-base-wwm-chinese
使用台灣藥品仿單資料進行 MLM fine-tuning

作者：MediSnap Team
環境：Google Colab（建議使用 T4 GPU）

使用方式：
    1. 掛載 Google Drive
    2. 確認 RAGchunks 資料夾存在
    3. 依序執行各個區塊

輸出：
    medbert-pharma-v4-final（存於 Google Drive）
"""

# ============================================================
# Step 1：掛載 Google Drive
# ============================================================

from google.colab import drive
drive.mount('/content/drive')


# ============================================================
# Step 2：安裝套件
# ============================================================

# pip install transformers torch datasets


# ============================================================
# Step 3：載入藥品 chunks
# ============================================================

import json
import os

# 藥品仿單 JSON chunks 資料夾路徑
CHUNKS_FOLDER = '/content/drive/MyDrive/RAGchunks'

all_chunks = []

for filename in os.listdir(CHUNKS_FOLDER):
    if filename.endswith('.json'):
        filepath = os.path.join(CHUNKS_FOLDER, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
            print(f"✅ 載入：{filename}，共 {len(chunks)} 個 chunks")
        except json.JSONDecodeError as e:
            print(f"❌ 載入錯誤：{filename} - 無法解析 JSON。錯誤：{e}")
        except Exception as e:
            print(f"❌ 載入錯誤：{filename} - 發生未知錯誤。錯誤：{e}")

print(f"\n總共載入：{len(all_chunks)} 個 chunks")


# ============================================================
# Step 4：切割成句子，存成 train.txt
# ============================================================

import re

def split_into_sentences(text):
    """
    把段落切成句子
    - 以句號、驚嘆號、問號作為分割點
    - 過濾掉太短的句子（< 10 字元）
    """
    sentences = re.split(r'(?<=[。！？\.!?])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 10]
    return sentences


TRAIN_TXT_PATH = '/content/drive/MyDrive/MediSnap/train.txt'
all_sentences = []

with open(TRAIN_TXT_PATH, 'w', encoding='utf-8') as f:
    for chunk in all_chunks:
        sentences = split_into_sentences(chunk["content"])
        for sent in sentences:
            f.write(sent.strip() + '\n\n')
            all_sentences.append(sent)

print(f"✅ 原本：{len(all_chunks)} 個 chunks")
print(f"✅ 切割後：{len(all_sentences)} 個句子")
print(f"✅ 已存到：{TRAIN_TXT_PATH}")


# ============================================================
# Step 5：載入基礎模型 trueto/medbert-base-wwm-chinese
# ============================================================

import torch
from transformers import BertTokenizer, BertForMaskedLM, BertModel

# 從 Hugging Face 載入預訓練的醫療中文 BERT
# wwm = Whole Word Masking，遮蔽整個詞而非單一字元，對中文效果更好
BASE_MODEL = "trueto/medbert-base-wwm-chinese"

tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
model = BertForMaskedLM.from_pretrained(BASE_MODEL)

print("✅ 基礎模型載入完成！")
print(f"模型參數數量：{model.num_parameters():,}")
print(f"GPU 可用：{torch.cuda.is_available()}")


# ============================================================
# Step 6：建立自定義 Dataset
# ============================================================

from torch.utils.data import Dataset

class PharmDataset(Dataset):
    """
    藥品仿單句子 Dataset

    讀取 train.txt，把每一行 tokenize 成模型輸入格式
    """
    def __init__(self, tokenizer, file_path, max_length=256):
        self.examples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"總共 {len(lines)} 行文字")

        for line in lines:
            encoding = tokenizer(
                line,
                max_length=max_length,
                truncation=True,       # 超過 max_length 截斷
                padding='max_length',  # 不足 max_length 補 [PAD]
                return_tensors='pt'
            )
            self.examples.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ============================================================
# Step 7：定義 embedding 函式（用於訓練後測試）
# ============================================================

def get_embedding(text, embed_model, tokenizer, device='cuda'):
    """
    用 Mean Pooling 把句子轉成向量

    Mean Pooling：把所有真實 token 的向量平均
    用 attention_mask 過濾掉 [PAD] token
    避免 PAD token 影響平均結果
    """
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=256,
        truncation=True,
        padding='max_length'
    ).to(device)

    with torch.no_grad():
        outputs = embed_model(**inputs)

    # Mean Pooling
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state       # [1, 256, 768]
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    embedding = (sum_embeddings / sum_mask).squeeze()

    return embedding.cpu().numpy()


def cosine_similarity(a, b):
    """計算兩個向量的餘弦相似度"""
    import numpy as np
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ============================================================
# Step 8：MLM Fine-tuning
# ============================================================

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# 建立 Dataset
dataset_v4 = PharmDataset(
    tokenizer=tokenizer,
    file_path=TRAIN_TXT_PATH,
    max_length=256
)

# 訓練參數
training_args_v4 = TrainingArguments(
    output_dir='/content/drive/MyDrive/MediSnap/medbert-pharma-v4',
    num_train_epochs=80,              # 訓練 80 個 epoch
    per_device_train_batch_size=16,   # 每批 16 筆
    save_steps=200,                   # 每 200 步存一次 checkpoint
    save_total_limit=2,               # 最多保留 2 個 checkpoint
    logging_steps=500,
    report_to="none"                  # 不回報到 wandb 等平台
)

# MLM Data Collator
# mlm_probability=0.20：隨機遮蔽 20% 的 token
# 模型需要預測這些被遮蔽的字，學習藥品領域語意
data_collator_v4 = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.20
)

trainer_v4 = Trainer(
    model=model,
    args=training_args_v4,
    data_collator=data_collator_v4,
    train_dataset=dataset_v4,
)

total_steps = len(dataset_v4) // training_args_v4.per_device_train_batch_size * training_args_v4.num_train_epochs
print(f"總訓練步數：{total_steps}")
print("🚀 開始訓練 v4...")

trainer_v4.train()

print("✅ 訓練完成！")


# ============================================================
# Step 9：儲存模型
# ============================================================

SAVE_PATH_V4 = '/content/drive/MyDrive/MediSnap/medbert-pharma-v4-final'

model.save_pretrained(SAVE_PATH_V4)
tokenizer.save_pretrained(SAVE_PATH_V4)

print(f"✅ 模型已儲存：{SAVE_PATH_V4}")


# ============================================================
# Step 10：測試 embedding 品質
# ============================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 載入儲存的模型（用 BertModel，不用 BertForMaskedLM）
# BertForMaskedLM 用於訓練，BertModel 用於產生 embedding
embed_model = BertModel.from_pretrained(SAVE_PATH_V4).to(device)
embed_model.eval()

# 測試四種句子
test1 = get_embedding("頭暈、嗜睡、共濟失調、複視、噁心、嘔吐", embed_model, tokenizer, device)
test2 = get_embedding("服藥後出現頭暈嘔吐皮膚起疹", embed_model, tokenizer, device)
test3 = get_embedding("用於癲癇症大發作小發作混合型治療", embed_model, tokenizer, device)
test4 = get_embedding("今天天氣很好適合出門散步", embed_model, tokenizer, device)

print("\n測試結果 v4：")
print(f"副作用 vs 相關副作用（應該最高）：{cosine_similarity(test1, test2):.4f}")
print(f"副作用 vs 適應症（應該中等）：    {cosine_similarity(test1, test3):.4f}")
print(f"副作用 vs 天氣（應該最低）：      {cosine_similarity(test1, test4):.4f}")

# 理想結果：
# 副作用 vs 相關副作用 > 副作用 vs 適應症 > 副作用 vs 天氣
