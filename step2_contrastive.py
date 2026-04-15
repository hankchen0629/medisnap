# -*- coding: utf-8 -*-
"""
MediSnap - Step 2: MedBERT Contrastive Learning (v5)
=====================================================
在 v4c 基礎上用 Triplet Loss 做 Contrastive Learning
讓相似 section 的向量靠近，不相似的推遠

訓練完成後產生：medbert-pharma-v5-contrastive
這個模型會作為 Step 3 建庫的 embedding model

Triplet Loss 概念：
    Anchor：一個句子
    Positive：同 section 的另一個句子（應該靠近）
    Negative：不同 section 的句子（應該推遠）

環境：本機（建議使用 GPU）或 Google Colab T4
作者：MediSnap Team
"""

import json
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

# ==================== 設定區（依環境修改這裡）====================
PRETRAIN_DIR  = os.path.join('models', 'medbert-pharma-v4c-final')       # Step 1 輸出
CHUNKS_FOLDER = os.path.join('data', 'RAGchunks')                         # 藥品 JSON 資料夾
SAVE_DIR      = os.path.join('models', 'medbert-pharma-v5-contrastive')   # 最終模型
LOG_DIR       = os.path.join('runs', 'medbert-v5-contrastive')            # TensorBoard logs
CURVE_PNG     = os.path.join('outputs', 'training_curve_v5.png')          # loss 曲線圖

# Triplet Loss 超參數
MARGIN       = 0.3    # positive 和 negative 之間最小要差多少
EPOCHS       = 30     # 訓練幾個 epoch
BATCH_SIZE   = 16     # 每批幾個 triplet
LR           = 2e-5   # 學習率
MAX_LENGTH   = 256    # token 最大長度
WARMUP_RATIO = 0.1    # warmup 比例
N_TRIPLETS   = 2000   # 生成幾個 triplet
# =================================================================

# 建立需要的資料夾
for folder in ['models', 'runs', 'outputs']:
    os.makedirs(folder, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用裝置：{device}")


# ─────────────────────────────────────────
# Step 1：載入 chunks，按 section 分組
# ─────────────────────────────────────────
print("=" * 60)
print("Step 1：載入 chunks 並按 section 分組")
print("=" * 60)

# section 正規化對照表（合併同義 section）
SECTION_NORMALIZE = {
    '適應症':       'indications',
    '禁忌症':       'contraindications',
    '警語':         'warnings',
    '副作用':       'adverse_effects',
    '不良反應':     'adverse_effects',
    '藥物交互作用': 'interactions',
    '交互作用':     'interactions',
    '使用劑量':     'dosage',
    '用法用量':     'dosage',
    '使用方式':     'dosage',
}

section_to_sentences = defaultdict(list)

for filename in os.listdir(CHUNKS_FOLDER):
    if not filename.endswith('.json'):
        continue
    with open(os.path.join(CHUNKS_FOLDER, filename), 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    for chunk in chunks:
        raw_section = chunk.get('section', chunk.get('metadata', {}).get('section', ''))
        section = SECTION_NORMALIZE.get(raw_section, raw_section)
        content = chunk.get('content', '').strip()
        if content and len(content) >= 15:
            section_to_sentences[section].append(content)

print("各 section 句子數量：")
for sec, sents in section_to_sentences.items():
    print(f"  {sec}: {len(sents)} 句")

all_sections = list(section_to_sentences.keys())
print(f"\n共 {len(all_sections)} 個 section 類型")


# ─────────────────────────────────────────
# Step 2：建立 Triplet Dataset
# ─────────────────────────────────────────
class TripletDataset(Dataset):
    """
    Triplet Dataset

    每筆資料包含三個句子：
        Anchor：隨機選一個 section 的句子
        Positive：同 section 的另一個句子（應靠近）
        Negative：不同 section 的句子（應推遠）
    """
    def __init__(self, section_to_sentences, n_triplets=2000):
        self.triplets = []
        sections = [s for s, sents in section_to_sentences.items() if len(sents) >= 2]

        for _ in range(n_triplets):
            anchor_sec = random.choice(sections)
            neg_sec    = random.choice([s for s in sections if s != anchor_sec])

            anchor, positive = random.sample(section_to_sentences[anchor_sec], 2)
            negative = random.choice(section_to_sentences[neg_sec])

            self.triplets.append((anchor, positive, negative))

        print(f"✅ 生成 {len(self.triplets)} 個 triplets")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


def collate_fn(batch, tokenizer, max_length):
    """把一批 triplet 轉成模型輸入格式"""
    anchors, positives, negatives = zip(*batch)

    def encode(texts):
        return tokenizer(
            list(texts),
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

    return encode(anchors), encode(positives), encode(negatives)


# ─────────────────────────────────────────
# Step 3：模型（CLS token + L2 normalize）
# ─────────────────────────────────────────
class ContrastiveBERT(nn.Module):
    """
    在 BertModel 外面包一層
    輸出 CLS token 的 L2 normalized 向量
    直接適合做 cosine similarity 計算
    """
    def __init__(self, model_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return F.normalize(cls_emb, p=2, dim=1)


# ─────────────────────────────────────────
# Step 4：Triplet Loss
# ─────────────────────────────────────────
class TripletLoss(nn.Module):
    """
    Triplet Loss

    目標：
        positive distance（anchor 和 positive 的距離）越小越好
        negative distance（anchor 和 negative 的距離）越大越好
        兩者差距至少要大於 margin

    公式：
        loss = max(0, pos_dist - neg_dist + margin)
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # cosine distance = 1 - cosine_similarity
        pos_dist = 1 - F.cosine_similarity(anchor, positive)
        neg_dist = 1 - F.cosine_similarity(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean(), pos_dist.mean().item(), neg_dist.mean().item()


# ─────────────────────────────────────────
# Step 5：訓練
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5：開始 Contrastive Learning 訓練")
print("=" * 60)

tokenizer = BertTokenizer.from_pretrained(PRETRAIN_DIR)
model     = ContrastiveBERT(PRETRAIN_DIR).to(device)

dataset    = TripletDataset(section_to_sentences, n_triplets=N_TRIPLETS)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, tokenizer, MAX_LENGTH)
)

criterion = TripletLoss(margin=MARGIN)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps  = len(dataloader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

writer = SummaryWriter(log_dir=LOG_DIR)

loss_history     = []
pos_dist_history = []
neg_dist_history = []

print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Margin: {MARGIN} | LR: {LR}")
print(f"總步數：{total_steps} | Warmup：{warmup_steps}")
print(f"📊 TensorBoard logs → {LOG_DIR}")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    epoch_pos  = 0
    epoch_neg  = 0

    for anc_enc, pos_enc, neg_enc in dataloader:
        anc_enc = {k: v.to(device) for k, v in anc_enc.items()}
        pos_enc = {k: v.to(device) for k, v in pos_enc.items()}
        neg_enc = {k: v.to(device) for k, v in neg_enc.items()}

        anc_emb = model(**anc_enc)
        pos_emb = model(**pos_enc)
        neg_emb = model(**neg_enc)

        loss, pos_d, neg_d = criterion(anc_emb, pos_emb, neg_emb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        epoch_pos  += pos_d
        epoch_neg  += neg_d

    avg_loss = epoch_loss / len(dataloader)
    avg_pos  = epoch_pos  / len(dataloader)
    avg_neg  = epoch_neg  / len(dataloader)
    margin_gap = avg_neg - avg_pos  # 越大越好，應 > MARGIN

    loss_history.append(avg_loss)
    pos_dist_history.append(avg_pos)
    neg_dist_history.append(avg_neg)

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Loss: {avg_loss:.4f} | "
          f"pos_dist: {avg_pos:.4f} | "
          f"neg_dist: {avg_neg:.4f} | "
          f"margin gap: {margin_gap:.4f}")

    writer.add_scalar('Loss/train', avg_loss, epoch + 1)
    writer.add_scalar('Distance/positive', avg_pos, epoch + 1)
    writer.add_scalar('Distance/negative', avg_neg, epoch + 1)
    writer.add_scalar('Margin_gap', margin_gap, epoch + 1)

print("\n✅ 訓練完成！")
writer.close()


# ─────────────────────────────────────────
# Step 6：儲存模型
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 6：儲存模型")
print("=" * 60)

os.makedirs(SAVE_DIR, exist_ok=True)
model.bert.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ 模型已儲存：{SAVE_DIR}")


# ─────────────────────────────────────────
# Step 7：畫訓練曲線
# ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(loss_history, color='steelblue', linewidth=1.5, label='Triplet Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('MedBERT v5 Contrastive - Triplet Loss')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

ax2.plot(pos_dist_history, color='green', linewidth=1.5, label='Positive Distance')
ax2.plot(neg_dist_history, color='red',   linewidth=1.5, label='Negative Distance')
ax2.axhline(y=MARGIN, color='gray', linestyle='--', alpha=0.7, label=f'Margin={MARGIN}')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Cosine Distance')
ax2.set_title('Positive vs Negative Distance')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig(CURVE_PNG, dpi=150)
plt.close()
print(f"📈 Training curve 已儲存：{CURVE_PNG}")


# ─────────────────────────────────────────
# Step 8：快速驗證
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 8：快速驗證")
print("=" * 60)

model.eval()

def get_embedding(text):
    inputs = tokenizer(
        text, return_tensors='pt',
        max_length=MAX_LENGTH, truncation=True, padding='max_length'
    ).to(device)
    with torch.no_grad():
        emb = model(**inputs)
    return emb.squeeze().cpu().numpy()

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

t_adv1 = get_embedding("頭暈、嗜睡、共濟失調、複視、噁心、嘔吐")
t_adv2 = get_embedding("服藥後出現頭暈嘔吐皮膚起疹")
t_ind  = get_embedding("用於癲癇症大發作小發作混合型治療")
t_wthr = get_embedding("今天天氣很好適合出門散步")
t_ci1  = get_embedding("本藥禁用於對 Aspirin 或其他非類固醇消炎藥過敏者")
t_ci2  = get_embedding("本藥禁用於嚴重肝功能不全或活動性肝臟疾病患者")
t_ind2 = get_embedding("本藥適用於輕度至中度疼痛及發炎症狀之治療")

print(f"副作用 vs 相關副作用（應最高）：{cosine_sim(t_adv1, t_adv2):.4f}")
print(f"副作用 vs 適應症    （應中等）：{cosine_sim(t_adv1, t_ind):.4f}")
print(f"副作用 vs 天氣      （應最低）：{cosine_sim(t_adv1, t_wthr):.4f}")
print(f"禁忌症 vs 禁忌症    （應最高）：{cosine_sim(t_ci1,  t_ci2):.4f}")
print(f"禁忌症 vs 適應症    （應明顯低）：{cosine_sim(t_ci1, t_ind2):.4f}")

print("\n✅ 可以繼續跑 step3_build_chromadb.py")
