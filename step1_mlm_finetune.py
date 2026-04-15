# -*- coding: utf-8 -*-
"""
MediSnap - Step 1: MedBERT MLM Fine-tuning (v4c)
=================================================
基於 trueto/medbert-base-wwm-chinese
使用台灣藥品仿單資料進行 MLM fine-tuning

訓練完成後產生：medbert-pharma-v4c-final
這個模型會作為 Step 2 Contrastive Learning 的起點

環境：本機（建議使用 GPU）或 Google Colab T4
作者：MediSnap Team
"""

import json
import os
import re
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import (
    BertTokenizer, BertForMaskedLM, BertModel,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    TrainerCallback
)
from torch.utils.data import Dataset

# ==================== 設定區（依環境修改這裡）====================
CHUNKS_FOLDER = os.path.join('data', 'RAGchunks')       # 藥品 JSON 資料夾
TRAIN_TXT     = os.path.join('data', 'train.txt')        # 切割後的句子
SAVE_DIR      = os.path.join('models', 'medbert-v4c')    # 訓練中途 checkpoint
FINAL_DIR     = os.path.join('models', 'medbert-pharma-v4c-final')  # 最終模型
LOG_DIR       = os.path.join('runs', 'medbert-v4c')      # TensorBoard logs
CURVE_PNG     = os.path.join('outputs', 'training_curve_v4c.png')   # loss 曲線圖

BASE_MODEL    = "trueto/medbert-base-wwm-chinese"        # Hugging Face 基礎模型

# 訓練超參數
NUM_EPOCHS    = 10
BATCH_SIZE    = 24
MAX_LENGTH    = 256
MLM_PROB      = 0.20   # 遮蔽比例
LOGGING_STEPS = 50
SAVE_STEPS    = 200
# =================================================================

# 建立需要的資料夾
for folder in ['data', 'models', 'runs', 'outputs']:
    os.makedirs(folder, exist_ok=True)


# ─────────────────────────────────────────
# Step 1：載入所有 RAG chunks
# ─────────────────────────────────────────
print("=" * 60)
print("Step 1：載入藥品 chunks")
print("=" * 60)

all_chunks = []

for filename in os.listdir(CHUNKS_FOLDER):
    if not filename.endswith('.json'):
        continue
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


# ─────────────────────────────────────────
# Step 2：切割句子，存成 train.txt
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2：切割句子")
print("=" * 60)

def split_into_sentences(text):
    """
    把段落切成句子
    以句號、驚嘆號、問號作為分割點
    過濾掉太短的句子（< 10 字元）
    """
    sentences = re.split(r'(?<=[。！？\.!?])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 10]
    return sentences


all_sentences = []

with open(TRAIN_TXT, 'w', encoding='utf-8') as f:
    for chunk in all_chunks:
        sentences = split_into_sentences(chunk["content"])
        for sent in sentences:
            f.write(sent.strip() + '\n\n')
            all_sentences.append(sent)

print(f"✅ 原本：{len(all_chunks)} 個 chunks")
print(f"✅ 切割後：{len(all_sentences)} 個句子")
print(f"✅ 已存到：{TRAIN_TXT}")


# ─────────────────────────────────────────
# Step 3：載入基礎 MedBERT 模型
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3：載入基礎模型")
print("=" * 60)

tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
model = BertForMaskedLM.from_pretrained(BASE_MODEL)

print(f"✅ 基礎模型載入完成：{BASE_MODEL}")
print(f"模型參數數量：{model.num_parameters():,}")
print(f"GPU 可用：{torch.cuda.is_available()}")


# ─────────────────────────────────────────
# Step 4：建立 Dataset
# ─────────────────────────────────────────
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
                truncation=True,
                padding='max_length',
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


# ─────────────────────────────────────────
# Step 5：定義 Embedding 工具（訓練後測試用）
# ─────────────────────────────────────────
def get_embedding(text, embed_model, tokenizer, device='cuda'):
    """
    用 CLS token 把句子轉成向量
    加上 L2 normalize 讓 cosine similarity 計算更穩定
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

    # 取 CLS token（第 0 個位置）
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    # L2 normalize
    cls_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)

    return cls_embedding.squeeze().cpu().numpy()


def cosine_similarity(a, b):
    """計算兩個向量的餘弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ─────────────────────────────────────────
# Step 6：自訂 Callback（記錄 loss 用於畫圖）
# ─────────────────────────────────────────
class LossRecorderCallback(TrainerCallback):
    """訓練過程中收集 loss，供事後畫 PNG 用"""
    def __init__(self):
        self.loss_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.loss_history.append((state.global_step, logs["loss"]))


# ─────────────────────────────────────────
# Step 7：MLM Fine-tuning
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 7：開始 MLM Fine-tuning")
print("=" * 60)

dataset_v4 = PharmDataset(
    tokenizer=tokenizer,
    file_path=TRAIN_TXT,
    max_length=MAX_LENGTH
)

training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    report_to="tensorboard",
    logging_dir=LOG_DIR,
    logging_strategy="steps",
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
)

# MLM Data Collator
# mlm_probability：隨機遮蔽這個比例的 token
# 模型需要預測被遮蔽的字，學習藥品領域語意
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MLM_PROB
)

loss_recorder = LossRecorderCallback()

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_v4,
    callbacks=[loss_recorder],
)

total_steps = len(dataset_v4) // BATCH_SIZE * NUM_EPOCHS
print(f"總訓練步數：{total_steps}")
print(f"📊 TensorBoard logs → {LOG_DIR}")
print("🚀 開始訓練...")

trainer.train()
print("✅ 訓練完成！")


# ─────────────────────────────────────────
# Step 8：儲存模型
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 8：儲存模型")
print("=" * 60)

os.makedirs(FINAL_DIR, exist_ok=True)
model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
print(f"✅ 模型已儲存：{FINAL_DIR}")


# ─────────────────────────────────────────
# Step 9：畫 Training Curve
# ─────────────────────────────────────────
if loss_recorder.loss_history:
    steps, losses = zip(*loss_recorder.loss_history)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, losses, color='steelblue', linewidth=1.5, label='Train Loss (MLM)')
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('MedBERT v4c — Training Curve', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

    # 標記最低 loss
    min_idx = int(np.argmin(losses))
    ax.annotate(
        f'min={losses[min_idx]:.4f}',
        xy=(steps[min_idx], losses[min_idx]),
        xytext=(steps[min_idx], losses[min_idx] + 0.05),
        arrowprops=dict(arrowstyle='->', color='red'),
        color='red', fontsize=10
    )

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(CURVE_PNG, dpi=150)
    plt.close()
    print(f"📈 Training curve 已儲存：{CURVE_PNG}")

print(f"\n📊 開啟 TensorBoard：")
print(f"   tensorboard --logdir {LOG_DIR}")
print(f"   然後瀏覽器開啟 http://localhost:6006")


# ─────────────────────────────────────────
# Step 10：測試 Embedding 品質
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 10：測試 Embedding 品質")
print("=" * 60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

embed_model = BertModel.from_pretrained(FINAL_DIR).to(device)
embed_model.eval()

# 測試組 1：基準測試
print("\n📌 測試組 1：基準測試（語意差距明顯）")
t1a = get_embedding("頭暈、嗜睡、共濟失調、複視、噁心、嘔吐", embed_model, tokenizer, device)
t1b = get_embedding("服藥後出現頭暈嘔吐皮膚起疹", embed_model, tokenizer, device)
t1c = get_embedding("用於癲癇症大發作小發作混合型治療", embed_model, tokenizer, device)
t1d = get_embedding("今天天氣很好適合出門散步", embed_model, tokenizer, device)
print(f"  副作用 vs 相關副作用（應最高 >0.90）：{cosine_similarity(t1a, t1b):.4f}")
print(f"  副作用 vs 適應症    （應中等 ~0.70）：{cosine_similarity(t1a, t1c):.4f}")
print(f"  副作用 vs 天氣      （應最低 <0.50）：{cosine_similarity(t1a, t1d):.4f}")

# 測試組 2：禁忌症 vs 適應症
print("\n📌 測試組 2：禁忌症 vs 適應症")
t2a = get_embedding("本藥禁用於對 Aspirin 或其他非類固醇消炎藥過敏者", embed_model, tokenizer, device)
t2b = get_embedding("本藥禁用於嚴重肝功能不全或活動性肝臟疾病患者", embed_model, tokenizer, device)
t2c = get_embedding("本藥適用於輕度至中度疼痛及發炎症狀之治療", embed_model, tokenizer, device)
t2d = get_embedding("本藥適用於預防心肌梗塞及缺血性腦中風之復發", embed_model, tokenizer, device)
print(f"  禁忌症 vs 禁忌症（應最高）：{cosine_similarity(t2a, t2b):.4f}")
print(f"  適應症 vs 適應症（應最高）：{cosine_similarity(t2c, t2d):.4f}")
print(f"  禁忌症 vs 適應症（應明顯低）：{cosine_similarity(t2a, t2c):.4f}")

# 測試組 3：嚴重程度區分
print("\n📌 測試組 3：相似副作用，不同嚴重程度")
t3a = get_embedding("可能出現輕微頭痛及短暫性頭暈，通常無需停藥", embed_model, tokenizer, device)
t3b = get_embedding("出現嚴重頭痛、視力模糊、意識障礙時應立即停藥就醫", embed_model, tokenizer, device)
t3c = get_embedding("偶有噁心、腸胃不適，飯後服用可改善症狀", embed_model, tokenizer, device)
t3d = get_embedding("嚴重肝毒性、黃疸、腹痛，應立即停藥並緊急就醫", embed_model, tokenizer, device)
print(f"  輕微副作用 vs 嚴重副作用（應可區分）：{cosine_similarity(t3a, t3b):.4f}")
print(f"  腸胃副作用 vs 肝毒性（應可區分）：{cosine_similarity(t3c, t3d):.4f}")

# 測試組 4：交互作用 vs 副作用
print("\n📌 測試組 4：交互作用 vs 副作用")
t4a = get_embedding("與 Warfarin 併用可能增加出血風險，需監測 INR 值", embed_model, tokenizer, device)
t4b = get_embedding("與 Digoxin 併用可能使其血中濃度上升，需監測心律", embed_model, tokenizer, device)
t4c = get_embedding("服藥後可能出現不正常出血、瘀青或傷口不易癒合", embed_model, tokenizer, device)
t4d = get_embedding("可能引起心跳過速、心悸或心律不整等心臟副作用", embed_model, tokenizer, device)
print(f"  交互作用 vs 交互作用（應最高）：{cosine_similarity(t4a, t4b):.4f}")
print(f"  副作用   vs 副作用（應最高）：{cosine_similarity(t4c, t4d):.4f}")
print(f"  出血交互 vs 出血副作用（難題）：{cosine_similarity(t4a, t4c):.4f}")

# 測試組 5：台灣仿單真實句型
print("\n📌 測試組 5：台灣仿單真實句型")
t5a = get_embedding("本藥不得與單胺氧化酶抑制劑（MAOI）併用", embed_model, tokenizer, device)
t5b = get_embedding("服用本藥期間請勿飲酒，酒精會加重中樞神經抑制作用", embed_model, tokenizer, device)
t5c = get_embedding("孕婦及哺乳中婦女應避免使用本藥", embed_model, tokenizer, device)
t5e = get_embedding("今天台北天氣晴朗，適合外出運動", embed_model, tokenizer, device)
print(f"  MAOI 警語 vs 飲酒警語（應相近）：{cosine_similarity(t5a, t5b):.4f}")
print(f"  懷孕警語 vs MAOI 警語（應中等）：{cosine_similarity(t5a, t5c):.4f}")
print(f"  仿單警語 vs 天氣（應最低）：{cosine_similarity(t5a, t5e):.4f}")

print("\n✅ 測試完成！可以繼續跑 step2_contrastive.py")
