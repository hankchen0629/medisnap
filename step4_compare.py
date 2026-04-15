# -*- coding: utf-8 -*-
"""
MediSnap - Step 4: LLM Only vs RAG 回答比較系統
================================================
用 Gradio 建立網頁介面，同時顯示：
    左欄：純 LLM 回答（無仿單資料）
    右欄：LLM + RAG 回答（有仿單資料庫）

讓你直觀比較 RAG 對回答品質的提升效果

RAG 檢索使用三條軌道：
    軌道 1：強制抓 high priority chunks
    軌道 2：關鍵字路由（問到劑量就抓劑量 section）
    軌道 3：語意搜尋（CLS embedding + cosine）

執行方式：
    python step4_compare.py
    然後瀏覽器開啟 http://localhost:7860

環境：本機（需先啟動 Ollama）
作者：MediSnap Team
"""

import os
import torch
import gradio as gr
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import chromadb

# ==================== 設定區（依環境修改這裡）====================
MODEL_PATH      = os.path.join('models', 'medbert-pharma-v5-contrastive')
CHROMA_DB_DIR   = os.path.join('models', 'chromadb_store_v5')
COLLECTION_NAME = 'pharma_chunks'
LLM_MODEL       = 'qwen2.5:7b'         # Ollama 模型名稱
OLLAMA_BASE_URL = 'http://127.0.0.1:11434/v1'  # Ollama API 位址

# RAG 檢索參數
TOP_K_SEMANTIC  = 8     # 語意搜尋最多回傳幾筆
TOP_K_PRIORITY  = 2     # 強制抓 high priority 最多幾筆
MAX_CONTEXT_LEN = 3000  # context 最大字元數
# =================================================================

# 藥品清單（Gradio 下拉選單用）
DRUG_LIST = [
    '（不指定藥品）',
    'Esidrex 25mg',
    'Eveness 10mg',
    'Fixuric 40mg',
    'Euthyrox 100mcg',
    '伏栓腸溶微粒膠囊 100 毫克',
    '來喜妥錠４０毫克（服樂泄麥）',
    '保栓通膜衣錠 300毫克',
    '保栓通膜衣錠 75毫克',
    '冠脂妥膜衣錠 5、10、20 毫克',
    '康肯膜衣錠 5mg / 10mg',
    '昂斯妥錠 50微克',
    '普拿疼膜衣錠５００毫克',
    '樂命達',
    '特平癲膜衣錠 (Letram)',
    '癲妥錠 200毫克',
    '癲躂錠100毫克',
    '癲通長效膜衣錠 200毫克',
    '硝基甘油錠',
    '臟得樂錠 200mg',
    '艾必克凝膜衣錠2.5毫克',
    '賽羅力錠 100毫克',
    '鈉催離持續性藥效膜衣錠 1.5 毫克',
    '驅異樂膜衣錠 5毫克',
]

# 測試題庫（分三類：LLM 容易答對 / 邊界模糊 / RAG 才有優勢）
TEST_BANK = {
    "🟢 通用知識（LLM 應答得好）": [
        ("（不指定藥品）", "阿斯匹靈的常見副作用有哪些？"),
        ("（不指定藥品）", "Warfarin 有哪些藥物交互作用需要注意？"),
        ("（不指定藥品）", "Metformin 的禁忌症是什麼？"),
        ("（不指定藥品）", "懷孕期間可以使用 Ibuprofen 嗎？"),
    ],
    "🟡 台灣藥品通用名（邊界模糊）": [
        ("保栓通膜衣錠 75毫克",        "保栓通的禁忌症有哪些？"),
        ("癲妥錠 200毫克",             "癲妥錠可以突然停藥嗎？"),
        ("普拿疼膜衣錠５００毫克",       "普拿疼每日最高劑量是多少？"),
        ("冠脂妥膜衣錠 5、10、20 毫克", "冠脂妥有哪些副作用需要注意？"),
    ],
    "🔴 具體劑量與數字（LLM 容易猜錯）": [
        ("鈉催離持續性藥效膜衣錠 1.5 毫克", "鈉催離的每日建議劑量與最大劑量各是多少？"),
        ("硝基甘油錠",                     "硝基甘油舌下錠使用後幾分鐘沒有緩解應如何處置？"),
        ("昂斯妥錠 50微克",                "昂斯妥錠的起始劑量與最大劑量是多少？"),
        ("癲通長效膜衣錠 200毫克",          "癲通長效錠每日幾次、每次幾毫克？"),
        ("樂命達",                         "樂命達的建議起始劑量與維持劑量為何？"),
        ("Esidrex 25mg",                  "Esidrex 治療高血壓的每日建議劑量範圍？"),
    ],
    "🔴 多藥交互作用（LLM 容易漏項）": [
        ("保栓通膜衣錠 75毫克",          "保栓通與哪些具體藥物併用需要特別警告？請列出所有藥物名稱。"),
        ("癲妥錠 200毫克",              "癲妥錠與哪些藥物有交互作用？對各藥物濃度的影響為何？"),
        ("臟得樂錠 200mg",              "臟得樂與 Warfarin 併用時需要注意什麼？"),
        ("冠脂妥膜衣錠 5、10、20 毫克", "冠脂妥與 Cyclosporine 或 Niacin 併用有何風險？"),
        ("特平癲膜衣錠 (Letram)",        "特平癲與哪些抗癲癇藥物有交互作用？各有何影響？"),
        ("伏栓腸溶微粒膠囊 100 毫克",    "伏栓腸溶膠囊與抗凝血劑併用需要注意什麼？"),
    ],
    "🔴 警語原文細節（RAG 才有完整內容）": [
        ("驅異樂膜衣錠 5毫克",           "驅異樂的完整警語內容為何？有哪些特殊族群需要注意？"),
        ("艾必克凝膜衣錠2.5毫克",        "艾必克凝針對腎功能不全患者有什麼特別警語？"),
        ("來喜妥錠４０毫克（服樂泄麥）",  "來喜妥錠有哪些警語是關於肝功能的？"),
        ("賽羅力錠 100毫克",             "賽羅力錠的警語中提到哪些需要停藥的情況？"),
        ("Euthyrox 100mcg",             "Euthyrox 有哪些心臟相關的警語？"),
        ("Fixuric 40mg",                "Fixuric 在腎功能不全患者的使用警語為何？"),
    ],
    "🔴 罕見或特定族群副作用（LLM 只知常見的）": [
        ("康肯膜衣錠 5mg / 10mg",        "康肯對於糖尿病患者有哪些特別需要注意的副作用？"),
        ("樂命達",                        "樂命達有哪些皮膚相關的嚴重不良反應需要立即停藥？"),
        ("昂斯妥錠 50微克",               "昂斯妥錠有哪些內分泌相關的副作用？"),
        ("癲躂錠100毫克",                 "癲躂錠對於老年患者有哪些特別的副作用警告？"),
        ("Eveness 10mg",                 "Eveness 有哪些需要立即就醫的嚴重副作用？"),
        ("硝基甘油錠",                    "硝基甘油錠除了頭痛以外還有哪些較少見的副作用？"),
    ],
}

# section 中文標籤對照
SECTION_LABEL = {
    'indications':       '適應症',
    'adverse_effects':   '副作用',
    'dosage':            '使用方式',
    'contraindications': '禁忌症',
    'interactions':      '藥物交互作用',
    'warnings':          '警語',
}

# 關鍵字路由對照表
SECTION_KEYWORDS = {
    '交互作用': ['交互作用', '併用', '合用', '藥物交互', '相互作用'],
    '副作用':   ['副作用', '不良反應', '不良事件', '副反應'],
    '禁忌症':   ['禁忌', '禁用', '不能用', '不可用'],
    '使用方式': ['劑量', '幾毫克', '幾次', '用法', '用量', '起始劑量', '最大劑量', '建議劑量'],
    '警語':     ['警語', '警告', '注意事項', '特殊族群'],
}

SYSTEM_PROMPT = """你是一位專業的藥師助理，擅長解讀藥品仿單（Package Insert）。

回答規則：
1. 若資料中有 ⚠️ 標記的警語/禁忌症，必須優先且完整呈現
2. 若問題涉及藥物安全，請特別提醒使用者諮詢醫師或藥師
3. 使用繁體中文回答
4. 若無法確定答案，請誠實說明"""


# ─────────────────────────────────────────
# 初始化模型和資料庫
# ─────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"載入 MedBERT embedding 模型... (device={device})")

tokenizer   = AutoTokenizer.from_pretrained(MODEL_PATH)
embed_model = AutoModel.from_pretrained(MODEL_PATH).to(device)
embed_model.eval()

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection    = chroma_client.get_collection(COLLECTION_NAME)
print(f"✅ ChromaDB 已連接，共 {collection.count()} 筆")

# Ollama 透過 OpenAI 相容 API 呼叫
llm_client = OpenAI(
    api_key='ollama',
    base_url=OLLAMA_BASE_URL
)
print(f"✅ {LLM_MODEL} (Ollama) client 初始化完成")


# ─────────────────────────────────────────
# Embedding 函式
# ─────────────────────────────────────────
def get_embedding(text):
    """把文字轉成 CLS token 向量（L2 normalized）"""
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        output = embed_model(**encoded)

    cls_emb = output.last_hidden_state[:, 0, :]
    cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
    return cls_emb.cpu().numpy()


# ─────────────────────────────────────────
# RAG 檢索（三條軌道）
# ─────────────────────────────────────────
def retrieve_chunks(drug_name: str, query: str) -> list[dict]:
    """
    三條軌道的 RAG 檢索：

    軌道 1：強制抓 high priority chunks
        → 確保重要的警語和禁忌症一定會出現

    軌道 2：關鍵字路由
        → 問到「劑量」就強制抓「使用方式」section
        → 問到「交互作用」就強制抓「交互作用」section

    軌道 3：語意搜尋
        → CLS embedding + cosine similarity
        → 補充上面兩條軌道沒抓到的相關資料
    """
    retrieved = []
    seen_docs = set()

    # ── 軌道 1：強制抓 high priority chunks ──
    if drug_name.strip():
        priority_result = collection.get(
            where={
                "$and": [
                    {"drug_name": {"$eq": drug_name.strip()}},
                    {"priority":  {"$eq": "high"}}
                ]
            },
            include=['documents', 'metadatas'],
            limit=TOP_K_PRIORITY
        )
        for doc, meta in zip(
            priority_result['documents'],
            priority_result['metadatas']
        ):
            if doc not in seen_docs:
                retrieved.append({
                    'text': doc, 'meta': meta,
                    'score': 1.0, 'source': 'priority'
                })
                seen_docs.add(doc)

    # ── 軌道 2：關鍵字路由 → 強制抓對應 section ──
    forced_sections = []
    for section, keywords in SECTION_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            forced_sections.append(section)

    if drug_name.strip() and forced_sections:
        for sec in forced_sections:
            sec_result = collection.get(
                where={
                    "$and": [
                        {"drug_name": {"$eq": drug_name.strip()}},
                        {"section":   {"$eq": sec}}
                    ]
                },
                include=['documents', 'metadatas'],
                limit=2
            )
            for doc, meta in zip(
                sec_result['documents'],
                sec_result['metadatas']
            ):
                if doc not in seen_docs:
                    retrieved.append({
                        'text': doc, 'meta': meta,
                        'score': 0.95, 'source': 'keyword_route'
                    })
                    seen_docs.add(doc)

    # ── 軌道 3：語意搜尋 ──
    query_emb    = get_embedding(query)
    where_filter = {"drug_name": {"$eq": drug_name.strip()}} if drug_name.strip() else None

    semantic_result = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=TOP_K_SEMANTIC,
        where=where_filter,
        include=['documents', 'metadatas', 'distances']
    )

    for doc, meta, dist in zip(
        semantic_result['documents'][0],
        semantic_result['metadatas'][0],
        semantic_result['distances'][0]
    ):
        if doc not in seen_docs:
            retrieved.append({
                'text': doc, 'meta': meta,
                'score': 1 - dist, 'source': 'semantic'
            })
            seen_docs.add(doc)

    # priority 的排在最前面，其次按分數排序
    retrieved.sort(
        key=lambda x: (x['source'] == 'priority', x['score']),
        reverse=True
    )
    return retrieved


def build_context(chunks: list[dict]) -> tuple[str, str]:
    """
    把 chunks 組裝成 LLM 的 context 文字
    超過 MAX_CONTEXT_LEN 就截斷
    """
    context_parts = []
    total_len     = 0
    seen_sources  = set()
    source_lines  = []

    for chunk in chunks:
        meta         = chunk['meta']
        sec          = SECTION_LABEL.get(meta.get('section', ''), meta.get('section', ''))
        drug         = meta.get('drug_name', '未知藥品')
        priority_tag = '⚠️ ' if meta.get('priority') == 'high' else ''
        text         = chunk['text'].strip()
        part         = f"【{priority_tag}{drug} - {sec}】\n{text}"

        total_len += len(part)
        if total_len > MAX_CONTEXT_LEN:
            break

        context_parts.append(part)
        src = meta.get('source', '?')
        if src not in seen_sources:
            source_lines.append(src)
            seen_sources.add(src)

    return '\n\n'.join(context_parts), '\n'.join(source_lines)


# ─────────────────────────────────────────
# LLM 呼叫
# ─────────────────────────────────────────
def call_llm(drug_name: str, query: str, context: str = None) -> str:
    """
    呼叫 Ollama LLM

    context=None → 純 LLM 模式（沒有仿單資料）
    context 有值 → RAG 模式（有仿單資料）
    """
    drug_hint = f"針對藥品「{drug_name}」，" if drug_name.strip() else ""

    if context:
        user_prompt = f"""以下是從藥品仿單中檢索到的相關資料：

{context}

---

使用者問題：{drug_hint}{query}

請根據以上仿單資料回答問題。"""
    else:
        user_prompt = f"使用者問題：{drug_hint}{query}"

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────
# 主查詢邏輯
# ─────────────────────────────────────────
def compare_query(drug_name: str, question: str):
    """
    同時跑 LLM Only 和 RAG 兩種模式
    回傳兩個答案和 RAG 的參考來源
    """
    if not question.strip():
        return "請輸入問題", "請輸入問題", ""

    actual_drug = "" if drug_name == '（不指定藥品）' else drug_name

    # 純 LLM 回答（不給 context）
    llm_only_answer = call_llm(actual_drug, question, context=None)

    # RAG 回答（先檢索，再給 context）
    chunks = retrieve_chunks(actual_drug, question)
    if chunks:
        context, sources = build_context(chunks)
        rag_answer = call_llm(actual_drug, question, context=context)
    else:
        rag_answer = f"⚠️ 找不到「{drug_name}」的相關仿單資料。"
        sources    = "無"

    return llm_only_answer, rag_answer, sources


# ─────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="LLM vs RAG 回答比較") as demo:
        gr.Markdown("""
        # ⚖️ LLM Only vs RAG 回答比較
        **左欄**：純 qwen2.5:7b（無仿單資料）　｜　**右欄**：qwen2.5:7b + RAG（仿單資料庫）
        > ⚠️ 本系統僅供研究測試，實際用藥請諮詢醫師或藥師
        """)

        # ── 輸入區 ──
        with gr.Row():
            drug_dropdown = gr.Dropdown(
                choices=DRUG_LIST,
                label="選擇藥品",
                value=DRUG_LIST[0],
                scale=1
            )
            question_input = gr.Textbox(
                label="輸入問題",
                placeholder="例如：這個藥有哪些禁忌症？",
                lines=2,
                scale=3
            )
            submit_btn = gr.Button("🔍 同時查詢", variant="primary", scale=1)

        # ── 測試題庫區 ──
        with gr.Accordion("📋 測試題庫（點選自動填入）", open=True):
            for category, questions in TEST_BANK.items():
                gr.Markdown(f"**{category}**")
                with gr.Row():
                    for drug, q in questions:
                        btn = gr.Button(
                            f"{drug}｜{q[:18]}{'...' if len(q) > 18 else ''}",
                            size="sm",
                            variant="secondary"
                        )
                        btn.click(
                            fn=lambda d=drug, q=q: (
                                gr.update(value=d),
                                gr.update(value=q)
                            ),
                            outputs=[drug_dropdown, question_input]
                        )

        # ── 輸出區 ──
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🤖 純 LLM 回答（無 RAG）")
                llm_output = gr.Textbox(
                    label=f"{LLM_MODEL} 直接回答",
                    lines=18,
                    interactive=False
                )
            with gr.Column():
                gr.Markdown("### 📚 RAG 增強回答（仿單資料庫）")
                rag_output = gr.Textbox(
                    label=f"{LLM_MODEL} + RAG 回答",
                    lines=18,
                    interactive=False
                )

        sources_output = gr.Textbox(
            label="📎 RAG 參考來源",
            lines=3,
            interactive=False
        )

        submit_btn.click(
            fn=compare_query,
            inputs=[drug_dropdown, question_input],
            outputs=[llm_output, rag_output, sources_output]
        )
        question_input.submit(
            fn=compare_query,
            inputs=[drug_dropdown, question_input],
            outputs=[llm_output, rag_output, sources_output]
        )

    return demo


# ─────────────────────────────────────────
# 啟動
# ─────────────────────────────────────────
if __name__ == '__main__':
    demo = build_ui()
    demo.launch(
        server_name='0.0.0.0',
        server_port=7860,
        share=True,
        inbrowser=True,
        theme=gr.themes.Soft()
    )
