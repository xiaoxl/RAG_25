# rag_query.py
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("index")


def load_index():
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        data = pickle.load(f)
    return index, data


def embed_query(query, model_name):
    model = SentenceTransformer(model_name)
    vec = model.encode([query])
    return np.array(vec).astype("float32")


# 这里你自己实现：可以是 OpenAI，也可以是本地模型
def call_llm(prompt: str) -> str:
    """
    TODO: 在这里接你自己的大模型
    比如：
    - OpenAI ChatCompletion
    - 本地 Ollama / vLLM / LM Studio
    """
    # 这里先用个假的占位
    return "【这里应该是大模型的回答，你可以在 call_llm 里接 OpenAI 等】"


def build_prompt(query: str, retrieved_chunks):
    context_text = "\n\n".join([f"[文档片段 {i + 1}]\n{ch}" for i, ch in enumerate(retrieved_chunks)])

    prompt = f"""你是一个检索增强问答助手。下面是与用户问题相关的文档片段，请优先基于这些内容回答。如果文档中没有答案，可以据实说明不知道，不要编造。

用户问题：
{query}

相关文档：
{context_text}

请用中文给出清晰、简洁的回答：
"""
    return prompt


def rag_query(query: str, top_k: int = 5):
    index, data = load_index()
    chunks = data["chunks"]
    meta = data["meta"]
    embed_model = data["embed_model"]

    qvec = embed_query(query, embed_model)
    # faiss 搜索
    distances, indices = index.search(qvec, top_k)
    indices = indices[0]

    retrieved_chunks = [chunks[i] for i in indices]
    retrieved_meta = [meta[i] for i in indices]

    print("检索到的相关片段：")
    for i, (ch, m) in enumerate(zip(retrieved_chunks, retrieved_meta), 1):
        print(f"\n=== 片段 {i} | 文件: {m['source']} | chunk_id: {m['chunk_id']} ===")
        print(ch[:300] + ("..." if len(ch) > 300 else ""))

    prompt = build_prompt(query, retrieved_chunks)
    # answer = call_llm(prompt)
    answer = prompt
    print("\n=== RAG 回答 ===")
    print(answer)


if __name__ == "__main__":
    while True:
        q = input("\n请输入你的问题（或 q 退出）：")
        if q.strip().lower() in {"q", "quit", "exit"}:
            break
        rag_query(q)
