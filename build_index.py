# build_index.py
import os
import pickle
from pathlib import Path

from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True)

# 一个在多语言（含中文）上还不错的模型
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_documents():
    docs = []
    for path in DATA_DIR.glob("*.txt"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append((path.name, text))
    return docs

def split_into_chunks(text, chunk_size=500, overlap=100):
    """
    简单按字符长度切块：
    - chunk_size: 每块长度
    - overlap: 相邻两块的重叠长度
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def main():
    docs = load_documents()
    if not docs:
        print("data/ 目录下没有 txt 文件，请先放一些文档进去。")
        return

    model = SentenceTransformer(EMBED_MODEL_NAME)

    all_chunks = []
    meta = []  # 保存每个 chunk 来自哪个文件、哪一段

    print("切分文档为 chunks ...")
    for fname, text in docs:
        chunks = split_into_chunks(text)
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            meta.append({"source": fname, "chunk_id": i})

    print(f"总共 {len(all_chunks)} 个文本块，开始计算向量 ...")
    embeddings = model.encode(all_chunks, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # 建立 FAISS 索引（简单的 L2 索引）
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 保存索引和文本
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(
            {
                "chunks": all_chunks,
                "meta": meta,
                "embed_model": EMBED_MODEL_NAME,
            },
            f,
        )

    print("索引构建完成 ✅")
    print(f"- 向量数: {len(all_chunks)}")
    print(f"- 索引文件: {INDEX_DIR / 'faiss.index'}")
    print(f"- 文本文件: {INDEX_DIR / 'chunks.pkl'}")

if __name__ == "__main__":
    main()
