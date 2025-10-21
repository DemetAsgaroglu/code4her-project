# retrieval.py
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# --- MODELİ YÜKLE ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- JSON EMBEDDINGLERİ YÜKLE ---
DATA_DIR = r"C:\Users\beyza\Desktop\code4her-project\knowledgw_base"
INPUT_FILE = os.path.join(DATA_DIR, "qa_embeddings.json")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Embedding'leri numpy array'e dönüştür
embeddings = np.array([item["embedding"] for item in data]).astype("float32")

# --- FAISS İNDEX OLUŞTUR ---
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(f"FAISS index oluşturuldu. Toplam {index.ntotal} embedding eklendi.")

# --- SORU AL VE ARAMA YAP ---
def retrieve_documents(query, top_k=5):
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        results.append({
            "question": data[idx]["question"],
            "answer": data[idx]["answer"],
            "topic": data[idx]["topic"],
            "source": data[idx]["source"]
        })
    return results

if __name__ == "__main__":
    query = input("Bir soru sor: ")
    docs = retrieve_documents(query)
    print("\n🔹 En alakalı dökümanlar:\n")
    for i, d in enumerate(docs, 1):
        print(f"{i}. [{d['topic']}] {d['question']} → {d['answer']}")
