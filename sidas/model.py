import os
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz

# === AYARLAR ===
data_folder = Path(r"C:\Users\n0661\OneDrive\Masaüstü\sidas\cleaned_data")
jsonl_path = data_folder / "dataset.jsonl"
embedding_model_path = r"C:\Users\n0661\OneDrive\Masaüstü\sidas\model\all-MiniLM-L6-v2"
memory_file = "chat_memory.json"

# === 1️⃣ Dataset yükle ===
dataset, questions, answers = [], [], []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        dataset.append(item)
        questions.append(item["instruction"])
        answers.append(item["output"])

print(f"{len(dataset)} soru-cevap yüklendi.")

# === 2️⃣ Embedding modeli & FAISS ===
embedder = SentenceTransformer(embedding_model_path)
question_embeddings = embedder.encode(questions, convert_to_numpy=True)

embedding_dim = question_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(question_embeddings)
print("FAISS indeksi oluşturuldu.")

# === 3️⃣ Chat Memory ===
if os.path.exists(memory_file):
    with open(memory_file, "r", encoding="utf-8") as f:
        chat_memory = json.load(f)
else:
    chat_memory = []

def save_memory():
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(chat_memory, f, ensure_ascii=False, indent=2)

# === 🔎 4️⃣ Benzerlik kontrol testi ===
def debug_check(query, k=5):
    """Girilen sorgunun dataset içindeki en benzer 5 kaydını ve skorlarını gösterir."""
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    print("\n🔍 Benzerlik Testi:")
    for rank, idx in enumerate(I[0]):
        ratio = fuzz.ratio(query.lower(), questions[idx].lower())
        print(f"{rank+1}. '{questions[idx]}' | Benzerlik: {ratio}")
    print()

# === 5️⃣ Chat Fonksiyonu ===
def rag_chat(query, k=3, similarity_threshold=0.55):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)

    best_idx = None
    best_score = 0.0

    for idx in I[0]:
        # FAISS: küçük mesafe = yüksek benzerlik
        emb_score = 1 / (1 + D[0][list(I[0]).index(idx)])  # normalize
        fuzz_score = fuzz.token_sort_ratio(query.lower(), questions[idx].lower()) / 100

        # iki skorun ortalaması (embedding %70, fuzz %30 etkili)
        final_score = (emb_score * 0.7) + (fuzz_score * 0.3)

        if final_score > best_score:
            best_score = final_score
            best_idx = idx

    if best_score >= similarity_threshold:
        answer = answers[best_idx]
        answer = " ".join(answer.split(".")[:3]) + "."  # ilk birkaç cümleyle sınırla
        chat_memory.append((query, answer))
        chat_memory[:] = chat_memory[-20:]
        save_memory()
        return f"(📚 Datasetten gelen cevap)\n{answer}\n\n🔎 Benzerlik skoru: {best_score:.2f}"
    else:
        return "Üzgünüm, bu konuda elimde veri bulunamadı."


# === 6️⃣ Sohbet döngüsü ===
print("🧠 Dataset Tabanlı Chatbot hazır! ('çık' yazınca kapanır)")
print("💡 Test için: '!test şiddet' yazarak benzerlik skorlarını görebilirsin.\n")

while True:
    query = input("Sen: ").strip()
    if not query:
        continue
    if query.lower() in ["çık", "exit", "quit"]:
        print("Bot: Görüşürüz 👋")
        break

    # 🔎 Test modu
    if query.startswith("!test "):
        q = query.replace("!test ", "").strip()
        debug_check(q)
        continue

    # Normal sohbet
    answer = rag_chat(query)
    print("Bot:", answer)
