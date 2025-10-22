import os
import json
import re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import platform

# --- Terminal temizle ---
os.system("cls" if platform.system() == "Windows" else "clear")

# --- Ayarlar ---
data_folder = Path(r"C:\Users\n0661\OneDrive\Masaüstü\sidas\cleaned_data")
embedding_model_path = r"C:\Users\n0661\OneDrive\Masaüstü\sidas\model\all-MiniLM-L6-v2"
memory_file = "chat_memory.json"
max_sentence_len = 300
top_k = 5

# --- Metinleri parçalara böl ---
def split_text(text, max_len=max_sentence_len):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    for p_idx, para in enumerate(paragraphs):
        sentences = re.split(r'(?<=[.!?])\s+', para)
        for s_idx, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) > max_len:
                for i in range(0, len(sent), max_len):
                    chunks.append((sent[i:i+max_len], f"Paragraf {p_idx+1} | Cümle {s_idx+1}"))
            else:
                chunks.append((sent, f"Paragraf {p_idx+1} | Cümle {s_idx+1}"))
    return chunks

# --- Dosyaları oku ---
dataset, chunks_list, sources = [], [], []
for txt_file in data_folder.glob("*.txt"):
    with open(txt_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
        chunks = split_text(content)
        for text, src in chunks:
            dataset.append({"text": text, "source": src})
            chunks_list.append(text)
            sources.append(f"{txt_file.name} | {src}")

print(f"{len(dataset)} chunk yüklendi ve işlendi.")

# --- Embedding ---
embedder = SentenceTransformer(embedding_model_path)
chunk_embeddings = embedder.encode(chunks_list, convert_to_numpy=True, show_progress_bar=True)
chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)

# --- Türkçe destekli mT5 modeli ---
t5_model_name = "google/mt5-base"  # dil desteği geniş, Türkçe dahil
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# --- Cevap üretici ---
def generate_answer(query, context):
    # mT5'e uygun, açık yönerge
    prompt = (
        f"Aşağıdaki bilgilere dayanarak soruya Türkçe olarak kısa ve net bir cevap ver.\n\n"
        f"Bilgiler: {context}\n\n"
        f"Soru: {query}\n\n"
        f"Cevap:"
    )

    inputs = t5_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    outputs = t5_model.generate(
        **inputs,
        max_length=150,
        num_beams=4,
        early_stopping=True,
        repetition_penalty=1.2
    )

    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # mT5 bazen "Cevap:" veya "Soru:" kelimelerini geri döndürebilir
    answer = answer.replace("Cevap:", "").replace("Soru:", "").strip()
    return answer

# --- Chat hafızası ---
if os.path.exists(memory_file):
    with open(memory_file, "r", encoding="utf-8") as f:
        chat_memory = json.load(f)
else:
    chat_memory = []

def save_memory():
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(chat_memory, f, ensure_ascii=False, indent=2)

# --- RAG Chat fonksiyonu ---
def rag_chat(query, k=top_k):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    sims = np.dot(chunk_embeddings, q_emb.T).squeeze()
    top_k_idx = sims.argsort()[::-1][:k]
    top_k_scores = sims[top_k_idx]

    similarity_threshold = 0.30  # sabit eşik
    selected_chunks = [
        chunks_list[idx]
        for idx, score in zip(top_k_idx, top_k_scores)
        if score >= similarity_threshold
    ]

    context_text = " ".join(selected_chunks) if selected_chunks else "Veri setinde ilgili bilgi bulunamadı."
    answer_text = generate_answer(query, context_text)

    chat_memory.append({"soru": query, "cevap": answer_text})
    chat_memory[:] = chat_memory[-20:]
    save_memory()
    return f"💬 CEVAP:\n{answer_text}"

# --- Komut satırı arayüzü ---
print("🧠 Dataset + RAG Chatbot hazır! ('çık' yazınca kapanır)")
print("💡 '!test <soru>' komutuyla en benzer verileri görebilirsin.\n")

while True:
    query = input("Sen: ").strip()
    if not query:
        continue
    if query.lower() in ["çık", "exit", "quit"]:
        print("Bot: Görüşürüz 👋")
        break
    if query.startswith("!test "):
        q = query.replace("!test ", "").strip()
        q_emb = embedder.encode([q], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        sims = np.dot(chunk_embeddings, q_emb.T).squeeze()
        top_idx = sims.argsort()[::-1][:top_k]
        print("\n🔍 Benzerlik Testi:")
        for i, idx in enumerate(top_idx):
            print(f"{i+1}. {chunks_list[idx][:70]}... | {sources[idx]} | Cosine: {sims[idx]:.2f}")
        print()
        continue

    print(rag_chat(query))
