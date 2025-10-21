import os #transformer hem keras da hem pythorch da kullanılıyor tensorflow yani keras kullanmasın diye onu ayarlardan kapatıyoruz.
os.environ["USE_TF"] = "0"  # TensorFlow tamamen kapatılır
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TensorFlow uyarılarını sustur
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Tokenizer uyarılarını engeller

import json
import torch
from sentence_transformers import SentenceTransformer

# 🔹 Dosya yolları
DATA_DIR = r"C:\Users\beyza\Desktop\code4her-project\knowledgw_base"
INPUT_FILE = os.path.join(DATA_DIR, "processed_qa_c.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "qa_embeddings.json")

# 🔹 Cihaz seçimi (GPU varsa kullanılır)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Kullanılan cihaz: {device}")

# 🔹 Model yükle (tamamen PyTorch modunda)
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# 🔹 JSON dosyasını oku
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

# 🔹 Her soru-cevap için embedding oluştur
for item in qa_data:
    text = item['question'] + " " + item['answer']
    embedding = model.encode(text, show_progress_bar=False, convert_to_numpy=True)
    item['embedding'] = embedding.tolist()  # JSON’a kaydedebilmek için listeye çevir

# 🔹 Embedding’li veriyi kaydet
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(qa_data, f, ensure_ascii=False, indent=4)

print(f"Toplam {len(qa_data)} embedding oluşturuldu. Dosya: {OUTPUT_FILE}")
