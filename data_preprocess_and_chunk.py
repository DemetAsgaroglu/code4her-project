import os
import re
import json

DATA_DIR = r"C:\Users\beyza\Desktop\code4her-project\knowledgw_base"

# Desenler
TOPIC_PATTERN = re.compile(r'^# Konu:\s*(.*)', re.MULTILINE)  # Dosyadaki konu başlıkları
QA_BLOCK_PATTERN = re.compile(r'(## Soru:.*?)(?=(## Soru:|$))', re.DOTALL)
ANSWER_PATTERN = re.compile(r'Cevap:\s*(.*)', re.DOTALL)
CLEAN_CITE_PATTERN = re.compile(r'\[cite_start\]|\[cite:.*?\]')  # cite_start ve cite:123 gibi kısımları kaldırma

all_chunks = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".txt"):
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Dosyadaki tüm # Konu: başlıklarını ve pozisyonlarını bul
        topic_positions = [(m.start(), m.group(1).strip()) for m in TOPIC_PATTERN.finditer(text)]
        if not topic_positions:
            topic_positions = [(0, "Unknown")]  # Eğer konu yoksa

        # QA bloklarını bul
        qa_blocks = QA_BLOCK_PATTERN.findall(text)
        for block, _ in qa_blocks:
            # Soruyu al
            question_match = re.search(r'## Soru:\s*(.*)', block)
            question = question_match.group(1).strip() if question_match else ""
            # Cevabı al
            answer_match = ANSWER_PATTERN.search(block)
            if answer_match:
                answer_raw = answer_match.group(1).strip()
                # Gereksiz referansları temizle
                answer_clean = CLEAN_CITE_PATTERN.sub('', answer_raw)
                # "---\n\n# Konu:" gibi son ekleri temizle
                answer_clean = re.split(r'---\s*\n\s*# Konu:', answer_clean)[0].strip()
            else:
                answer_clean = ""

            # Topic bul: blok başından önceki en son # Konu: satırı
            block_start = text.find(block)
            topic = "Unknown"
            for pos, t in topic_positions:
                if pos <= block_start:
                    topic = t
                else:
                    break

            if question and answer_clean:
                all_chunks.append({
                    "question": question,
                    "answer": answer_clean,
                    "topic": topic,
                    "source": filename
                })

# JSON olarak kaydet
output_file = os.path.join(DATA_DIR, "processed_qa_c.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=4)

print(f"Toplam {len(all_chunks)} QA chunk oluşturuldu. JSON dosyası: {output_file}")
