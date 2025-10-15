import os
import re

# Klasör yolu,
# # Tüm txt dosyalarını bul

folder_path = r"C:\Users\n0661\OneDrive\Документы\GitHub\code4her-project\develop\knowledge_base"

files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

# Prompt için birleştirilecek string
prompt_kismi = ""

for file in files:
    file_path = os.path.join(folder_path, file)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Satır satır işle
    questions = re.findall(r'## Soru:(.*?)\n(?:\[cite_start\])?Cevap:(.*?)(?=\n## Soru:|\Z)', content, re.DOTALL)
    
    for i, (soru, cevap) in enumerate(questions, 1):
        soru = soru.strip().replace("\n", " ")
        cevap = cevap.strip().replace("\n", " ")
        prompt_kismi += f"{i}. Soru: {soru}\n   Cevap: {cevap}\n"

# Sonucu kaydet
with open("hazir_prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt_kismi)

print("Tüm txt dosyaları işlendi, hazir_prompt.txt oluşturuldu.")
print(prompt_kismi[:500], "...")  # İlk 500 karakteri göster
