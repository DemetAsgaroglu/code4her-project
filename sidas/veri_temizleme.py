import os
import re
import shutil

# Kaynak klasör
source_folder = r"C:\Users\n0661\OneDrive\Masaüstü\sidas\knowledgw_base"
# Temizlenmiş verilerin kaydedileceği hedef klasör
target_folder = r"C:\Users\n0661\OneDrive\Masaüstü\sidas\cleaned_data"

# Hedef klasör varsa silip tekrar oluştur
if os.path.exists(target_folder):
    shutil.rmtree(target_folder)
os.makedirs(target_folder, exist_ok=True)

# Temizleme fonksiyonu
def clean_text(text):
    # Baş ve sondaki boşlukları kaldır
    text = text.strip()
    # [cite_start] ve [cite: sayı] kalıplarını temizle
    text = re.sub(r'\[cite_start\]', '', text)
    text = re.sub(r'\[cite:\s*\d+\]', '', text)
    # Fazla boşlukları tek boşluk yap
    text = re.sub(r'\s+', ' ', text)
    return text

# Tüm klasör ve dosyaları dolaş
for root, dirs, files in os.walk(source_folder):
    # Hedef klasörde aynı klasör yapısını oluştur
    relative_path = os.path.relpath(root, source_folder)
    target_path = os.path.join(target_folder, relative_path)
    os.makedirs(target_path, exist_ok=True)

    for file in files:
        if file.endswith(".txt"):
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_path, file)

            # Dosyayı güvenli şekilde oku
            try:
                with open(source_file, "r", encoding="utf-8-sig") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Eğer UTF-8 ile okunmazsa Latin1 ile oku
                with open(source_file, "r", encoding="latin1") as f:
                    content = f.read()

            # Temizle
            cleaned_content = clean_text(content)

            # Temizlenmiş içeriği kaydet
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(cleaned_content)

print("Temizleme tamamlandı! Temizlenmiş dosyalar:", target_folder)
