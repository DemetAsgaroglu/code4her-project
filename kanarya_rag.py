import os

os.environ['USE_TF'] = 'NO'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("\nSIDAS - Kadina Yonelik Siddet Danisma Sistemi")

# Retriever yukleme
print("\n[1/2] RETRIEVER")

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

FAISS_DB_PATH = "./faiss_db"
embedding_model_name = "intfloat/multilingual-e5-base"

embedding = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={'normalize_embeddings': False, 'batch_size': 8}
)
print("Embedding yuklendi")

vectorstore = FAISS.load_local(
    FAISS_DB_PATH,
    embedding,
    allow_dangerous_deserialization=True
)
print("FAISS yuklendi")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
print("Retriever hazir")

# LLM yukleme
print("\n[2/2] LLM")

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "asafaya/kanarya-2b"
print(f"Model: {model_id} (CPU)")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("Tokenizer yuklendi")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
print("Model yuklendi")

# Prompt sablonu
PROMPT = """Aşağıdaki bilgiyi kullanarak soruyu kısa cevapla.

Bilgi: {context}

Soru: {question}

Cevap:"""


def rag_cevap_uret(soru: str):
    """RAG tabanli cevap uretme fonksiyonu"""

    print(f"\n{soru}")

    try:
        # Dokuman arama
        print("\n[1/3] Arama...")
        docs = retriever.invoke(soru)

        if not docs:
            return {"soru": soru, "cevap": "Bilgi yok.", "kaynaklar": []}

        print(f"Bulunan: {len(docs)} dokuman")

        # Context hazirlama
        context = docs[0].page_content[:200] # Sadece ilk dok, 200 kar

        # Prompt hazirlama
        prompt = PROMPT.format(context=context, question=soru)

        # Tokenizasyon
        print("\n[2/3] Token...")
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=300,
            truncation=True,
            padding=True
        )

        print(f"Token sayisi: {inputs['input_ids'].shape[1]}")

        # Metin uretme
        print("\n[3/3] Uretim...")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Cevap cikartma
        full = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Cevap temizleme
        if "Cevap:" in full:
            cevap = full.split("Cevap:")[-1].strip()
        else:
            cevap = full[len(prompt):].strip()

        # Ilk paragraf veya cumleyi al
        if "\n\n" in cevap:
            cevap = cevap.split("\n\n")[0]

        # Soru tekrari varsa kes
        if "Soru:" in cevap:
            cevap = cevap.split("Soru:")[0].strip()

        # Maksimum uzunluk kontrolu
        if len(cevap) > 250:
            son_nokta = cevap[:250].rfind('.')
            if son_nokta > 100:
                cevap = cevap[:son_nokta + 1]
            else:
                cevap = cevap[:250] + "..."

        # Kaynak ve uyari ekleme
        kaynak = os.path.basename(docs[0].metadata.get('source', '?'))
        cevap += f"\n\n[Kaynak: {kaynak}]"
        cevap += "\nUYARI: Bu bilgi yasal tavsiye niteligi tasimaz."

        print("Hazir")

        return {
            "soru": soru,
            "cevap": cevap,
            "kaynaklar": [os.path.basename(d.metadata.get('source', '?')) for d in docs]
        }

    except Exception as e:
        print(f"\nHata: {e}")
        return {"soru": soru, "cevap": f"Hata: {e}", "kaynaklar": []}


# Test modu

print("TEST")
test = [
    "6284 sayılı kanun neyi düzenler?",
    "KADES nasıl çalışır?",
    "SONİM nedir?",
]

for i, soru in enumerate(test, 1):
    print(f"\nTEST {i}/{len(test)}")
    print("-" * 70)

    sonuc = rag_cevap_uret(soru)

    print("\nCEVAP:")
    print(sonuc['cevap'])

print("\nTest tamamlandi")

#kullanıcı ile canlı sohbet
def interaktif_mod():

    print("SIDAS - INTERAKTIF MOD")
    print("""
Merhaba! Ben SIDAS, kadina yonelik siddetle ilgili sorularinizi 
yanitlayan bir yapay zeka asistaniyim.

KOMUTLAR:
   - Soru sorun (ornek: "6284 kanunu nedir?")
   - 'yardim' -> Ornek sorular
   - 'exit' veya 'cikis' -> Cikis

UYARI: Verdigim bilgiler yasal tavsiye niteligi tasimaz.
    """)

    ornek_sorular = [
        "6284 sayılı kanun neyi düzenler?",
        "KADES uygulaması nasıl çalışır?",
        "Koruma kararı nasıl alınır?",
        "ŞÖNİM nedir?",
        "Geçici koruma kararı nedir?",
        "Tedbir kararları nelerdir?",
        "Şiddet türleri nelerdir?",
        "Maddi yardım alabilir miyim?",
    ]

    soru_sayisi = 0

    while True:
        try:
            print("\n" + "-" * 70)
            kullanici_girisi = input("Sorunuz: ").strip()

            if kullanici_girisi.lower() in ['exit', 'çıkış', 'quit', 'q']:
                print("\nGuvenle kalin. Cikis yapiliyor...")
                break

            if not kullanici_girisi:
                print("UYARI: Lutfen bir soru yazin.")
                continue

            if kullanici_girisi.lower() in ['yardım', 'yardim', 'help', '?']:
                print("\nORNEK SORULAR:")
                for i, soru in enumerate(ornek_sorular, 1):
                    print(f"   {i}. {soru}")
                continue

            soru_sayisi += 1

            print("\nAraniyor...")
            sonuc = rag_cevap_uret(kullanici_girisi)

            print("\nSIDAS:")
            print("-" * 70)
            print(sonuc['cevap'])
            print("-" * 70)

            if sonuc.get('kaynaklar'):
                print(f"\nKaynaklar: {', '.join(sonuc['kaynaklar'])}")

        except KeyboardInterrupt:
            print("\n\nGuvenle kalin. Cikis yapiliyor...")
            break
        except Exception as e:
            print(f"\nBir hata olustu: {e}")
            print("Lutfen tekrar deneyin.")


    print(f"OTURUM ISTATISTIKLERI:")
    print(f"Toplam soru: {soru_sayisi}")
    print("\nTesekkurler! SIDAS'i kullandiginiz icin.")


# Interaktif modu baslat
interaktif_mod()