"""
SIDAS Veri Yukleme ve Bolme
Gorev: knowledgw_base klasorundeki .txt dosyalarini yukle ve chunklara ayir
Faiss veritabanı oluştur ve embedding sonuçlarını yükle.

"""

# CRITICAL: TensorFlow'u devre disi birak (import'lardan ONCE)
import os
os.environ['USE_TF'] = 'NO'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import re
import shutil
import torch
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from collections import Counter, defaultdict

# Reranker (CrossEncoder) için
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except Exception:
    RERANKER_AVAILABLE = False

# Ortam hazırlığı
load_dotenv()

# ADIM 1: TUM DOSYALARI YUKLE
print("ADIM 1: Dosya Yukleme")
file_paths = [
    "knowledgw_base/6284_sayili_kanun.txt",
    "knowledgw_base/gumushane_el_kitabi.txt",
    "knowledgw_base/KADES_bilgilendirme.txt",
    "knowledgw_base/kadin_dayanisma_vakfi_rehber.txt",
    "knowledgw_base/mor_cati_rehber.txt",
    "knowledgw_base/sıdas.txt",
    "knowledgw_base/sonim_yonetmelik.txt"
]

all_documents = []
failed_files = []

print(f"Yuklenecek dosya sayisi: {len(file_paths)}\n")

for path in file_paths:
    print(f"Yukleniyor: {path}")
    try:
        loader = TextLoader(path, encoding="utf-8")
        documents = loader.load()
        # TextLoader tipik olarak bir liste döner; her Document metadata içerir.
        # Eğer metadata'da source yoksa ekleyelim.
        for d in documents:
            if not d.metadata:
                d.metadata = {}
            d.metadata['source'] = d.metadata.get('source', path)
        all_documents.extend(documents)
        # documents[0] genelde tüm içeriği tutar; güvenlik için toplam karakter uzunluğu gösterelim
        total_chars = sum(len(d.page_content) for d in documents)
        print(f"  Basarili! Toplam karakter (dokumanlar): {total_chars}")
    except FileNotFoundError:
        print(f"  UYARI: Dosya bulunamadi, atlanıyor...")
        failed_files.append(path)
        continue
    except Exception as e:
        print(f"  HATA: {e}")
        failed_files.append(path)
        continue


print(f"YUKLEME SONUCLARI:")
print(f"  Basarili: {len(all_documents)} dokuman")
print(f"  Basarisiz: {len(failed_files)} dosya")
if failed_files:
    print(f"  Hatali dosyalar: {failed_files}")

if not all_documents:
    print("\nHATA: Hic dokuma yuklenmedi!")
    exit(1)


# ADIM 2: METIN BOLME (OPTIMIZED)

print("adım 2: Metin Bolme Islemi OPTIMIZE")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1100,
    chunk_overlap=200,
    length_function=len,
    separators=[
        "\n\n## Soru:",
        "\n---\n",
        "\n# Konu:",
        "\n\n",
        "\n",
        " "
    ],
    is_separator_regex=False
)

print(f"\nSplitter Ayarlari:")
print(f"   - Chunk Size: {text_splitter._chunk_size} karakter")
print(f"   - Chunk Overlap: {text_splitter._chunk_overlap} karakter")
print(f"   - Separator Oncelikleri:")
for i, sep in enumerate(text_splitter._separators[:6], 1):
    sep_display = sep.replace("\n", "\\n")
    print(f"      {i}. '{sep_display}'")

# split_documents döndürürken metadata korunur ancak chunklara ek metadata ekleyelim (source, chunk_id)
try:
    raw_chunks = text_splitter.split_documents(all_documents)

    # Çok kısa chunk'ları temizle
    min_length = 60
    original_count = len(raw_chunks)

    removed_chunks = []
    clean_chunks = []
    # chunk sayısını dosya bazında numaralandırmak için sayaç
    per_source_counter = defaultdict(int)

    for idx, chunk in enumerate(raw_chunks):
        chunk_text = chunk.page_content.strip()
        if len(chunk_text) <= min_length:
            removed_chunks.append({
                'index': idx + 1,
                'length': len(chunk_text),
                'content': chunk_text[:50]
            })
            continue

        # metadata kaynak ve chunk_id ekle
        src = chunk.metadata.get('source', 'Bilinmiyor')
        per_source_counter[src] += 1
        chunk.metadata['source'] = src
        chunk.metadata['chunk_id'] = f"{os.path.basename(src)}__{per_source_counter[src]}"
        #  kısa özet veya ilk 200 karakter metadaya koymak istiyorsan ekle
        chunk.metadata['summary_preview'] = chunk_text[:200]
        clean_chunks.append(chunk)

    chunks = clean_chunks
    print(f"BOLME SONUCLARI:")
    print(f"  Orijinal chunk: {original_count}")
    print(f"  Temizlenen: {len(removed_chunks)}")
    print(f"  Final chunk: {len(chunks)}")

    # Dosya bazinda chunk dagilimi
    source_dist = Counter(chunk.metadata.get('source', 'Bilinmiyor') for chunk in chunks)

    print(f"\nDOSYA BAZINDA CHUNK DAGILIMI:")
    for source, count in source_dist.most_common():
        print(f"  {os.path.basename(source)}: {count} chunk")


    print("ILK 3 CHUNK ORNEKLEME")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\nCHUNK #{i}")
        print(f"  Kaynak: {os.path.basename(chunk.metadata.get('source', 'Bilinmiyor'))}")
        print(f"  Uzunluk: {len(chunk.page_content)} karakter")
        print(f"  Icerik: {chunk.page_content[:200]}...")

    # Istatistikler
    print("CHUNK ISTATISTIKLERI")

    chunk_lengths = [len(chunk.page_content) for chunk in chunks]
    print(f"  Ortalama: {sum(chunk_lengths) / len(chunk_lengths):.0f} karakter")
    print(f"  En kucuk: {min(chunk_lengths)} karakter")
    print(f"  En buyuk: {max(chunk_lengths)} karakter")

    soru_count = sum(1 for chunk in chunks if "## Soru:" in chunk.page_content)
    print(f"  '## Soru:' iceren: {soru_count} ({soru_count/len(chunks)*100:.1f}%)")

except Exception as e:
    print(f"HATA: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ADIM 2.2: KAYNAK REFERANS ANALIZI
print("KAYNAK REFERANS ANALIZI")

cite_pattern = r'\[cite:\s*(\d+)\]'
all_cites = []

for chunk in chunks:
    found_cites = re.findall(cite_pattern, chunk.page_content)
    all_cites.extend([int(c) for c in found_cites])

if all_cites:
    unique_cites = sorted(set(all_cites))
    citation_counts = Counter(all_cites)

    print(f"\n  Toplam referans: {len(all_cites)}")
    print(f"  Benzersiz cite: {len(unique_cites)}")
    print(f"  Cite aralik: {min(unique_cites)} - {max(unique_cites)}")

    print(f"\n  En cok kullanilan cite'lar:")
    for cite, count in citation_counts.most_common(5):
        print(f"    [cite: {cite}] -> {count} kez")

    cite_start_count = sum(1 for chunk in chunks if "[cite_start]" in chunk.page_content)
    print(f"\n  '[cite_start]' iceren chunk: {cite_start_count}")
else:
    print("\n  UYARI: Hic cite referansi yok!")


# ADIM 3: EMBEDDING MODELI
FAISS_DB_PATH = "./faiss_db"
embedding_model_name = "intfloat/multilingual-e5-base"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nCihaz: {device.upper()}")

start_time = time.time()

try:
    model_kwargs = {'device': device}

    if device == 'cuda':
        encode_kwargs = {'normalize_embeddings': False, 'batch_size': 32}
    else:
        encode_kwargs = {'normalize_embeddings': False, 'batch_size': 16}

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    load_time = time.time() - start_time
    print(f"  Basarili! Sure: {load_time:.2f}s")

except Exception as e:
    print(f"  HATA embedding yukleme: {e}")
    # fallback to original model if load fails
    try:
        print("  Deneme: fallback olarak 'paraphrase-multilingual-mpnet-base-v2' yukleniyor...")
        embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print("  Fallback basarili.")
    except Exception as e2:
        print("  FATAL: Tum embedding yukleme girişimleri başarısız:", e2)
        exit(1)

# Ornek embedding testi
if len(chunks) >= 1:
    try:
        print(f"\nOrnek embedding testi...")
        sample_chunk = chunks[0]
        sample_vector = embeddings.embed_query(sample_chunk.page_content)
        print(f"  Vektor boyutu: {len(sample_vector)} boyut")
        print(f"  Ilk 5 eleman: {sample_vector[:5]}")
    except Exception as e:
        print("  UYARI: Ornek embedding testi sırasında hata:", e)


# ADIM 4: FAISS VERITABANI
print(f"Konum: {FAISS_DB_PATH}")
print(f"Chunk sayisi: {len(chunks)}")
print(f"Bu islem birkac dakika surebilir...\n")

try:
    # Eski veritabanini sil
    if os.path.exists(FAISS_DB_PATH):
        shutil.rmtree(FAISS_DB_PATH)
        print("  Eski veritabani silindi")

    start_time_db = time.time()

    # FAISS olustururken chunks zaten Document objeleri ve metadata içeriyor.
    # Böylece metadatayı embedding ile birlikte saklamış oluruz (RAG için gerekli).
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Kaydet
    vectorstore.save_local(FAISS_DB_PATH)

    db_time = time.time() - start_time_db

    # ntotal bilgisi güvenli şekilde alalım
    try:
        total_chunks = int(vectorstore.index.ntotal)
    except Exception:
        total_chunks = len(chunks)

    print(f"  Basarili!")
    print(f"  Veritabanindaki chunk: {total_chunks}")
    print(f"  Kaydetme suresi: {db_time:.2f}s")

except Exception as e:
    print(f"  HATA: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# ADIM 5: TEST SORGULAMA (KAPSAMLI  RERANKING)
# Daha çeşitli ve bağlamsal test sorguları ekle
test_queries = [
    "6284 sayili kanun nedir?",
    "SONIM nedir?",
    "Koruma karari nasil alinir?",
    "Maddi yardim alabilir miyim?",
    "KADES uygulaması nedir?",
    "Şiddet mağduru kadın hangi kurumlara başvurabilir?",
    "Mor Çatı hangi illerde hizmet verir?",
    "Kadın dayanışma vakfı ne tür destekler sağlar?"
]

# Reranker var ise hazırla
if RERANKER_AVAILABLE:
    try:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
        print("Reranker yuklendi: cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        print("Reranker yuklenemedi, reranking atlanacak:", e)
        reranker = None
else:
    print("Reranker paketine ulasilamadi; reranking atlanacak.")
    reranker = None

def do_rerank(query: str, docs, top_k: int = 2):
    if not reranker:
        return docs[:top_k]
    # prepare pairs (query, doc_text)
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, s in ranked][:top_k]

for query in test_queries:
    print(f"\n Soru: {query}")

    try:
        # önce genişçe retrieve et (k=8) -> sonra rerank ile en uygunları al
        initial_results = vectorstore.similarity_search(query, k=8)

        # rerank ile top 2 al
        final_results = do_rerank(query, initial_results, top_k=2)

        for i, doc in enumerate(final_results, 1):
            print(f"\n  [Sonuc {i}]")
            src = os.path.basename(doc.metadata.get('source', 'Bilinmiyor'))
            print(f"  Kaynak: {src}")
            # Cite referanslarini bul
            citations = re.findall(r'\[cite:\s*(\d+)\]', doc.page_content)
            if citations:
                print(f"  Cite: [{', '.join(citations)}]")
            # Soruyu bul (eğer varsa)
            if "## Soru:" in doc.page_content:
                soru_match = re.search(r'## Soru: ([^\n]+)', doc.page_content)
                if soru_match:
                    print(f"  Soru: {soru_match.group(1)[:120]}...")
            # Metadata preview varsa göster
            preview = doc.metadata.get('summary_preview') or doc.page_content[:200]
            print(f"  Icerik: {preview}...")
    except Exception as e:
        print(f"  HATA (sorgulama): {e}")


# FINAL OZET
print(" indexleme tamamlandı!")

print(f"\nBilgi Bankasi Ozeti:")
print(f"  Yuklenen dokuman sayisi: {len(all_documents)}")
print(f"  Basarisiz dosya: {len(failed_files)}")
print(f"  Toplam chunk: {len(chunks)}")
print(f"  Soru-cevap chunk: {soru_count}")
print(f"  Veritabani: {FAISS_DB_PATH}")
print(f"  Model: {embedding_model_name}")
# embedding çıktısı sample_vector değişkeni varsa boyut yaz
try:
    print(f"  Embedding boyutu: {len(sample_vector)} boyut")
except Exception:
    print(f"  Embedding boyutu: bilinmiyor")
if all_cites:
    print(f"  Benzersiz cite: {len(set(all_cites))}")

# Dosya bazinda ozet
print(f"\nDosya Bazinda Ozet:")
for source, count in source_dist.most_common():
    filename = os.path.basename(source)
    percentage = (count / len(chunks)) * 100 if len(chunks) > 0 else 0
    print(f"  {filename}: {count} chunk ({percentage:.1f}%)")
