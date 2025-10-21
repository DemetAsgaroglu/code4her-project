from transformers import AutoTokenizer, AutoModelForCausalLM
from retrieval import retrieve_documents
import torch

# --- MODELİ YÜKLE (CPU) ---
model_name = "tiiuae/Falcon-E-3B-Instruct"  # Daha küçük model, CPU uyumlu
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=None)  # CPU kullanımında None

# --- SORU AL VE RETRIEVE ---
query = input("Bir soru sor: ")
docs = retrieve_documents(query)

# --- RETRIEVAL SONUÇLARINI BİRLEŞTİR ---
context = "\n".join([f"{d['question']} → {d['answer']}" for d in docs])
prompt = f"Soru: {query}\nBilgi: {context}\nCevap:"

# --- MODELLE ÜRETİM ---
inputs = tokenizer(prompt, return_tensors="pt")

# ⚡ CPU modeli için token_type_ids varsa kaldır
if "token_type_ids" in inputs:
    del inputs["token_type_ids"]

# CPU’da RAM kullanımını azaltmak için no_grad kullan
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n🔹 Cevap:\n", answer)
