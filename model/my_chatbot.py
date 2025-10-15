from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="gemma3")

# Hazır prompt'u oku
with open("hazir_prompt.txt", "r", encoding="utf-8") as f:
    knowledge_text = f.read()

# Prompt template oluştur
template = """Bu bilgilerden yola çıkarak soruyu cevapla:

Bilgi: {knowledge}

Soru: {question}
Cevap:"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

# Örnek soru
response = chain.invoke({
    "knowledge": knowledge_text,
    "question": "şönim nedir?"
})

print(response)



