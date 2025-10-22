import json
from pathlib import Path

# === KLASÖR ===
data_folder = Path(r"C:\Users\n0661\OneDrive\Masaüstü\sidas\cleaned_data")
output_file = data_folder / "dataset.jsonl"

# === JSONL oluştur ===
with open(output_file, "w", encoding="utf-8") as out_f:
    for txt_file in data_folder.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                example = {
                    "instruction": f"{txt_file.stem} hakkında bilgi ver.",
                    "input": "",
                    "output": text
                }
                json.dump(example, out_f, ensure_ascii=False)
                out_f.write("\n")

print(f"✅ {output_file} oluşturuldu.")
