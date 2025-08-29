# save as tools/prepare_slimpajama.py
from datasets import load_dataset
import json, os

OUT_DIR = "data/slimpajama_jsonl"
SPLITS = {"train": "train", "valid": "validation"}

os.makedirs(OUT_DIR, exist_ok=True)

def dump(split_name, hf_split, shard_size=200_000):
    ds = load_dataset("/projects/llama-cpt/data/SlimPajama-6B/", split=hf_split, streaming=True)
    n, shard_idx = 0, 0
    fw = open(os.path.join(OUT_DIR, f"{split_name}-{shard_idx:03d}.jsonl"), "w", encoding="utf-8")
    for ex in ds:
        text = ex.get("text") or ex.get("content") or ""
        if not text or text.strip() == "":
            continue
        fw.write(json.dumps({"text": text.strip()}, ensure_ascii=False) + "\n")
        n += 1
        if n % shard_size == 0:
            fw.close()
            shard_idx += 1
            fw = open(os.path.join(OUT_DIR, f"{split_name}-{shard_idx:03d}.jsonl"), "w", encoding="utf-8")
    fw.close()

for k, v in SPLITS.items():
    dump(k, v)
print("done")

