import json
with open("data/slimpajama_jsonl/all_train.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:  # 只看前5行
            break
        try:
            obj = json.loads(line)
        except Exception as e:
            print("解析失败:", e, line[:200])
            continue
        print(list(obj.keys()))
        print({k: (obj[k][:80] if isinstance(obj[k], str) else obj[k]) for k in obj})
        print("-----")
