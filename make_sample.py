from datasets import load_dataset
import json

ds = load_dataset("CCRss/arXiv_dataset", split="train", streaming=True)
n = 0
with open("sample_cs_100.json", "w") as f:
    for row in ds:
        if any(c.startswith("cs.") for c in row["categories"].split()):
            f.write(json.dumps(row, default=str) + "\n")
            n += 1
            if n >= 100:
                break
print("Done:", n)
