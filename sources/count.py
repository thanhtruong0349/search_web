import json

# Đọc file JSON
with open("../datasets/arXiv_papers.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# In số lượng object
print("leng:", len(data))
