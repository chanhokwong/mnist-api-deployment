import json

with open("flower_classifier/label_map.json", "r") as f:
    flower_labels = json.load(f)
    print(flower_labels)

# 格式如：{0: 'pink primrose', 1: 'hard-leaved pocket orchid', ...}
