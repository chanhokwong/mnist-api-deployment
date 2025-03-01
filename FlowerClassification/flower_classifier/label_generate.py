import json

labels = {}
with open("labels.txt", "r") as f:
    for idx, line in enumerate(f):
        # 移除單引號和空白字符
        class_name = line.strip().strip("' ")
        labels[str(idx)] = class_name

with open("flower_labels.json", "w") as f:
    json.dump(labels, f, indent=4)