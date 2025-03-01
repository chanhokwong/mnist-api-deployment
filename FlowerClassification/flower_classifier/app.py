from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import json
import os

app = Flask(__name__)

# 加載標籤
with open("flower_labels.json", "r") as f:
    flower_labels = json.load(f)


# 定義預處理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize([224,224]),
        # transforms.RandomRotation(30),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


# 加載模型
# def load_model():
#     model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
#     model.fc = torch.nn.Linear(model.fc.in_features, 102)
#     model.load_state_dict(torch.load("flower_classifier.pth", map_location=torch.device('cpu')))
#     model.eval()
#     return model

def load_model():
    # 直接載入完整模型（包含架構和權重）
    # model = torch.load("model.pt",
    #                    map_location='cpu',
    #                    weights_only=False)  # 如果是完整模型，需要設為False
    model = torch.load("model.pt",
                       map_location='cpu')
    model.eval()
    return model

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    model = load_model()
    print("Model weights loaded successfully!")
    print("First layer weight:", model.fc[3].weight[0][:5])  # 打印全連接層部分權重

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # 保存上傳的圖片
    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    image_path = os.path.join(upload_dir, file.filename)
    file.save(image_path)

    # 預測
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
    predicted_idx = torch.argmax(output).item()
    predicted_name = flower_labels.get(str(predicted_idx), "Unknown")

    return jsonify({
        "prediction": predicted_name,
        "image_url": image_path
    })


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(debug=False)
    # app.run(debug=True, extra_files=["model.pt","flower_labels.json"])