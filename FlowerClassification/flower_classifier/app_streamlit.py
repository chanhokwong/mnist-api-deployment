# 創建文件 app_streamlit.py
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import json

# 加載模型和標籤（同上）
# 加載標籤
with open("label_map.json", "r") as f:
    flower_labels = json.load(f)


# 定義預處理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(224),
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
    model = torch.load("model.pt",
                      map_location=torch.device('cpu'),
                      weights_only=False)  # 如果是完整模型，需要設為False
    model.eval()
    return model

st.title("Flower Classifier")
uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    input_tensor = preprocess_image(image)  # 需修改preprocess_image直接接受PIL圖像
    with torch.no_grad():
        model = load_model()
        output = model(input_tensor)
    predicted_idx = torch.argmax(output).item()
    predicted_name = flower_labels.get(str(predicted_idx), "Unknown")

    st.success(f"Prediction: {predicted_name}")