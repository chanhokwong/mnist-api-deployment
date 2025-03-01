import torch
from PIL import Image
from torchvision import transforms

# 定義預處理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize([224,224]),
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

# 加載模型和預處理（同 app.py）
model = load_model()
image = Image.open("static/uploads/fire-lily-plant-profile-4768477-hero-df73484ca54d4351baa431787d9ad9bf.jpg")
input_tensor = preprocess_image(image)

# 執行推理
with torch.no_grad():
    output = model(input_tensor)
    print("Output logits:", output[0][:5])  # 查看前5個類別的輸出值
    predicted_idx = torch.argmax(output).item()
    print("Predicted index:", predicted_idx)