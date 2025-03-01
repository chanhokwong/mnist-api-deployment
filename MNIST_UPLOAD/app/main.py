import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torchvision.transforms as transforms
import io
import numpy as np


# --- 1. 複製模型定義 ---
# 在生產環境中，最好將模型定義單獨放在一個文件中導入
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc1(x)
        return x


# --- 2. 加載模型 ---
app = FastAPI(title="MNIST Digit Recognizer")

# 確保模型權重文件路徑正確
# 如果 main.py 在 app/ 文件夾內，權重文件在上一層，路徑是 ../mnist_cnn.pth
MODEL_PATH = "mnist_cnn.pth"
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))  # 在 CPU 上運行
model.eval()  # 設置為評估模式

# --- 3. 定義圖像轉換 ---
# 確保這個轉換與訓練時完全一致！
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# --- 4. 創建預測端點 ---
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    接收圖片文件，返回預測的數字。
    """
    # 讀取上傳的圖片
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # 預處理圖片
    image_tensor = transform(image).unsqueeze(0)  # 增加一個 batch 維度

    # 模型預測
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()  # 獲取預測的標籤
        confidence = probabilities.max().item()  # 獲取最大的概率值

    return {
        "predicted_digit": predicted_class,
        "confidence": f"{confidence:.4f}"
    }


@app.get("/")
def read_root():
    return {"message": "歡迎來到 MNIST 數字辨識 API。請訪問 /docs 查看 API 文檔。"}