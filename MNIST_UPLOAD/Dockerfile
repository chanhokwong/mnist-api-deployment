# 使用一個包含 Python 的輕量級基礎鏡像
FROM python:3.10-slim

# 設置工作目錄
WORKDIR /MNIST_UPLOAD

# 複製依賴文件並安裝
# 這裡將模型文件和 requirements.txt 一起複製，以利用層緩存
COPY ./requirements.txt /MNIST_UPLOAD/requirements.txt
COPY ./mnist_cnn.pth /MNIST_UPLOAD/mnist_cnn.pth

# 安裝依賴
RUN pip install --no-cache-dir --upgrade -r /MNIST_UPLOAD/requirements.txt
# RUN pip install --no-cache-dir -r /MNIST_UPLOAD/requirements.txt
#RUN pip install --no-cache-dir --no-deps -r /MNIST_UPLOAD/requirements.txt \
#    && pip install --no-cache-dir -r /MNIST_UPLOAD/requirements.txt \
# RUN pip install --no-cache-dir --no-hashes -r /MNIST_UPLOAD/requirements.txt
#RUN python -m pip install --upgrade pip && \
#    pip install --isolated --no-cache-dir -r /MNIST_UPLOAD/requirements.txt

# 複製應用代碼
COPY ./app /MNIST_UPLOAD/app

# 暴露端口
EXPOSE 8000

# 啟動命令
# 告訴 uvicorn 去 /code/app 文件夾裡找 main.py 裡的 app 對象
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]