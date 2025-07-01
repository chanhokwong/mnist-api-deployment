# 項目: 基於分類模型的docker快速部署

## 服務器配置
coreOS: Ubuntu 22.04
memory: 32GB

## 上載鏡像至docker hub
```
docker build -t mnist-api .

docker tag mnist-api chanhokwong/mnist-api:latest

docker push chanhokwong/mnist-api:latest

docker run -p 8000:8000 mnist-api
```

## 服務器部署
```
# 在安裝主系統時設置用戶 帳號:chanhokwong 密碼:12345678

# 使用ssh連接雲服務端
ssh chanhokwong@172.17.0.1  # chanhokwong為username 172.17.0.1是服務器IP 
12345678  # 輸入密碼

# Ubuntu系統

# 更新包列表
sudo apt-get update

# 安裝必要的工具
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

# 添加 Docker 的官方 GPG 密鑰
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 設置 Docker 的穩定版倉庫
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 再次更新包列表並安裝 Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# 把chanhokwong帳號設置為超級使用者權限 可讀取、寫入
sudo usermod -aG docker chanhokwong

# 更改權限後 需退出ssh 再重新連接 這樣權限才能更新

# 連接你在docker hub倉庫的帳號
docker login -u chanhokwong  # username為chanhokwong

# 輸入你docker hub對應帳號的密碼
12345678

# 從docker hub下載鏡像images
docker pull chanhokwong/mnist-api:latest

# 創建容器mnist-container並運行下載的images(chanhokwong/mnist-api:latest)
docker run -d -p 8000:8000 --name mnist-container chanhokwong/mnist-api:latest
```

對於部署時空間儲存不足的方案
```
# 查看卷組 (Volume Group) 信息
sudo vgdisplay

# 查看邏輯卷 (Logical Volume) 信息 可以理解為系統最大可使用的空間
sudo lvdisplay

# 擴展邏輯卷，使用所有剩餘空間 
# 邏輯:把卷組其餘多出的空間都給邏輯卷使用
sudo lvextend -l +100%FREE /dev/ubuntu-vg/ubuntu-lv

# 調整文件系統大小以匹配邏輯卷 讓系統知道你的可用空間變大了
sudo resize2fs /dev/mapper/ubuntu--vg-ubuntu--lv

# 查看整個系統的儲存空間分配
df -hT

# 清理docker中先前下載失敗的殘留文件
sudo docker system prune -a -f
```
