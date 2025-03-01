import requests

url = 'http://127.0.0.1:8000/predict/'

headers = {
    'accept': 'application/json',
    # requests won't add a boundary if this header is set when you pass files=
    # 'Content-Type': 'multipart/form-data',
}

files = {
    'file': ('number_9.jpg', open('number_9.jpg', 'rb'), 'image/jpeg'),
}

response = requests.post(url, headers=headers, files=files)

print(response.status_code)
print(response.json())