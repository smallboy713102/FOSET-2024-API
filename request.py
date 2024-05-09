import requests

input_text = input('Enter your transcript/query : --> ')

response = requests.post("http://localhost:8000/predict_disease", json={"text": input_text})

print(response.json())
