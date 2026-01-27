import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

request = {
    "url": "https://raw.githubusercontent.com/Uthmanee/machine-learning-zoomcamp-capstone2/master/testImage.jpeg"
}

result = requests.post(url, json=request).json()
print(result)

# You can also run the command below directly in the terminal instead of running test.py
# curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -H "Content-Type: application/json" -d '{"url": "https://raw.githubusercontent.com/Uthmanee/machine-learning-zoomcamp-capstone2/master/testImage.jpeg"}'