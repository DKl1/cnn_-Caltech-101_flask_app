import requests

resp = requests.post('http://127.0.0.1:5000/check', files={'file': open('test_data/watch.jpeg', 'rb')})

print(resp.text)