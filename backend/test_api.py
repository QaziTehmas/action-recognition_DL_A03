import requests
from PIL import Image
import io

# Create a simple dummy image
img = Image.new('RGB', (224, 224), color = 'red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG')
img_byte_arr = img_byte_arr.getvalue()

url = 'http://localhost:5000/predict'
files = {'image': ('test.jpg', img_byte_arr, 'image/jpeg')}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response JSON:")
        print(response.json())
    else:
        print("Error Response:")
        print(response.text)
except Exception as e:
    print(f"Request failed: {e}")
