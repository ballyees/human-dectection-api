from sanic import Sanic
from sanic.response import json
from sanic_cors import CORS
from model import detector
import base64
from io import BytesIO
from PIL import Image
import numpy as np

app = Sanic(__name__)
CORS(app)
def read_image_base64(image):
    image = base64.b64decode(image)
    image = Image.open(BytesIO(image)).convert('RGB')
    image = np.asarray(image, dtype=np.uint8)
    return image

@app.route('/')
async def test(request):
    return json({})

@app.post('/detect')
async def test(request):
    json_body = request.json
    gps = json_body.get('gps', None)
    thresh = json_body.get('thresh', 0.2)
    isRGB = json_body.get('rgb', True)
    image = read_image_base64(json_body['image'])
    count = detector.run_count(image, isRGB, thresh)
    count = int(count)
    return json({"count": count})

if __name__ == '__main__':
    app.run()