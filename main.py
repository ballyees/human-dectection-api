from sanic import Sanic
from sanic.response import json
from model import detector
app = Sanic(__name__)

@app.route('/')
async def test(request):
    return json({})

@app.route('/detect', methods=['POST'])
async def test(request):
    json_body = request.json
    # detector.run()
    return json({})

if __name__ == '__main__':
    app.run()