import sys
sys.path.insert(0, '.')
from app import create_app
import json, io
from PIL import Image

app = create_app()

test_cases = [
    ('pothole on main road',          (60,  50,  40)),
    ('garbage dump waste litter',     (100, 90,  70)),
    ('waterlogging flood stagnant',   (40,  60, 120)),
    ('drainage blocked sewer',        (30,  30,  30)),
    ('streetlight lamp dark pole',    (200,200, 180)),
    ('damaged road surface crack',    (80,  70,  60)),
]

with app.test_client() as c:
    for desc, color in test_cases:
        img = Image.new('RGB', (100,100), color=color)
        buf = io.BytesIO()
        img.save(buf, 'JPEG')
        buf.seek(0)
        r = c.post('/api/ml/classify',
            data={'image': (buf,'t.jpg','image/jpeg'), 'description': desc},
            content_type='multipart/form-data')
        d = json.loads(r.data)['result']
        print(f"{desc[:35]:35} => {d['category']:25} conf={d['confidence']}")

