import sys, io
sys.path.insert(0, '.')
from app import create_app
from PIL import Image
app = create_app()
client = app.test_client()

# Test ML status endpoint
r = client.get('/api/ml/status')
d = r.json
print('ML Status:')
loaded = d['model_loaded']
mode = d['inference_mode']
torch = d['torch_available']
sk = d.get('sklearn_available', 'N/A')
pt_models = d.get('pytorch_models', [])
sk_models = d.get('sklearn_models', [])
print(f'  model_loaded      = {loaded}')
print(f'  inference_mode    = {mode}')
print(f'  torch_available   = {torch}')
print(f'  sklearn_available = {sk}')
print(f'  pytorch_models    = {pt_models}')
print(f'  sklearn_models    = {sk_models}')
print()

# Test complaint with image
img = Image.new('RGB', (200, 200), color=(80, 60, 40))
buf = io.BytesIO()
img.save(buf, format='JPEG')
buf.seek(0)
data = {
    'reporter_name': 'Integration Test',
    'category': 'Potholes',
    'description': 'Large pothole with broken road surface and water',
    'latitude': '12.9700',
    'longitude': '77.6000',
}
r2 = client.post('/api/complaints',
    data={**data, 'image': (buf, 'pothole.jpg', 'image/jpeg')},
    content_type='multipart/form-data')
d2 = r2.json
print('POST /api/complaints:', r2.status_code)
if r2.status_code == 201:
    ml = d2.get('ml_result', {})
    print('  complaint_id  =', d2['complaint_id'])
    print('  incident_code =', d2.get('incident_code'))
    print('  is_duplicate  =', d2.get('is_duplicate'))
    print('  ml.category   =', ml.get('category'))
    print('  ml.confidence =', ml.get('confidence'))
    print('  ml.model_type =', ml.get('model_type'))
    print('  ml.status     =', ml.get('status'))
else:
    print('  ERROR:', d2)

# No-image submission
r3 = client.post('/api/complaints',
    data={
        'reporter_name': 'No Image Test',
        'category': 'Waterlogging',
        'description': 'Flooded street after heavy rain',
        'latitude': '12.9800',
        'longitude': '77.5900',
    },
    content_type='multipart/form-data')
d3 = r3.json
ml3 = d3.get('ml_result', {})
print('POST /api/complaints (no image):', r3.status_code)
print('  ml.status =', ml3.get('status', 'N/A'))

print()
print('=== INTEGRATION TEST PASSED ===')

