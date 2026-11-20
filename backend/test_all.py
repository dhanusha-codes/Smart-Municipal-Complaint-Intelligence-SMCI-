import sys, os, json, io
from PIL import Image
sys.path.insert(0, '.')
from app import create_app
app = create_app()
client = app.test_client()

print('=== COMPREHENSIVE API TEST ===')

r = client.get('/api/health')
print('GET /api/health:', r.status_code)

r = client.get('/api/dashboard/stats')
d = r.json
tc = d['total_complaints']
ti = d['total_incidents']
print(f'GET /api/dashboard/stats: {r.status_code} - {tc} complaints, {ti} incidents')

r = client.get('/api/incidents')
d = r.json
cnt = len(d['incidents'])
print(f'GET /api/incidents: {r.status_code} - {cnt} incidents')
if d['incidents']:
    inc = d['incidents'][0]
    pc = inc.get('priority_category')
    ps = inc.get('priority_score')
    print(f'  Top: {inc["incident_code"]} | {inc["category"]} | {pc} | {ps}')

r = client.get('/api/incidents/map')
d = r.json
print(f'GET /api/incidents/map: {r.status_code} - {len(d["incidents"])} points')

r = client.get('/api/admin/incidents')
d = r.json
print(f'GET /api/admin/incidents: {r.status_code} - total={d["total"]}')

r = client.get('/api/admin/complaints')
d = r.json
print(f'GET /api/admin/complaints: {r.status_code} - total={d["total"]}')

r = client.get('/api/complaints')
d = r.json
first_cid = d['complaints'][0]['complaint_id'] if d['complaints'] else None
print(f'GET /api/complaints: {r.status_code} - {len(d["complaints"])} complaints, first={first_cid}')

if first_cid:
    r = client.get('/api/complaints/' + first_cid)
    d = r.json
    print(f'GET /api/complaints/{first_cid}: {r.status_code} - status={d.get("status")}')

r = client.get('/api/admin/duplicate-matches')
d = r.json
print(f'GET /api/admin/duplicate-matches: {r.status_code} - {d.get("total", 0)} matches')

r = client.get('/api/ml/status')
d = r.json
print(f'GET /api/ml/status: {r.status_code} - model_loaded={d["model_loaded"]}')

# POST complaint with image
img = Image.new('RGB', (100, 100), color='blue')
buf = io.BytesIO()
img.save(buf, format='JPEG')
buf.seek(0)
data = {
    'reporter_name': 'Test User',
    'reporter_email': 'test@test.com',
    'category': 'Potholes',
    'description': 'Large pothole on main road causing accidents',
    'latitude': '12.9716',
    'longitude': '77.5946',
    'location_address': 'Test Road, Bangalore'
}
r = client.post('/api/complaints',
    data={**data, 'image': (buf, 'test.jpg', 'image/jpeg')},
    content_type='multipart/form-data')
d = r.json
print(f'POST /api/complaints: {r.status_code}')
if r.status_code == 201:
    cid = d['complaint_id']
    inc = d.get('incident_code')
    ml = d.get('ml_result', {}).get('category')
    is_dup = d.get('is_duplicate')
    print(f'  complaint_id={cid}, incident={inc}, is_duplicate={is_dup}, ml={ml}')
else:
    print(f'  ERROR: {d}')

print('=== ALL TESTS DONE ===')

