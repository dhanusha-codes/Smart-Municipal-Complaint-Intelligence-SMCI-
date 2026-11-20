import sys, json, io, sqlite3
sys.path.insert(0, '.')
from app import create_app
from PIL import Image

app = create_app()

with app.test_client() as c:
    img = Image.new('RGB', (100,100), color=(50,40,30))

    for i in range(2):
        buf = io.BytesIO()
        img.save(buf, 'JPEG')
        buf.seek(0)
        r = c.post('/api/complaints', data={
            'reporter_name': 'DupTest',
            'category': 'Waterlogging',
            'description': 'waterlogging flood stagnant water near junction',
            'latitude': '12.9810',
            'longitude': '77.5520',
            'image': (buf, 'dup.jpg', 'image/jpeg'),
        }, content_type='multipart/form-data')
        d = json.loads(r.data)
        cid = d.get('complaint_id')
        inc = d.get('incident_code')
        dup = d.get('is_duplicate')
        mat = d.get('duplicate_matches')
        print(f"Submit {i+1}: complaint={cid} incident={inc} is_dup={dup} match={mat}")

conn = sqlite3.connect('instance/smci.db')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM duplicate_matches')
print('duplicate_matches rows total:', cur.fetchone()[0])
cur.execute('SELECT geo_distance_km, visual_similarity, combined_score, match_type FROM duplicate_matches ORDER BY rowid DESC LIMIT 3')
for row in cur.fetchall():
    print('  match:', row)
conn.close()

