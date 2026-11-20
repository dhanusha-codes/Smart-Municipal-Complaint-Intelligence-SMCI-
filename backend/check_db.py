import sqlite3
conn = sqlite3.connect('instance/smci.db')
c = conn.cursor()

c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
print("=== Tables ===")
for row in c.fetchall():
    print(' -', row[0])

print()
print("=== Complaints (last 3) ===")
c.execute("SELECT complaint_id, category, status, image_path, latitude, longitude FROM complaints ORDER BY rowid DESC LIMIT 3")
for row in c.fetchall():
    img = str(row[3])[:12] + '...' if row[3] else 'None'
    print(f"  {row[0]} | {row[1]} | {row[2]} | img:{img} | {row[4]},{row[5]}")

print()
print("=== Incidents (last 3) ===")
c.execute("SELECT incident_code, category, priority_category, priority_score, complaint_count, status FROM incidents ORDER BY rowid DESC LIMIT 3")
for row in c.fetchall():
    print(f"  {row[0]} | {row[1]} | {row[2]} | score:{row[3]} | reports:{row[4]} | {row[5]}")

print()
print("=== Counts ===")
for t in ['users', 'complaints', 'incidents', 'duplicate_matches', 'priority_analyses']:
    c.execute(f"SELECT COUNT(*) FROM {t}")
    print(f"  {t}: {c.fetchone()[0]}")

conn.close()

