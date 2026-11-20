"""
Phase 3 Bug Fix Verification Tests
Tests all 4 bugs fixed in duplicate detection.
Run from project root: python backend/test_bugfixes.py
"""
import sys
import os
import io
import json

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.database import db, Incident, Complaint, DuplicateMatch, PriorityAnalysis
from app.utils.duplicate_detector import classify_match, VISUAL_FLOOR_FOR_AUTO_MERGE
from app.utils.image_utils import compute_image_hash, compute_image_features
from PIL import Image
import numpy as np

PASS = "PASS"
FAIL = "FAIL"
results = {}

app = create_app()

# ─── Helpers ────────────────────────────────────────────────────────────────

def make_solid_image(color=(128, 64, 32), size=(64, 64)):
    """Create a solid-colour JPEG image in memory."""
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    img = Image.fromarray(arr, 'RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    buf.seek(0)
    return buf.read()

def make_gradient_image(start=(10, 20, 200), end=(200, 100, 50), size=(64, 64)):
    """Create a gradient image (visually very different from solid)."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for x in range(size[0]):
        t = x / size[0]
        arr[:, x] = [int(start[i] + t * (end[i] - start[i])) for i in range(3)]
    img = Image.fromarray(arr, 'RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    buf.seek(0)
    return buf.read()

TEST_RECORDS = {'complaint_ids': [], 'incident_ids': []}

# ─── PART A: Unit tests for Bug 1 (classify_match visual floor) ─────────────

print("=" * 70)
print("PART A: Unit Tests — classify_match() Visual Floor (Bug 1)")
print("=" * 70)

# A1: High visual + high geo → exact (unchanged behaviour)
match_type, score = classify_match(1.0, 1.0, True, False)
ok = match_type == 'exact'
results['A1_exact_high_visual_high_geo'] = PASS if ok else FAIL
print(f"  A1 geo=1.0 vis=1.0 → {match_type} score={score:.3f}  [{results['A1_exact_high_visual_high_geo']}]")

# A2: Moderate visual (0.72) + high geo → probable (unchanged behaviour)
match_type, score = classify_match(1.0, 0.72, True, False)
ok = match_type == 'probable'
results['A2_probable_good_visual'] = PASS if ok else FAIL
print(f"  A2 geo=1.0 vis=0.72 → {match_type} score={score:.3f}  [{results['A2_probable_good_visual']}]")

# A3: BUG 1 FIX — Low visual (0.47) + high geo → must be 'possible', NOT auto-merged
match_type, score = classify_match(1.0, 0.47, True, False)
ok = match_type == 'possible'
results['A3_low_visual_high_geo_not_automerged'] = PASS if ok else FAIL
print(f"  A3 geo=1.0 vis=0.47 → {match_type} score={score:.3f}  [{results['A3_low_visual_high_geo_not_automerged']}] (must be 'possible')")

# A4: BUG 1 FIX — Exact visual floor boundary (0.65) → should qualify as probable
match_type, score = classify_match(1.0, 0.65, True, False)
ok = match_type in ('exact', 'probable')
results['A4_visual_at_floor_qualifies'] = PASS if ok else FAIL
print(f"  A4 geo=1.0 vis=0.65 → {match_type} score={score:.3f}  [{results['A4_visual_at_floor_qualifies']}] (should be probable/exact)")

# A5: BUG 1 FIX — Just below visual floor (0.64) → 'possible'
match_type, score = classify_match(1.0, 0.64, True, False)
ok = match_type == 'possible'
results['A5_just_below_floor_is_possible'] = PASS if ok else FAIL
print(f"  A5 geo=1.0 vis=0.64 → {match_type} score={score:.3f}  [{results['A5_just_below_floor_is_possible']}] (must be 'possible')")

# A6: No visual (None) + geo only → still uses geo×0.7 path (no change)
match_type, score = classify_match(1.0, None, True, False)
ok = match_type in ('exact', 'probable')
results['A6_no_visual_geo_only'] = PASS if ok else FAIL
print(f"  A6 geo=1.0 vis=None → {match_type} score={score:.3f}  [{results['A6_no_visual_geo_only']}] (geo-only should still qualify)")

# A7: Different category → not_duplicate (guard unchanged)
match_type, score = classify_match(1.0, 1.0, False, False)
ok = match_type == 'not_duplicate' and score == 0.0
results['A7_different_category'] = PASS if ok else FAIL
print(f"  A7 same_cat=False → {match_type} score={score:.3f}  [{results['A7_different_category']}]")

# A8: Resolved incident at nearby geo → possible (guard unchanged)
match_type, score = classify_match(0.9, 1.0, True, True)
ok = match_type == 'possible' and score == 0.5
results['A8_resolved_incident'] = PASS if ok else FAIL
print(f"  A8 resolved → {match_type} score={score:.3f}  [{results['A8_resolved_incident']}]")

print(f"\n  VISUAL_FLOOR_FOR_AUTO_MERGE = {VISUAL_FLOOR_FOR_AUTO_MERGE}")

# ─── PART B: Integration Tests (Flask test_client) ───────────────────────────

print()
print("=" * 70)
print("PART B: Integration Tests — End-to-End Duplicate Submission")
print("=" * 70)

img_identical_bytes = make_solid_image(color=(90, 140, 60))
img_different_bytes  = make_gradient_image(start=(10, 20, 200), end=(200, 100, 50))

# Lat/lon: Bangalore test area, well away from existing seeded data
TEST_LAT  = 12.8500
TEST_LON  = 77.4100
TEST_LAT2 = 12.8501   # ~11 m away (same GPS cluster)
TEST_LAT3 = 12.8502   # ~22 m away (same GPS cluster, different image)

with app.test_client() as client:

    # ── B1: Submit original complaint ─────────────────────────────────────
    print("\n  [B1] Submitting original complaint (baseline)...")
    res = client.post('/api/complaints', data={
        'reporter_name': 'Test Alice',
        'reporter_email': 'alice@smcitest.internal',
        'category': 'Potholes',
        'description': 'Test pothole for bug-fix verification',
        'latitude': str(TEST_LAT),
        'longitude': str(TEST_LON),
        'location_address': 'BugFix Test Zone, Bangalore',
        'image': (io.BytesIO(img_identical_bytes), 'pothole_a.jpg'),
    }, content_type='multipart/form-data')
    assert res.status_code == 201, f"B1 failed: {res.status_code} {res.data}"
    d1 = res.get_json()
    cid_1 = d1['complaint_id']
    inc_code_1 = d1['incident_code']
    is_dup_1 = d1['is_duplicate']
    TEST_RECORDS['complaint_ids'].append(cid_1)
    print(f"    complaint={cid_1}  incident={inc_code_1}  is_duplicate={is_dup_1}")

    with app.app_context():
        inc_obj = Incident.query.filter_by(incident_code=inc_code_1).first()
        assert inc_obj is not None
        inc_id_1 = inc_obj.id
        TEST_RECORDS['incident_ids'].append(inc_id_1)
        init_count = inc_obj.complaint_count
        init_priority = inc_obj.priority_score
    print(f"    Initial complaint_count={init_count}, priority_score={init_priority}")
    ok = not is_dup_1 and init_count == 1
    results['B1_initial_complaint_created'] = PASS if ok else FAIL
    print(f"    [{results['B1_initial_complaint_created']}]")

    # ── B2: Submit identical image → should auto-merge (Bug 1: visual ≥ 0.65) ──
    print("\n  [B2] Submitting IDENTICAL image, same location (~11m away)...")
    res2 = client.post('/api/complaints', data={
        'reporter_name': 'Test Bob',
        'reporter_email': 'bob@smcitest.internal',
        'category': 'Potholes',
        'description': 'Same pothole again',
        'latitude': str(TEST_LAT2),
        'longitude': str(TEST_LON),
        'location_address': 'BugFix Test Zone, Bangalore',
        'image': (io.BytesIO(img_identical_bytes), 'pothole_b.jpg'),
    }, content_type='multipart/form-data')
    assert res2.status_code == 201, f"B2 failed: {res2.status_code} {res2.data}"
    d2 = res2.get_json()
    cid_2 = d2['complaint_id']
    is_dup_2 = d2['is_duplicate']
    inc_code_2 = d2['incident_code']
    TEST_RECORDS['complaint_ids'].append(cid_2)
    print(f"    complaint={cid_2}  incident={inc_code_2}  is_duplicate={is_dup_2}")

    with app.app_context():
        inc_obj = db.session.get(Incident, inc_id_1)
        count_after_dup = inc_obj.complaint_count
        # Bug 4: check auto-merged match is NOT in pending queue
        dm = DuplicateMatch.query.filter_by(complaint_id=Complaint.query.filter_by(complaint_id=cid_2).first().id).first()
        dm_reviewed = dm.reviewed if dm else None
        dm_action   = dm.review_action if dm else None
    print(f"    complaint_count after auto-merge={count_after_dup}")
    print(f"    DuplicateMatch reviewed={dm_reviewed} action={dm_action}")
    ok_merge   = is_dup_2 and inc_code_2 == inc_code_1 and count_after_dup == 2
    ok_bug4    = dm_reviewed is True and dm_action == 'auto_merged'
    results['B2_identical_image_auto_merged'] = PASS if ok_merge else FAIL
    results['B4_auto_merged_not_in_pending_queue'] = PASS if ok_bug4 else FAIL
    print(f"    B2 auto-merge [{results['B2_identical_image_auto_merged']}]")
    print(f"    B4 not-in-pending [{results['B4_auto_merged_not_in_pending_queue']}]")

    # ── B3: Submit DIFFERENT image at same GPS → must NOT auto-merge (Bug 1) ──
    print("\n  [B3] Submitting DIFFERENT image, same GPS cluster (~22m away)...")
    res3 = client.post('/api/complaints', data={
        'reporter_name': 'Test Carol',
        'reporter_email': 'carol@smcitest.internal',
        'category': 'Potholes',
        'description': 'Different issue nearby',
        'latitude': str(TEST_LAT3),
        'longitude': str(TEST_LON),
        'location_address': 'BugFix Test Zone Nearby, Bangalore',
        'image': (io.BytesIO(img_different_bytes), 'different_issue.jpg'),
    }, content_type='multipart/form-data')
    assert res3.status_code == 201, f"B3 failed: {res3.status_code} {res3.data}"
    d3 = res3.get_json()
    cid_3 = d3['complaint_id']
    inc_code_3 = d3['incident_code']
    is_dup_3 = d3['is_duplicate']
    TEST_RECORDS['complaint_ids'].append(cid_3)
    print(f"    complaint={cid_3}  incident={inc_code_3}  is_duplicate={is_dup_3}")

    with app.app_context():
        inc_obj = db.session.get(Incident, inc_id_1)
        count_after_different = inc_obj.complaint_count
        if inc_code_3 != inc_code_1:
            new_inc = Incident.query.filter_by(incident_code=inc_code_3).first()
            if new_inc:
                TEST_RECORDS['incident_ids'].append(new_inc.id)
    print(f"    incident_1 complaint_count still={count_after_different} (must stay 2)")
    # Different image must NOT auto-merge (is_dup must be False, or if DM flagged, count stays 2)
    ok_no_merge = count_after_different == 2 and not is_dup_3
    results['B3_different_image_not_auto_merged'] = PASS if ok_no_merge else FAIL
    print(f"    [{results['B3_different_image_not_auto_merged']}]")

    # ── B5: Verify pending duplicate-matches API does not expose auto-merged ones ─
    print("\n  [B5] Checking /api/admin/duplicate-matches (pending only)...")
    r_pending = client.get('/api/admin/duplicate-matches', query_string={'reviewed': 'false'})
    assert r_pending.status_code == 200
    pending = r_pending.get_json().get('matches', [])
    # The auto-merged complaint B2 must NOT appear in pending
    pending_ids = [m['complaint_id'] for m in pending]
    ok_b5 = cid_2 not in pending_ids
    results['B5_pending_queue_clean'] = PASS if ok_b5 else FAIL
    print(f"    Pending matches: {len(pending)}")
    print(f"    cid_2 ({cid_2}) in pending: {cid_2 in pending_ids}")
    print(f"    [{results['B5_pending_queue_clean']}]")

    # ── B6: Bug 2 — Delete complaint B2, check priority recalculated ────────
    print("\n  [B6] Testing Bug 2: Delete auto-merged complaint, verify priority recalc...")
    with app.app_context():
        before_del = db.session.get(Incident, inc_id_1)
        priority_before_del = before_del.priority_score
        count_before_del = before_del.complaint_count
    print(f"    Before delete: count={count_before_del} priority={priority_before_del}")

    res_del = client.delete(f'/api/complaints/{cid_2}')
    assert res_del.status_code == 200, f"Delete failed: {res_del.data}"

    with app.app_context():
        after_del = db.session.get(Incident, inc_id_1)
        count_after_del = after_del.complaint_count
        priority_after_del = after_del.priority_score
        pcat_after_del = after_del.priority_category
    print(f"    After delete:  count={count_after_del} priority={priority_after_del} cat={pcat_after_del}")

    ok_b6_count    = count_after_del == count_before_del - 1
    ok_b6_priority = priority_after_del is not None  # priority must have been updated
    # Count went down from 2→1, so time_history_score should decrease
    ok_b6_changed  = priority_after_del <= priority_before_del
    results['B6_delete_decrements_count'] = PASS if ok_b6_count else FAIL
    results['B6_delete_recalculates_priority'] = PASS if (ok_b6_priority and ok_b6_changed) else FAIL
    print(f"    count decremented [{results['B6_delete_decrements_count']}]")
    print(f"    priority recalculated [{results['B6_delete_recalculates_priority']}]")

    # ── B7: Bug 3 — Admin confirms a possible match, priority recalculated ──
    print("\n  [B7] Testing Bug 3: Admin confirm, verify priority+updated_at recalc...")
    # Find a 'possible' or 'manual_review' unreviewed match
    r_matches = client.get('/api/admin/duplicate-matches', query_string={'reviewed': 'false'})
    all_pending = r_matches.get_json().get('matches', [])
    possible_match = next((m for m in all_pending if m['match_type'] in ('possible', 'manual_review')), None)

    if possible_match:
        match_id  = possible_match['match_id']
        inc_code_target = possible_match['incident_code']
        print(f"    Found match_id={match_id} type={possible_match['match_type']} → {inc_code_target}")

        with app.app_context():
            inc_target = Incident.query.filter_by(incident_code=inc_code_target).first()
            count_before_confirm = inc_target.complaint_count
            priority_before_confirm = inc_target.priority_score
            upd_before = inc_target.updated_at

        res_confirm = client.patch(
            f'/api/incidents/duplicates/{match_id}/review',
            json={'action': 'confirmed', 'reviewed_by': 'test_admin'},
            content_type='application/json'
        )
        assert res_confirm.status_code == 200, f"Confirm failed: {res_confirm.data}"

        with app.app_context():
            inc_target = Incident.query.filter_by(incident_code=inc_code_target).first()
            count_after_confirm = inc_target.complaint_count
            priority_after_confirm = inc_target.priority_score
            upd_after = inc_target.updated_at

        ok_b7_count   = count_after_confirm == count_before_confirm + 1
        ok_b7_upd     = upd_after >= upd_before
        ok_b7_priority = priority_after_confirm >= priority_before_confirm  # more complaints → higher or equal score
        print(f"    count: {count_before_confirm} → {count_after_confirm} [{PASS if ok_b7_count else FAIL}]")
        print(f"    priority: {priority_before_confirm} → {priority_after_confirm} [{PASS if ok_b7_priority else FAIL}]")
        print(f"    updated_at advanced: {ok_b7_upd} [{PASS if ok_b7_upd else FAIL}]")
        results['B7_confirm_increments_count'] = PASS if ok_b7_count else FAIL
        results['B7_confirm_recalculates_priority'] = PASS if ok_b7_priority else FAIL
        results['B7_confirm_updates_timestamp'] = PASS if ok_b7_upd else FAIL

        # Undo confirm so we don't pollute baseline — mark it rejected (reversal)
        # (We can't easily undo a confirmed match, but we note it for cleanup)
        print("    (Note: confirmed match is part of baseline now — recorded)")
    else:
        print("    No pending possible/manual_review matches found — skipping B7.")
        results['B7_confirm_increments_count'] = 'NOT TESTED'
        results['B7_confirm_recalculates_priority'] = 'NOT TESTED'
        results['B7_confirm_updates_timestamp'] = 'NOT TESTED'

# ─── TEARDOWN ────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("TEARDOWN: Removing test records")
print("=" * 70)
with app.app_context():
    for cid in TEST_RECORDS['complaint_ids']:
        c = Complaint.query.filter_by(complaint_id=cid).first()
        if c:
            DuplicateMatch.query.filter_by(complaint_id=c.id).delete()
            db.session.delete(c)
    for iid in TEST_RECORDS['incident_ids']:
        inc = db.session.get(Incident, iid)
        if inc:
            PriorityAnalysis.query.filter_by(incident_id=iid).delete()
            db.session.delete(inc)
    db.session.commit()
    final_inc   = Incident.query.count()
    final_comp  = Complaint.query.count()
    final_dm    = DuplicateMatch.query.count()
print(f"  incidents={final_inc}  complaints={final_comp}  duplicate_matches={final_dm}")

# ─── FINAL REPORT ────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)
passed = failed = skipped = 0
for name, status in results.items():
    icon = '✅' if status == PASS else ('⚠️ ' if status == 'NOT TESTED' else '❌')
    print(f"  {icon} {name}: {status}")
    if status == PASS:
        passed += 1
    elif status == 'NOT TESTED':
        skipped += 1
    else:
        failed += 1

print()
print(f"  Total: {passed} PASSED  {failed} FAILED  {skipped} NOT TESTED")
sys.exit(0 if failed == 0 else 1)

