"""
Phase 4: Priority Scoring and Complaint Status Verification
Run from project root: python backend/test_phase4.py
Inspection-only run first; no code changes.
"""
import sys, os, io, json

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.database import db, Incident, Complaint, DuplicateMatch, PriorityAnalysis, generate_complaint_id
from app.utils.priority_engine import (
    calculate_priority_score, classify_priority, compute_time_history_score,
    compute_location_importance, analyze_incident_priority, WEIGHTS,
    CATEGORY_SEVERITY, CATEGORY_IMPACT,
)
from PIL import Image
import numpy as np

PASS = "PASS"
FAIL = "FAIL"
NT   = "NOT TESTED"
results = {}
TEST_RECORDS = {'complaint_ids': [], 'incident_ids': []}
app = create_app()

# ─── Helper ─────────────────────────────────────────────────────────────────
def solid_jpg(color=(90, 140, 60)):
    arr = __import__('numpy').full((64, 64, 3), color, dtype=__import__('numpy').uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, 'RGB').save(buf, format='JPEG')
    buf.seek(0)
    return buf.read()

SEP = "=" * 70

# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print("PART 1  — Formula Verification  P = 0.35S + 0.30I + 0.20L + 0.15T")
print(SEP)

# P1-1: Weights sum to 1.0
w_sum = sum(WEIGHTS.values())
ok = abs(w_sum - 1.0) < 1e-9
results['P1_1_weights_sum_to_1'] = PASS if ok else FAIL
print(f"  P1-1  Weights sum = {w_sum}  [{results['P1_1_weights_sum_to_1']}]")

# P1-2: Manual formula check
S, I, L, T = 8.5, 9.0, 8.5, 7.0
expected = round(S*0.35 + I*0.30 + L*0.20 + T*0.15, 2)
computed = calculate_priority_score(S, I, L, T)
ok = abs(computed - expected) < 0.01
results['P1_2_formula_matches_spec'] = PASS if ok else FAIL
print(f"  P1-2  S={S} I={I} L={L} T={T} → expected {expected}, got {computed}  [{results['P1_2_formula_matches_spec']}]")

# P1-3: Score capped at 10 max
computed_max = calculate_priority_score(10, 10, 10, 10)
ok = computed_max == 10.0
results['P1_3_score_capped_at_10'] = PASS if ok else FAIL
print(f"  P1-3  All 10s → score={computed_max}  [{results['P1_3_score_capped_at_10']}]")

# P1-4: Score floored at 0 min
computed_min = calculate_priority_score(0, 0, 0, 0)
ok = computed_min == 0.0
results['P1_4_score_floored_at_0'] = PASS if ok else FAIL
print(f"  P1-4  All 0s → score={computed_min}  [{results['P1_4_score_floored_at_0']}]")

# P1-5: Each category's default severity/impact constants
print("  P1-5  Category constants:")
for cat, sev in CATEGORY_SEVERITY.items():
    imp = CATEGORY_IMPACT[cat]
    print(f"        {cat}: S={sev} I={imp}")
ok = CATEGORY_SEVERITY['Waterlogging'] == 8.5 and CATEGORY_IMPACT['Waterlogging'] == 9.0
results['P1_5_category_constants_correct'] = PASS if ok else FAIL
print(f"  P1-5  Waterlogging S=8.5 I=9.0  [{results['P1_5_category_constants_correct']}]")

print()
print(SEP)
print("PART 2  — Priority Category Classification")
print(SEP)

boundaries = [
    (10.0, 'Critical'),
    (7.5,  'Critical'),
    (7.49, 'Urgent'),
    (4.0,  'Urgent'),
    (3.99, 'Normal'),
    (0.0,  'Normal'),
]
all_ok = True
for score, expected_cat in boundaries:
    got = classify_priority(score)
    ok  = got == expected_cat
    all_ok = all_ok and ok
    print(f"  P2  score={score:.2f} → expected '{expected_cat}', got '{got}'  [{'PASS' if ok else 'FAIL'}]")
results['P2_category_boundaries'] = PASS if all_ok else FAIL

print()
print(SEP)
print("PART 3  — Time/History Score & Location Importance")
print(SEP)

# T sub-tests
time_cases = [
    (1,  0,  False, 0.9,  "1 complaint, fresh"),
    (5,  0,  False, 4.5,  "5 complaints, fresh"),
    (7,  0,  False, 6.0,  "7 complaints, capped base*0.6=6.0"),
    (1,  0,  True,  2.0,  "resolved → 2.0"),
    (3,  30, False, None, "3 complaints, 30-day-old"),
]
t_ok = True
print("  Time/History tests:")
for count, days, resolved, expected, label in time_cases:
    got = compute_time_history_score(count, days, resolved)
    if expected is not None:
        ok = abs(got - expected) < 0.01
        t_ok = t_ok and ok
        print(f"    {label}: got={got}  expected={expected}  [{'PASS' if ok else 'FAIL'}]")
    else:
        # Just check it's in range [0,10]
        ok = 0.0 <= got <= 10.0
        t_ok = t_ok and ok
        print(f"    {label}: got={got}  [0-10 range: {'PASS' if ok else 'FAIL'}]")
results['P3_time_history_score'] = PASS if t_ok else FAIL

# Location importance
loc_cases = [
    (12.97, 77.59, 'MG Road, Bangalore', 8.5,  "MG Road → high"),
    (12.97, 77.59, 'Indiranagar nagar',  6.0,  "nagar keyword → medium"),
    (12.97, 77.59, None,                 5.0,  "no address → 5.0"),
    (12.97, 77.59, 'Unknown area',       5.0,  "generic → 5.0"),
    (12.97, 77.59, 'Silk Board Junction, Bangalore', 8.5, "junction → high"),
]
loc_ok = True
print("  Location importance tests:")
for lat, lon, addr, expected, label in loc_cases:
    got = compute_location_importance(lat, lon, addr)
    ok  = abs(got - expected) < 0.01
    loc_ok = loc_ok and ok
    print(f"    {label}: got={got}  expected={expected}  [{'PASS' if ok else 'FAIL'}]")
results['P3_location_importance'] = PASS if loc_ok else FAIL

print()
print(SEP)
print("PART 4  — Database: Stored Priority Scores vs Re-computed")
print(SEP)

discrepancies = []
with app.app_context():
    incidents = Incident.query.all()
    print(f"  Checking {len(incidents)} incidents...")
    for inc in incidents:
        priority = analyze_incident_priority(inc)
        recomputed = priority['priority_score']
        stored     = inc.priority_score
        stored_cat = inc.priority_category
        recomp_cat = priority['priority_category']

        # Allow ±0.5 tolerance (time_history changes with real-time age)
        score_ok = stored is not None and abs(stored - recomputed) <= 0.5
        cat_ok   = stored_cat == recomp_cat

        flag = "OK" if (score_ok and cat_ok) else "DRIFT"
        if not score_ok or not cat_ok:
            discrepancies.append({
                'code': inc.incident_code, 'stored': stored,
                'recomputed': recomputed, 'stored_cat': stored_cat,
                'recomp_cat': recomp_cat,
            })
        print(f"    {inc.incident_code}: stored={stored} recomputed={recomputed:.2f}  "
              f"cat stored={stored_cat} recomputed={recomp_cat}  [{flag}]")

results['P4_stored_scores_match_formula'] = PASS if len(discrepancies) == 0 else FAIL
if discrepancies:
    print(f"  DISCREPANCIES ({len(discrepancies)}):")
    for d in discrepancies:
        print(f"    {d}")
else:
    print("  All incidents within ±0.5 tolerance.")

# Check priority_analyses table population
with app.app_context():
    pa_count  = PriorityAnalysis.query.count()
    inc_count = Incident.query.count()
    pa_inc_ids = {pa.incident_id for pa in PriorityAnalysis.query.all()}
    inc_ids    = {i.id for i in Incident.query.all()}
    missing_pa = inc_ids - pa_inc_ids
print(f"  PriorityAnalysis rows: {pa_count}  (incidents: {inc_count})")
print(f"  Incidents with NO PriorityAnalysis row: {len(missing_pa)}")
results['P4_priority_analyses_populated'] = PASS if len(missing_pa) == 0 else FAIL

print()
print(SEP)
print("PART 5  — API: Admin incidents endpoint returns all priority fields")
print(SEP)

REQUIRED_PRIORITY_FIELDS = [
    'priority_score', 'priority_category', 'priority_explanation',
    'severity', 'public_impact', 'location_importance', 'time_history_score',
]
with app.test_client() as client:
    r = client.get('/api/admin/incidents', query_string={'per_page': 20})
    assert r.status_code == 200, f"Failed: {r.status_code}"
    data = r.get_json()
    incidents_api = data.get('incidents', [])
    all_fields_ok = True
    missing_fields_found = []
    for inc in incidents_api:
        for f in REQUIRED_PRIORITY_FIELDS:
            if f not in inc or inc[f] is None:
                missing_fields_found.append(f"{inc.get('incident_code','?')}.{f}")
                all_fields_ok = False
    results['P5_api_returns_priority_fields'] = PASS if all_fields_ok else FAIL
    print(f"  Checked {len(incidents_api)} incidents from API.")
    if missing_fields_found:
        print(f"  Missing/null fields: {missing_fields_found}")
    else:
        print("  All required priority fields present in API response.")

    # Check 'is_ml' label not misused in priority_explanation
    ml_mislabel_found = []
    for inc in incidents_api:
        for line in inc.get('priority_explanation', []):
            if 'machine learning' in str(line).lower() or 'ml model' in str(line).lower():
                ml_mislabel_found.append(inc.get('incident_code'))
    results['P5_priority_not_mislabeled_as_ml'] = PASS if not ml_mislabel_found else FAIL
    if ml_mislabel_found:
        print(f"  ML mislabel found in explanations: {ml_mislabel_found}")
    else:
        print("  Priority correctly NOT described as ML in explanations.")

print()
print(SEP)
print("PART 6  — Complaint Status Update: DB persistence")
print(SEP)

VALID_STATUSES = ['Submitted', 'AI Analysed', 'Under Review', 'Assigned', 'In Progress', 'Resolved']
img_bytes = solid_jpg()

with app.test_client() as client:
    # Submit a test complaint to get a known complaint ID
    r_new = client.post('/api/complaints', data={
        'reporter_name': 'Phase4 Tester',
        'reporter_email': 'phase4@smcitest.internal',
        'category': 'Street Lights',
        'description': 'Phase 4 status test complaint',
        'latitude': '12.7700',
        'longitude': '77.3900',
        'location_address': 'Phase4 Test Zone, Bangalore',
        'image': (io.BytesIO(img_bytes), 'streetlight.jpg'),
    }, content_type='multipart/form-data')
    assert r_new.status_code == 201, f"Complaint creation failed: {r_new.status_code}"
    d_new = r_new.get_json()
    test_cid  = d_new['complaint_id']
    test_uuid = d_new['complaint_uuid']
    test_inc  = d_new['incident_code']
    TEST_RECORDS['complaint_ids'].append(test_cid)
    print(f"  Test complaint: {test_cid}  incident: {test_inc}")

    with app.app_context():
        c_obj = Complaint.query.filter_by(complaint_id=test_cid).first()
        inc_obj = Incident.query.filter_by(incident_code=test_inc).first()
        TEST_RECORDS['incident_ids'].append(inc_obj.id)

    # Test all status transitions
    status_ok = True
    prev = 'Submitted'
    for new_status in ['AI Analysed', 'Under Review', 'Assigned', 'In Progress', 'Resolved']:
        r_upd = client.patch(
            f'/api/complaints/{test_uuid}/status',
            json={'status': new_status},
            content_type='application/json'
        )
        ok_api = r_upd.status_code == 200 and r_upd.get_json().get('status') == new_status
        # Verify DB persistence
        with app.app_context():
            db_status = Complaint.query.filter_by(complaint_id=test_cid).first().status
        ok_db = db_status == new_status
        ok = ok_api and ok_db
        status_ok = status_ok and ok
        print(f"  {prev} → {new_status}: api={'OK' if ok_api else 'FAIL'} db={'OK' if ok_db else 'FAIL'}  [{'PASS' if ok else 'FAIL'}]")
        prev = new_status
    results['P6_complaint_status_transitions'] = PASS if status_ok else FAIL

    # Test invalid status rejected
    r_bad = client.patch(
        f'/api/complaints/{test_uuid}/status',
        json={'status': 'FakeStatus'},
        content_type='application/json'
    )
    ok_reject = r_bad.status_code == 400
    results['P6_invalid_status_rejected'] = PASS if ok_reject else FAIL
    print(f"  Invalid status 'FakeStatus' → HTTP {r_bad.status_code}  [{results['P6_invalid_status_rejected']}]")

print()
print(SEP)
print("PART 7  — Incident Status Update (Admin route) & Priority Consistency")
print(SEP)

with app.test_client() as client:
    with app.app_context():
        inc_obj = Incident.query.filter_by(incident_code=test_inc).first()
        inc_uuid = inc_obj.id
        priority_before = inc_obj.priority_score
        priority_cat_before = inc_obj.priority_category

    # Update incident status to Resolved
    r_inc_upd = client.patch(
        f'/api/admin/incidents/{inc_uuid}',
        json={'status': 'Resolved'},
        content_type='application/json'
    )
    ok_api = r_inc_upd.status_code == 200
    returned = r_inc_upd.get_json().get('incident', {})
    print(f"  Incident PATCH status=Resolved: HTTP {r_inc_upd.status_code}  [{'PASS' if ok_api else 'FAIL'}]")

    with app.app_context():
        inc_check = db.session.get(Incident, inc_uuid)
        db_status_ok = inc_check.status == 'Resolved'
        resolved_at_ok = inc_check.resolved_at is not None
        # Priority fields must still be present (not wiped)
        priority_still_set = inc_check.priority_score is not None and inc_check.priority_category is not None

    results['P7_incident_status_persisted'] = PASS if (ok_api and db_status_ok) else FAIL
    results['P7_resolved_at_set'] = PASS if resolved_at_ok else FAIL
    results['P7_priority_fields_intact_after_status_change'] = PASS if priority_still_set else FAIL
    print(f"  DB status=Resolved: [{results['P7_incident_status_persisted']}]")
    print(f"  resolved_at set:    [{results['P7_resolved_at_set']}]")
    print(f"  Priority fields intact after status change: [{results['P7_priority_fields_intact_after_status_change']}]")

    # Check the returned incident has priority data
    has_priority_in_response = all(
        f in returned for f in ['priority_score', 'priority_category', 'severity', 'public_impact']
    )
    results['P7_patch_response_has_priority'] = PASS if has_priority_in_response else FAIL
    print(f"  PATCH response includes priority fields: [{results['P7_patch_response_has_priority']}]")

    # Bug check: does status change recalculate priority?
    # When status changes to Resolved, time_history_score should → 2.0
    # but the admin_update_incident route does NOT call analyze_incident_priority
    with app.app_context():
        inc_check2 = db.session.get(Incident, inc_uuid)
        stored_time_history = inc_check2.time_history_score
    # Re-run analysis to see what it SHOULD be now
    with app.app_context():
        inc_check3 = db.session.get(Incident, inc_uuid)
        reanalysed = analyze_incident_priority(inc_check3)
        expected_time_history = reanalysed['time_history']  # should be 2.0 (resolved)
    priority_recalc_on_status = abs(stored_time_history - expected_time_history) < 0.01
    results['P7_priority_recalculated_on_status_change'] = PASS if priority_recalc_on_status else FAIL
    print(f"  Priority recalculated on status change:")
    print(f"    stored time_history={stored_time_history}, expected after Resolved={expected_time_history}")
    print(f"    [{results['P7_priority_recalculated_on_status_change']}]")

print()
print(SEP)
print("PART 8  — Priority NOT Mislabeled as Machine Learning")
print(SEP)

# Check source file docstring
with open('backend/app/utils/priority_engine.py', 'r') as f:
    pe_src = f.read()
has_not_ml_note = 'NOT a trained ML model' in pe_src or 'rule-based' in pe_src.lower()
results['P8_engine_clearly_not_ml'] = PASS if has_not_ml_note else FAIL
print(f"  priority_engine.py clearly states NOT ML: [{results['P8_engine_clearly_not_ml']}]")

# Check no ML-language in frontend priority breakdown
# AdminIncidents.jsx references 'Priority Breakdown' — check for ML mentions
with open('frontend/src/components/admin/AdminIncidents.jsx', 'r') as f:
    admin_inc_src = f.read()
ui_mentions_ml = 'ml model' in admin_inc_src.lower() and 'priority' in admin_inc_src.lower()
results['P8_frontend_priority_not_labeled_as_ml'] = PASS if not ui_mentions_ml else FAIL
print(f"  AdminIncidents.jsx does not label priority as ML: [{results['P8_frontend_priority_not_labeled_as_ml']}]")

# ─── TEARDOWN ────────────────────────────────────────────────────────────────
print()
print(SEP)
print("TEARDOWN")
print(SEP)
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
    print(f"  Remaining: incidents={Incident.query.count()} "
          f"complaints={Complaint.query.count()} "
          f"duplicate_matches={DuplicateMatch.query.count()}")

# ─── FINAL REPORT ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("FINAL RESULTS")
print(SEP)
passed = failed = skipped = 0
for name, status in results.items():
    icon = '✅' if status == PASS else ('⚠️ ' if status == NT else '❌')
    print(f"  {icon} {name}: {status}")
    if status == PASS:   passed += 1
    elif status == NT:   skipped += 1
    else:                failed += 1

print()
print(f"  Total: {passed} PASSED  {failed} FAILED  {skipped} NOT TESTED")
sys.exit(0 if failed == 0 else 1)

