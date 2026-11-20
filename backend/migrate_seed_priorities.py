"""
Phase 4 Migration: Fix seeded incident priorities (Bug A) and
create missing PriorityAnalysis records (Bug B).

Run once from project root: python backend/migrate_seed_priorities.py
Safe to re-run - will not create duplicate PriorityAnalysis records.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.database import db, Incident, PriorityAnalysis
from app.utils.priority_engine import analyze_incident_priority
from datetime import datetime

app = create_app()

with app.app_context():
    incidents = Incident.query.order_by(Incident.incident_code).all()
    print(f"Total incidents: {len(incidents)}")
    print()

    pa_created = 0
    pa_skipped = 0
    priority_updated = 0

    for inc in incidents:
        # Recompute priority using existing function (not hardcoded values)
        priority = analyze_incident_priority(inc)
        new_score = priority['priority_score']
        new_cat   = priority['priority_category']
        new_expl  = json.dumps(priority['explanation'])
        new_time  = priority['time_history']

        old_score = inc.priority_score
        old_cat   = inc.priority_category

        # Bug A fix: update stale stored scores
        changed = (
            inc.priority_score is None or
            abs(inc.priority_score - new_score) > 0.01 or
            inc.priority_category != new_cat
        )
        if changed:
            inc.priority_score    = new_score
            inc.priority_category = new_cat
            inc.priority_explanation = new_expl
            inc.time_history_score = new_time
            inc.updated_at = datetime.utcnow()
            priority_updated += 1
            print(f"  {inc.incident_code}: UPDATED  "
                  f"score {old_score} → {new_score}  "
                  f"cat '{old_cat}' → '{new_cat}'")
        else:
            print(f"  {inc.incident_code}: OK       "
                  f"score={new_score}  cat='{new_cat}'")

        # Bug B fix: create PriorityAnalysis row if missing (never duplicate)
        existing_pa = PriorityAnalysis.query.filter_by(incident_id=inc.id).first()
        if existing_pa is None:
            pa = PriorityAnalysis(
                incident_id=inc.id,
                severity_score=priority['severity'],
                public_impact_score=priority['public_impact'],
                location_importance_score=priority['location_importance'],
                time_history_score=priority['time_history'],
                final_priority_score=priority['priority_score'],
                priority_category=priority['priority_category'],
                explanation_factors=json.dumps(priority['explanation']),
                computed_at=datetime.utcnow(),
                computed_by='migration_phase4',
            )
            db.session.add(pa)
            pa_created += 1
            print(f"    → PriorityAnalysis created")
        else:
            pa_skipped += 1
            print(f"    → PriorityAnalysis already exists (skipped)")

    db.session.commit()

    # Final verification
    total_inc = Incident.query.count()
    total_pa  = PriorityAnalysis.query.count()
    inc_ids   = {i.id for i in Incident.query.all()}
    pa_ids    = {pa.incident_id for pa in PriorityAnalysis.query.all()}
    still_missing = inc_ids - pa_ids

    print()
    print("=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(f"  Priority scores updated : {priority_updated}")
    print(f"  PriorityAnalysis created: {pa_created}")
    print(f"  PriorityAnalysis skipped: {pa_skipped}")
    print(f"  Total incidents         : {total_inc}")
    print(f"  Total PriorityAnalysis  : {total_pa}")
    print(f"  Incidents still missing : {len(still_missing)}")
    if still_missing:
        print(f"  WARNING: {still_missing}")
    else:
        print("  All incidents now have a PriorityAnalysis record.")

