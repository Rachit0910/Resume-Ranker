import csv
import json
import sys
sys.path.insert(0, "src")
from features import honeypot_flags

with open("output/submission.csv") as f:
    rows = list(csv.DictReader(f))
top_ids = set(r["candidate_id"] for r in rows)
print(f"checking {len(top_ids)} candidates from submission.csv")

flagged = 0
checked = 0
with open("data/candidates.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        if c["candidate_id"] in top_ids:
            checked += 1
            flags = honeypot_flags(c)
            if flags:
                flagged += 1
                print(c["candidate_id"], "->", flags)

print()
print(f"checked: {checked}/100")
print(f"flagged: {flagged}")
print(f"honeypot rate: {flagged}%  (must be <= 10% to pass Stage 3)")