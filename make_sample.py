import json

candidates = []
with open('data/candidates.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 50:
            break
        candidates.append(json.loads(line))

with open('sandbox/sample_candidates.json', 'w', encoding='utf-8') as f:
    json.dump(candidates, f, indent=2)

print(f"Saved {len(candidates)} candidates to sandbox/sample_candidates.json")