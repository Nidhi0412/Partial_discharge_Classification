"""Validate a supplied, populated manifest; never invent recording IDs or partitions.
Usage: python audit/check_split_manifest.py path/to/manifest.csv
Columns follow SPLIT_MANIFEST_TEMPLATE.csv. No training or file changes.
"""
import csv
import sys
from collections import defaultdict

def check(rows):
    assert rows, 'Manifest is empty: provenance and split checks are blocked'
    splits = defaultdict(lambda: defaultdict(list))
    global_tests = set()
    global_dev = set()
    for r in rows:
        for key in ['original_recording_id','material','image_path','image_sha256','outer_fold','partition']:
            assert r.get(key, '').strip(), f'Missing {key}'
        assert r['partition'] in {'train','validation','test'}, 'Unknown partition'
        splits[r['outer_fold']][r['partition']].append(r)
        (global_tests if r['partition']=='test' else global_dev).add(r['original_recording_id'])
    # Contract: this manifest describes fixed independent holdout + development CV.
    assert global_tests.isdisjoint(global_dev), 'Holdout recording used during development'
    for fold, parts in splits.items():
        assert all(parts[p] for p in ['train','validation','test']), f'Incomplete fold {fold}'
        for key in ['original_recording_id','image_path','image_sha256']:
            sets = {p:{r[key] for r in parts[p]} for p in parts}
            for a,b in [('train','validation'),('train','test'),('validation','test')]:
                assert sets[a].isdisjoint(sets[b]), f'{key} overlap {fold}: {a}/{b}'
    memberships = [{r['image_path'] for r in parts['validation']} for parts in splits.values()]
    assert len(memberships)>1, 'Need multiple folds'
    assert len({frozenset(x) for x in memberships})==len(memberships), 'Repeated fixed validation membership'
    for i,a in enumerate(memberships):
        for b in memberships[i+1:]:
            assert a.isdisjoint(b), 'An image has multiple outer validation assignments'
    group_folds = defaultdict(set)
    for fold, parts in splits.items():
        for r in parts['validation']:
            group_folds[r['original_recording_id']].add(fold)
    assert all(len(v)==1 for v in group_folds.values()), 'Recording crosses outer validation folds'
    print('PASS: populated manifest satisfies grouped membership and exact-hash overlap checks; near-duplicates need separate inspection')

if __name__=='__main__':
    with open(sys.argv[1], newline='', encoding='utf-8-sig') as f:
        check(list(csv.DictReader(f)))
