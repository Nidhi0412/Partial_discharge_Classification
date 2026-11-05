"""Recording-aware, non-destructive dataset indexing and splitting (stdlib only)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re

CLASS_NAMES = ('Nomex', 'Teflon')
FIELDS = ['image_id', 'image_path', 'recording_id', 'material', 'electrode', 'session_id', 'image_sha256',
          'source_csv_path', 'source_csv_sha256', 'wavelet', 'true_dwt_level', 'generation_config_sha256']
SPLIT_FIELDS = FIELDS + ['group_id', 'fold', 'partition']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows, fields):
    # Exclusive creation: source manifests and historical outputs are never replaced.
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_images(path):
    path = Path(path).resolve()
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Image manifest is empty')
    ids, paths, record_labels = set(), set(), {}
    for index, row in enumerate(rows, 2):
        row['recording_id'] = (row.get('recording_id') or row.get('original_recording_id') or '').strip()
        row['material'] = row.get('material', '').strip()
        if not row['recording_id'] or row['material'] not in CLASS_NAMES:
            raise ValueError(f'Row {index}: trusted recording_id and material (Nomex/Teflon) required')
        p = Path(row.get('image_path', ''))
        if not row.get('image_path'):
            raise ValueError(f'Row {index}: missing image_path')
        p = (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
        if not p.is_file() or p.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
            raise ValueError(f'Row {index}: missing/unsupported image {p}')
        row['image_path'] = str(p)
        row['image_id'] = row.get('image_id', '').strip() or hashlib.sha256(str(p).encode()).hexdigest()[:24]
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', row['image_id']):
            raise ValueError(f'Row {index}: image_id must be a safe identifier, not a path')
        if row['image_id'] in ids or str(p).casefold() in paths:
            raise ValueError(f'Row {index}: duplicate image ID or image path')
        ids.add(row['image_id']); paths.add(str(p).casefold())
        actual = digest(p)
        if row.get('image_sha256') and row['image_sha256'] != actual:
            raise ValueError(f'Row {index}: image content changed')
        row['image_sha256'] = actual
        row['electrode'] = row.get('electrode', '').strip()
        row['session_id'] = row.get('session_id', '').strip()
        for field in FIELDS:
            row.setdefault(field, '')
        label = (row['material'], row['electrode'], row['session_id'])
        if row['recording_id'] in record_labels and record_labels[row['recording_id']] != label:
            raise ValueError(f'Inconsistent acquisition metadata for recording {row["recording_id"]}')
        record_labels[row['recording_id']] = label
    return rows


def inventory_images(data_dir, output):
    """Create a mapping template; NEVER infer acquisition IDs from image numbering."""
    rows = []
    for p in sorted(Path(data_dir).resolve().rglob('*')):
        if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            material = next((part for part in p.parts if part in CLASS_NAMES), '')
            rows.append(dict(image_id=hashlib.sha256(str(p).encode()).hexdigest()[:24],
                             image_path=str(p), recording_id='', material=material,
                             electrode='', session_id='', image_sha256=digest(p)))
    if not rows:
        raise ValueError('No images found')
    write_csv(output, rows, FIELDS)
    return len(rows)


def assign_groups(rows, proportions, seed, group_by='recording_id'):
    """Deterministic greedy stratification by recording counts, material/electrode.

    Whole sessions can be grouped even if they contain multiple materials.
    This is a heuristic balance, not a claim of exact stratification.
    """
    if group_by not in {'recording_id', 'session_id', 'group_id'}:
        raise ValueError('Unknown group column')
    groups = defaultdict(list)
    for row in rows:
        key = row.get(group_by, '')
        if not key:
            raise ValueError(f'Missing {group_by}; cannot form independent groups')
        groups[key].append(row)
    k = len(proportions)
    if len(groups) < k or any(p <= 0 for p in proportions):
        raise ValueError('Insufficient groups or invalid split proportions')
    total_p = sum(proportions)
    proportions = [p / total_p for p in proportions]
    # Count recordings, not the number of derived images.
    features = {}
    for key, values in groups.items():
        recordings = {r['recording_id']: r for r in values}
        features[key] = Counter((r['material'], r['electrode']) for r in recordings.values())
    totals = sum(features.values(), Counter())
    rng = random.Random(seed)
    order = sorted(groups); rng.shuffle(order)
    order.sort(key=lambda key: (-sum(features[key].values()), -max(features[key].values())))
    counts = [Counter() for _ in range(k)]
    assignments = {}
    # Minimize normalized squared distance to desired per-stratum recording counts.
    for key in order:
        def score(part):
            cost = 0.0
            for j in range(k):
                for label, total in totals.items():
                    target = total * proportions[j]
                    value = counts[j][label] + (features[key][label] if j == part else 0)
                    cost += (value - target) ** 2 / max(target, 1.0)
            return cost
        choices = list(range(k)); rng.shuffle(choices)
        selected = min(choices, key=score)
        assignments[key] = selected
        counts[selected].update(features[key])
    if len(set(assignments.values())) != k:
        raise ValueError('Group constraints yielded an empty partition; fewer folds or more independent groups needed')
    return assignments


def validate_split(rows, group_by='recording_id'):
    if not rows:
        raise ValueError('Split manifest is empty')
    folds = sorted({r['fold'] for r in rows}, key=int)
    if len(folds) < 2:
        raise ValueError('At least two development folds required')
    seen = set(); validations = Counter(); reference_dev = reference_test = None
    canonical = {}
    for r in rows:
        if r['partition'] not in {'train', 'validation', 'test'}:
            raise ValueError('Unknown partition')
        if not r.get('recording_id') or not r.get('group_id') or not r.get('image_sha256'):
            raise ValueError('Missing recording/group/content identity')
        if r['group_id'] != r.get(group_by):
            raise ValueError('Stored group_id differs from grouping policy')
        identity = (r['image_path'], r['recording_id'], r['material'], r.get('electrode', ''),
                    r.get('session_id', ''), r['group_id'], r['image_sha256'])
        previous = canonical.setdefault(r['image_id'], identity)
        if previous != identity:
            raise ValueError('Image metadata differs across folds')
    for fold in folds:
        parts = {p: [r for r in rows if r['fold'] == fold and r['partition'] == p]
                 for p in ['train', 'validation', 'test']}
        for part, values in parts.items():
            if not values or {r['material'] for r in values} != set(CLASS_NAMES):
                raise ValueError(f'Fold {fold} {part} is empty or lacks a material class')
            for r in values:
                key = (fold, r['image_id'])
                if key in seen:
                    raise ValueError('An image has multiple assignments within a fold')
                seen.add(key)
        for key in ['recording_id', 'group_id', 'image_id', 'image_path', 'image_sha256']:
            sets = [{r[key] for r in parts[p]} for p in ['train', 'validation', 'test']]
            if any(sets[a] & sets[b] for a, b in [(0, 1), (0, 2), (1, 2)]):
                raise ValueError(f'Fold {fold}: {key} crosses partitions; inspect source mapping/content before splitting')
        dev = {r['image_id'] for r in parts['train'] + parts['validation']}
        test = {r['image_id'] for r in parts['test']}
        if reference_dev is None:
            reference_dev, reference_test = dev, test
        elif dev != reference_dev or test != reference_test:
            raise ValueError('Development/holdout membership changed across folds')
        validations.update(r['image_id'] for r in parts['validation'])
    if set(validations) != reference_dev or any(n != 1 for n in validations.values()):
        raise ValueError('Every development image must have exactly one validation fold')
    record_folds = defaultdict(set)
    for r in rows:
        if r['partition'] == 'validation':
            record_folds[r['group_id']].add(r['fold'])
    if any(len(v) != 1 for v in record_folds.values()):
        raise ValueError('A group has multiple outer validation folds')
    return folds


def content_groups(images,group_by):
    """Conservatively co-locate equal bytes without changing acquisition IDs."""
    parents={r[group_by]:r[group_by] for r in images}
    if '' in parents:raise ValueError(f'Missing {group_by}')
    def find(key):
        while parents[key]!=key:
            parents[key]=parents[parents[key]];key=parents[key]
        return key
    first={}
    for row in images:
        key=row[group_by];other=first.setdefault(row['image_sha256'],key)
        a,b=find(key),find(other)
        if a!=b:parents[max(a,b)]=min(a,b)
    members=defaultdict(set)
    for key in parents:members[find(key)].add(key)
    ids={root:next(iter(values)) if len(values)==1 else
         'content_group_'+hashlib.sha256(json.dumps(sorted(values)).encode()).hexdigest()[:24]
         for root,values in members.items()}
    for row in images:row['group_id']=ids[find(row[group_by])]
    return len(members)


def make_splits(manifest, output_dir, folds=5, test_fraction=.2, seed=42, group_by='recording_id',keep_identical_together=False):
    if folds < 2 or not 0 < test_fraction < .5:
        raise ValueError('Use >=2 folds and 0 < test_fraction < .5')
    images = read_images(manifest)
    acquisition_group_by=group_by
    if keep_identical_together:
        content_groups(images,group_by)
        group_by='group_id'
    holdout = assign_groups(images, [1-test_fraction, test_fraction], seed, group_by)
    dev = [r for r in images if holdout[r[group_by]] == 0]
    assignment = assign_groups(dev, [1] * folds, seed+1, group_by)
    rows = []
    for fold in range(folds):
        for original in images:
            r = dict(original); key = r[group_by]
            r.update(group_id=key, fold=str(fold+1), partition='test' if holdout[key] else
                     ('validation' if assignment[key] == fold else 'train'))
            rows.append(r)
    validate_split(rows, group_by)
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    write_csv(out/'splits.csv', rows, SPLIT_FIELDS)
    metadata = dict(schema_version=2, source_manifest=str(Path(manifest).resolve()),
                    source_manifest_sha256=digest(manifest), seed=seed, folds=folds,
                    test_fraction=test_fraction, group_by=group_by, class_names=list(CLASS_NAMES),
                    acquisition_group_by=acquisition_group_by,keep_identical_together=keep_identical_together,
                    effective_groups=len({r['group_id'] for r in rows}),
                    grouping_note='Equal-content files co-located only for split/uncertainty; recording IDs unchanged' if keep_identical_together else 'Acquisition grouping only',
                    split_method='deterministic greedy stratification of recording counts by material/electrode',
                    partitions={p: len({r['image_id'] for r in rows if r['partition'] == p}) for p in ['train','validation','test']},
                    split_sha256=digest(out/'splits.csv'))
    (out/'split_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return out/'splits.csv'


def load_splits(path):
    path = Path(path).resolve()
    config = json.loads((path.parent/'split_config.json').read_text(encoding='utf-8'))
    if digest(path) != config['split_sha256']:
        raise ValueError('Split manifest changed after construction')
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    validate_split(rows, config['group_by'])
    verified = set()
    for r in rows:
        if r['image_path'] not in verified:
            if digest(r['image_path']) != r['image_sha256']:
                raise ValueError(f'Image changed: {r["image_path"]}')
            verified.add(r['image_path'])
    return rows, config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inventory = sub.add_parser('inventory')
    inventory.add_argument('--data-dir', required=True); inventory.add_argument('--output', required=True)
    split = sub.add_parser('split')
    split.add_argument('--manifest', required=True); split.add_argument('--output-dir', required=True)
    split.add_argument('--folds', type=int, default=5); split.add_argument('--test-fraction', type=float, default=.2)
    split.add_argument('--seed', type=int, default=42)
    split.add_argument('--group-by', choices=['recording_id','session_id'], default='recording_id')
    split.add_argument('--keep-identical-together',action='store_true',help='Co-locate equal bytes without equating their recording IDs')
    args = parser.parse_args(argv)
    if args.command == 'inventory':
        print(f'Created {inventory_images(args.data_dir, args.output)} mapping rows. Fill trusted recording IDs before splitting.')
    else:
        print(make_splits(args.manifest, args.output_dir, args.folds, args.test_fraction, args.seed, args.group_by,args.keep_identical_together))


if __name__ == '__main__':
    main()
