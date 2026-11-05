"""Static historical recovery. Reads originals; writes only alongside this script.
No model loading, inference, pickle deserialization, dependencies or training.
"""
import csv
import hashlib
import json
import re
import struct
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

WORK = Path(r'D:\Nidhiiiiii')
OUT = Path(__file__).resolve().parent
MIN = OUT.parents[1]
PAPER = WORK/'Paper_04_Saurabh_wavelet/Paper_04_Saurabh_wavelet/with new results'
DATA = WORK/'Mutlitask_PD_Classification/Mutlitask_PD_Classification/Classifications_data/Classifications_data'
SKIP = {'.git','.venv','venv','node_modules','__pycache__','audit'}

def sha(b): return hashlib.sha256(b).hexdigest()
def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def table(name,rows,fields=None):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def file_sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()

files=[p for p in WORK.rglob('*') if p.is_file() and not any(x in SKIP for x in p.relative_to(WORK).parts)]
table('WORKSPACE_FILE_INVENTORY.csv',[dict(path=str(p),size_bytes=p.stat().st_size) for p in files])
archive_rows=[]
archive_hashes=[]
for p in files:
    if p.suffix.lower()!='.zip':continue
    archive_hashes.append(dict(path=str(p),size_bytes=p.stat().st_size,sha256=file_sha(p)))
    with zipfile.ZipFile(p) as z:
        for i in z.infolist():
            if not i.is_dir():archive_rows.append(dict(archive=str(p),member=i.filename,size_bytes=i.file_size,crc32=f'{i.CRC:08x}'))
table('ARCHIVE_MEMBER_INVENTORY.csv',archive_rows)
table('ARCHIVE_HASHES.csv',archive_hashes)

# Genuine processed-image records. Blank source IDs and transform fields mean unknown.
images=[]
by_hash=defaultdict(list)
for ds in sorted(DATA.iterdir()):
    if not ds.is_dir():continue
    for p in sorted(ds.rglob('*.png')):
        b=p.read_bytes();assert b[:8]==b'\x89PNG\r\n\x1a\n'
        width,height,depth,color=struct.unpack('>IIBB',b[16:26])
        part,cls=p.relative_to(ds).parts[:2]
        row=dict(dataset=ds.name,image_path=str(p),image_sha256=sha(b),size_bytes=len(b),
                 width=width,height=height,png_bit_depth=depth,png_color_type=color,
                 material_folder_label=cls.split('_')[0],electrode_folder_label=cls.split('_')[1] if '_' in cls else '',
                 partition=part,original_recording_id='',original_csv_path='',wavelet='',dwt_array_position='',true_dwt_level='',
                 acquisition_conditions='',same_content_fourclass_paths='',same_content_fourclass_labels='',
                 source_mapping_status='NOT VERIFIABLE',historical_AUT_membership_status='NOT VERIFIABLE',
                 observed_file_status='CONFIRMED',label_evidence='Observed folder label, not independently verified acquisition metadata')
        images.append(row);by_hash[row['image_sha256']].append(row)
for row in images:
    matches=[x for x in by_hash[row['image_sha256']] if x['dataset']=='Full_4Class_Classifier']
    row['same_content_fourclass_paths']=json.dumps([x['image_path'] for x in matches])
    row['same_content_fourclass_labels']=json.dumps(sorted({Path(x['image_path']).parent.name for x in matches}))
table('DATASET_PROVENANCE.csv',images)
duplicates=[]
summary={}
for ds in sorted({x['dataset'] for x in images}):
    subset=[x for x in images if x['dataset']==ds]
    groups=defaultdict(list)
    for row in subset:groups[row['image_sha256']].append(row)
    cross=[]
    for h,rows in groups.items():
        if len({x['partition'] for x in rows})>1:
            cross.append(rows)
            for row in rows:duplicates.append(dict(dataset=ds,sha256=h,partition=row['partition'],image_path=row['image_path'],status='CONFIRMED'))
    trains={x['image_sha256'] for x in subset if x['partition']=='Train'}
    tests=[x for x in subset if x['partition']=='Test']
    summary[ds]=dict(images=len(subset),unique_byte_hashes=len(groups),partition_counts=dict(Counter(x['partition'] for x in subset)),
       class_partition_counts=dict(Counter(x['partition']+'/'+Path(x['image_path']).parent.name for x in subset)),
       multiplicities=dict(Counter(len(v) for v in groups.values())),cross_partition_hash_groups=len(cross),
       cross_partition_files=sum(len(v) for v in cross),test_images_with_train_identical_bytes=sum(x['image_sha256'] in trains for x in tests),
       test_images=len(tests),class_unique_hashes={c:len({x['image_sha256'] for x in subset if Path(x['image_path']).parent.name==c}) for c in sorted({Path(x['image_path']).parent.name for x in subset})})
table('EXACT_DUPLICATE_OVERLAP.csv',duplicates)
full=Counter(x['image_sha256'] for x in images if x['dataset']=='Full_4Class_Classifier')
ins=Counter(x['image_sha256'] for x in images if x['dataset']=='Insulation_classifier')
summary['cross_dataset_content_multisets_equal']=full==ins
dump('DATASET_COUNTS_AND_OVERLAP.json',summary)

# Preserve actual ZIP-contained config JSON as evidence, not imported models.
archive=next(PAPER.glob('Insulation*.zip'))
checkpoints=[];members=[];histories=[];events=[]
local_hist={str(p):sha(p.read_bytes()) for p in PAPER.rglob('training_history.json')}
with zipfile.ZipFile(archive) as z:
    for n in z.namelist():
        if n.endswith('.h5'):
            h=hashlib.sha256()
            with z.open(n) as f:
                head=f.read(1024*1024);h.update(head)
                while b:=f.read(1024*1024):h.update(b)
            assert head[:8]==b'\x89HDF\r\n\x1a\n'
            assert b'model_config' in head
            start=head.find(b'{"class_name"')
            config,end=json.JSONDecoder().raw_decode(head[start:].decode('utf-8',errors='replace'))
            layers=config['config']['layers']
            names=[x['config']['name'] for x in layers]
            conv=[x['config']['name'] for x in layers if x['class_name']=='Conv2D']
            assert len(conv)==13 and all(f'block5_conv{i}' in conv for i in [1,2,3])
            assert not any(x['class_name']=='Add' for x in layers)
            fn='checkpoint_'+n.split('/')[-2]+'_'+Path(n).stem+'_model_config.json'
            dump(fn,config)
            training_start=head.find(b'{"loss"')
            training=None
            if training_start>=0:
                training,_=json.JSONDecoder().raw_decode(head[training_start:].decode(errors='replace'))
                dump(fn.replace('_model_config','_training_config'),training)
            checkpoints.append(dict(archive=str(archive),member=n,checkpoint_sha256=h.hexdigest(),
              model_config_json_start_byte=start,config_snapshot=fn,backbone='VGG16',status='CONFIRMED',
              layer_count=len(layers),conv2d_count=len(conv),pool_count=sum(x['class_name']=='MaxPooling2D' for x in layers),
              trainable_convolutions=[x['config']['name'] for x in layers if x['class_name']=='Conv2D' and x['config'].get('trainable')],
              output_config=layers[-1]['config'],training_config=training))
        elif n.endswith(('.json','.csv','.txt')):
            b=z.read(n)
            fn='archive_'+n.split('/')[-2]+'_'+Path(n).name
            (OUT/fn).write_bytes(b)
            members.append(dict(archive=str(archive),member=n,sha256=sha(b),snapshot=fn))
            if n.endswith('training_history.json'):
                obj=json.loads(b)
                histories.append(dict(member=n,sha256=sha(b),epochs=len(obj['accuracy']),keys=list(obj),
                  identical_local_files=[p for p,h in local_hist.items() if h==sha(b)]))
        elif 'events.out.tfevents' in n:
            b=z.read(n)
            # TFRecord envelope only; no TensorFlow/protobuf imports or execution.
            offset=0;records=[]
            while offset+16<=len(b):
                length=struct.unpack('<Q',b[offset:offset+8])[0]
                if offset+16+length>len(b):break
                payload=b[offset+12:offset+12+length]
                records.append(payload)
                offset+=16+length
            events.append(dict(member=n,sha256=sha(b),records=len(records),first_record_hex=records[0].hex() if records else '',
                               filename_references=[s for s in ['Insulation_classifier','Nomex','Teflon','train_idx','val_idx'] if s.encode() in b]))
dump('CHECKPOINT_METADATA.json',checkpoints)
dump('ARCHIVE_TEXT_EVIDENCE.json',members)
dump('HISTORY_LINKS.json',histories)
dump('TENSORBOARD_RECORD_INVENTORY.json',events)

# Hash all recovered small historical text artifacts and source, and snapshot git evidence.
evidence=[]
for p in files:
    if p.suffix.lower() in {'.py','.txt','.json','.log','.csv'} and (str(PAPER) in str(p) or str(MIN/'code') in str(p)):
        evidence.append(dict(path=str(p),sha256=file_sha(p),size_bytes=p.stat().st_size))
table('HISTORICAL_SOURCE_HASHES.csv',evidence)
git=[]
for top in WORK.iterdir():
    if top.is_dir():
        for p in top.rglob('.git'):
            if not p.is_dir():continue
            proc=subprocess.run(['git','--no-optional-locks','-c',f'safe.directory={p.parent.as_posix()}','-C',str(p.parent),'log','--all','--format=%H %aI %s'],capture_output=True,text=True)
            git.append(dict(repository=str(p.parent),returncode=proc.returncode,log=proc.stdout,error=proc.stderr))
dump('GIT_HISTORY_SCOPE.json',git)

# Read-only raw CSV header inventory: no claim that samples belong to AUT images.
raw=[]
for p in files:
    if p.suffix.lower()!='.csv':continue
    with p.open(errors='replace') as f:lines=[next(f,'').rstrip('\n') for _ in range(22)]
    if any('Sample Interval' in x or 'Waveform Type' in x for x in lines):
        raw.append(dict(path=str(p),sha256=file_sha(p),header_lines=lines))
dump('RAW_CSV_CANDIDATES.json',raw)
print(json.dumps(dict(workspace_files=len(files),archives=len(archive_hashes),checkpoints=len(checkpoints),datasets=summary,
 raw_candidates=len(raw),history_links=histories),indent=2))
