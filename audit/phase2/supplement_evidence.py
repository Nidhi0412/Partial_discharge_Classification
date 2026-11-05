"""Additional read-only joins, OOXML text extraction and preservation checks."""
import csv
import hashlib
import json
import xml.etree.ElementTree as E
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

OUT=Path(__file__).resolve().parent
WORK=Path(r'D:\Nidhiiiiii')
PAPER=WORK/'Paper_04_Saurabh_wavelet/Paper_04_Saurabh_wavelet/with new results'
MIN=OUT.parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

archive=next(PAPER.glob('Insulation*.zip'))
links=[]
with zipfile.ZipFile(archive) as z:
    for fold in range(2,6):
        n=next(n for n in z.namelist() if n.endswith(f'fold_{fold}/detailed_classification_report.txt'))
        b=z.read(n)
        local=MIN/f'results/VGG16/VGG16/fold_{fold}_report.txt'
        links.append(dict(archive=str(archive),member=n,sha256=sha(b),export_path=str(local),export_sha256=sha(local.read_bytes()),
                          byte_equal=b==local.read_bytes(),text_equal=b.decode().strip()==local.read_text().strip()))
dump('REPORT_LINKS.json',links)

provenance=list(csv.DictReader((OUT/'DATASET_PROVENANCE.csv').open(encoding='utf-8')))
byhash=defaultdict(list)
for x in provenance:byhash[x['image_sha256']].append(x)
joins=[]
for h,rows in byhash.items():
    full=[x for x in rows if x['dataset']=='Full_4Class_Classifier']
    ins=[x for x in rows if x['dataset']=='Insulation_classifier']
    assert Counter(x['material_folder_label'] for x in full)==Counter(x['material_folder_label'] for x in ins)
    if Counter(x['partition'] for x in full)!=Counter(x['partition'] for x in ins):
        joins.append(dict(sha256=h,fourclass_partitions=dict(Counter(x['partition'] for x in full)),insulation_partitions=dict(Counter(x['partition'] for x in ins))))
dump('CROSS_DATASET_PARTITION_DIFFERENCES.json',joins)

zpath=WORK/'Mutlitask_PD_Classification-20261027T045711Z-1-001.zip'
counts=Counter();localhash={x['image_path']:x['image_sha256'] for x in provenance}
with zipfile.ZipFile(zpath) as z:
    for n in z.namelist():
        if n.endswith('.png') and '/Classifications_data/' in n:
            relative=n.split('/',1)[1]
            p=WORK/'Mutlitask_PD_Classification/Mutlitask_PD_Classification'/relative
            counts['png_members']+=1
            if str(p) in localhash and sha(z.read(n))==localhash[str(p)]:counts['identical_to_local']+=1
dump('DATASET_ARCHIVE_COPY_VERIFICATION.json',dict(archive=str(zpath),counts=dict(counts)))

doc=PAPER/'paper with hybrid results_2025.docx'
with zipfile.ZipFile(doc) as z:
    root=E.fromstring(z.read('word/document.xml'))
ps=[''.join(t.text or '' for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')) for p in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p')]
dump('MANUSCRIPT_TEXT_EVIDENCE.json',dict(path=str(doc),sha256=sha(doc.read_bytes()),
    paragraph_numbering='1-based w:p order in word/document.xml; includes table paragraphs',
    paragraphs={str(i):ps[i-1] for i in [29,65,66,67,165,166,168]}))

workbook=PAPER/'VGG16_insulation/crossvalidation_avg_classification_report.xlsx'
ns={'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
cells={}
with zipfile.ZipFile(workbook) as z:
    shared=[''.join(x.itertext()) for x in E.fromstring(z.read('xl/sharedStrings.xml'))]
    root=E.fromstring(z.read('xl/worksheets/sheet1.xml'))
    for c in root.findall('.//x:c',ns):
        v=c.find('x:v',ns)
        text=v.text if v is not None else ''.join(c.itertext())
        if c.attrib.get('t')=='s':text=shared[int(text)]
        cells[c.attrib['r']]=text
dump('WORKBOOK_CELL_EVIDENCE.json',dict(path=str(workbook),sha256=sha(workbook.read_bytes()),sheet='xl/worksheets/sheet1.xml',cells=cells))

# Assert byte-for-byte preservation against Phase 1 source inventory.
checked=0
for r in csv.DictReader((MIN/'audit/SOURCE_INVENTORY.csv').open()):
    assert sha((MIN/r['path']).read_bytes())==r['sha256'],r['path']
    checked+=1
dump('PRESERVATION_CHECK.json',dict(phase1_original_files_checked=checked,status='CONFIRMED',all_hashes_unchanged=True))
print(json.dumps(dict(report_links=links,partition_hash_groups_reassigned=len(joins),dataset_zip=counts,preservation_files=checked),indent=2))
