"""Read-only check of installed distribution records and duplicate metadata."""
from pathlib import Path
import argparse, base64, csv, hashlib, importlib.metadata as md, json, sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    groups={}
    for d in md.distributions():groups.setdefault(d.metadata['Name'].lower().replace('_','-'),[]).append(d)
    duplicates={k:[dict(version=d.version,path=str(d._path)) for d in ds] for k,ds in groups.items() if len(ds)>1}
    files={}
    for name in ['filelock','fsspec']:
        d=md.distribution(name);checks=[];bad=[]
        for line in csv.reader(d.read_text('RECORD').splitlines()):
            relative,digest,size=line
            if not digest or not relative.startswith(name+'/'):continue
            path=d.locate_file(relative)
            algo,expected=digest.split('=',1)
            actual=base64.urlsafe_b64encode(hashlib.new(algo,path.read_bytes()).digest()).decode().rstrip('=') if path.exists() else None
            checks.append(relative)
            if expected!=actual:bad.append(relative)
        files[name]={'selected_version':d.version,'selected_metadata':str(d._path),'checked_source_files':len(checks),'mismatches':bad}
    result={'python':sys.executable,'duplicate_metadata':duplicates,'active_package_record_checks':files,
            'all_selected_source_records_match':all(not v['mismatches'] for v in files.values())}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
