"""Portable viewing bundle: report, figures, animation, and measured summaries.

This is for reviewing results on another device, not a packaged Python runtime.
"""
from pathlib import Path
import argparse, hashlib, json, zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/stages234_20261022')
    p.add_argument('--output',default='output/PushT_实验结果查看包.zip');a=p.parse_args()
    base=ROOT/a.run;out=ROOT/a.output
    if out.exists():raise FileExistsError(out)
    files={ROOT/'output/pdf/PushT_第二至四阶段实验报告_2026-10-22.pdf':'report.pdf'}
    for path in (base/'figures').glob('*'):
        if path.suffix in ['.png','.gif']:files[path]='figures/'+path.name
    for name in ['rule_test','contact_test','contact_seed1_test','contact_seed2_test','flow_action_test','visual_dp_test','contact_no_flow_test']:
        files[base/name/'summary.json']='metrics/'+name+'.json'
    for name in ['statistical_analysis.json','final_acceptance.json','log_collision_recovery.json']:
        files[base/name]='metrics/'+name
    for name in ['STAGES234_LIMITATIONS_AND_FAILURES.md','STAGES234_DATA_SCHEMA.md','STAGES234_PROTOCOL.md']:
        path=ROOT/'docs'/name
        if path.exists():files[path]='notes/'+name
    manifest=[]
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path,arc in files.items():
            data=path.read_bytes();z.writestr(arc,data)
            manifest.append({'file':arc,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        z.writestr('README.txt','PushT 第二至第四阶段实际实验结果\n\n先打开 report.pdf。\n'
                   'figures/comparison_episode_12.gif 是同一测试初态下四种方法的真实动画。\n'
                   'metrics 中是原始结果摘要，notes 中是研究边界与字段说明。\n\n'
                   '这是跨设备查看包，不包含Python环境或全部训练数据。完整代码、模型和复现命令保存在原项目目录的STAGES234_README.md。\n'
                   '元数据中的原机器绝对路径是证据记录；在本查看包里无需访问那些路径。\n')
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for record in manifest:assert hashlib.sha256(z.read(record['file'])).hexdigest()==record['sha256']
    print(json.dumps({'file':str(out),'bytes':out.stat().st_size,'verified_files':len(manifest)}),flush=True)

if __name__=='__main__':main()
