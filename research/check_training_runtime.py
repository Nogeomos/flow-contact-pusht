"""Record the real training runtime and exercise GPU forward/backward operations."""
from pathlib import Path
import argparse, importlib.metadata, json, platform, subprocess, sys, time
import torch
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    x=torch.randn(512,512,device=device,requires_grad=True)
    loss=(x@x.T).square().mean();loss.backward()
    if device=='cuda':torch.cuda.synchronize()
    assert bool(torch.isfinite(loss)) and bool(torch.isfinite(x.grad).all())
    versions={n:importlib.metadata.version(n) for n in ['torch','numpy','diffusers','einops','pymunk','gymnasium','av','pyarrow']}
    result=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),versions=versions,
                torch_cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available(),device=device,
                gpu_name=torch.cuda.get_device_name() if device=='cuda' else None,
                gpu_properties=str(torch.cuda.get_device_properties(0)) if device=='cuda' else None,
                loss=float(loss.detach()),gradient_finite=True,
                nvidia_smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout,
                environment_scope='Independent .venv-train; stage1 .venv unchanged')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
