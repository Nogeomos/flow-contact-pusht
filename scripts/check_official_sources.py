"""Read official public sources and compare pinned remote files with local files."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / 'official_sources.json'
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    manifest = json.loads((ROOT / 'resource_manifest.json').read_text(encoding='utf-8'))
    dataset_rev = manifest['dataset_revision']
    sources = [('dataset_metadata', f'https://huggingface.co/datasets/lerobot/pusht/resolve/{dataset_rev}/meta/info.json', ROOT / 'data/raw/lerobot_pusht/meta/info.json')]
    for name, (repo, commit) in manifest['repositories'].items():
        sources.append((name, f'https://raw.githubusercontent.com/{repo}/{commit}/README.md', ROOT / f'external/{name}/README.md'))
    sources.append(('project_website', 'https://diffusion-policy.cs.columbia.edu/', None))
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed', 'sources': []}
    for name, url, local in sources:
        entry = {'name': name, 'url': url}
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'FlowContact-Stage1'})
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()
                entry.update(http_status=response.status, final_url=response.url, bytes=len(content))
            entry['sha256'] = hashlib.sha256(content).hexdigest()
            if local:
                entry['matches_local'] = content == local.read_bytes()
                if not entry['matches_local']:
                    raise ValueError('Pinned remote content differs from local source')
            if name == 'dataset_metadata':
                entry['metadata'] = json.loads(content)
            entry['status'] = 'passed'
        except Exception as error:
            entry.update(status='failed', error=f'{type(error).__name__}: {error}')
            report['status'] = 'partial'
        report['sources'].append(entry)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'sources': [{k:v for k,v in entry.items() if k != 'metadata'} for entry in report['sources']]}), flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
