"""Verify pinned resources; download/extract only missing files, never overwrite."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def run(output, offline=False):
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'resources.json'
    if destination.exists():
        raise FileExistsError(f'Refusing to replace {destination}')
    manifest_path = ROOT / 'resource_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    result = {'status': 'running', 'manifest_sha256': sha(manifest_path),
              'dataset_revision': manifest['dataset_revision'], 'repositories': manifest['repositories'],
              'files': [], 'extracted_sources': [], 'network_used': False}
    try:
        for item in manifest['files']:
            path = (ROOT / item['path']).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError('Resource path escapes project')
            origin = 'verified_existing'
            if not path.exists():
                if offline:
                    raise FileNotFoundError(f'Offline mode: missing {path}')
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
                request = urllib.request.Request(item['url'], headers={'User-Agent': 'FlowContact-Stage1'})
                with urllib.request.urlopen(request, timeout=60) as response, temp.open('xb') as target:
                    shutil.copyfileobj(response, target)
                if temp.stat().st_size != item['bytes'] or sha(temp) != item['sha256']:
                    raise ValueError(f'Download checksum mismatch; preserved partial: {temp}')
                # Windows rename fails if another process created the destination.
                temp.rename(path)
                origin = 'downloaded_missing'
                result['network_used'] = True
            if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError(f'Existing file differs from pinned manifest; not overwritten: {path}')
            result['files'].append({'path': item['path'], 'bytes': item['bytes'],
                                    'sha256': item['sha256'], 'origin': origin})
        for name, (_, revision) in manifest['repositories'].items():
            archive = ROOT / f'downloads/{name}-{revision}.zip'
            folder = (ROOT / 'external' / name).resolve()
            count, created = 0, 0
            with zipfile.ZipFile(archive) as zipped:
                for member in zipped.infolist():
                    parts = Path(member.filename).parts[1:]
                    if not parts or member.is_dir():
                        continue
                    target = folder.joinpath(*parts).resolve()
                    if not target.is_relative_to(folder):
                        raise ValueError('Unsafe archive member')
                    content = zipped.read(member)
                    expected = hashlib.sha256(content).hexdigest()
                    if target.exists():
                        if sha(target) != expected:
                            raise ValueError(f'Local source differs from pinned archive, preserved: {target}')
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open('xb') as handle:
                            handle.write(content)
                        created += 1
                    count += 1
            result['extracted_sources'].append({'repository': name, 'revision': revision,
                                                 'verified_files': count, 'created_missing_files': created})
        result['status'] = 'passed'
    except Exception as error:
        result['status'] = 'failed'
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'network_used', 'extracted_sources']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--offline', action='store_true')
    arguments = parser.parse_args()
    run(arguments.output_dir.resolve(), arguments.offline)
