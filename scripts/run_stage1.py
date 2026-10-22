"""Execute Stage-1 acceptance checks in fresh processes with durable logs.

No training is performed. Existing outputs are preserved. Default output is a
new timestamped folder under runs/. Downloading is only needed for missing
resources; --offline verifies the existing pinned cache without network use.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def protected_snapshot():
    paths = [ROOT / 'README.md', ROOT / 'resource_manifest.json',
             ROOT / 'requirements-data.lock.txt']
    paths += [ROOT / 'scripts' / name for name in ['reconstruct_contacts.py', 'audit_dataset.py', 'acquire_resources.py']]
    for folder in ['data', 'reports', 'output']:
        paths += [p for p in (ROOT / folder).rglob('*') if p.is_file()]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--official-check', type=Path,
                        help='Optional earlier official HTTP check to include as separate evidence')
    args = parser.parse_args()
    output = (args.output_dir or ROOT / 'runs' / ('stage1_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if output.exists():
        raise FileExistsError(f'Acceptance requires a new directory: {output}')
    output.mkdir(parents=True)
    logs = output / 'logs'
    logs.mkdir()
    before = protected_snapshot()
    save(output / 'preserved_files_before.json', before)
    environment = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
                       SDL_VIDEODRIVER='dummy', PYGAME_HIDE_SUPPORT_PROMPT='1')
    report = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'run_directory': str(output), 'fresh_process_per_command': True,
              'model_training_started': False, 'commands': [], 'checks': {}, 'failures': []}

    def step(name, arguments, expected=0, timeout=180):
        command = [sys.executable, *arguments]
        started = time.perf_counter()
        entry = {'name': name, 'arguments': command, 'cwd': str(ROOT),
                 'expected_exit_code': expected, 'stdout_log': f'logs/{name}.stdout.txt',
                 'stderr_log': f'logs/{name}.stderr.txt'}
        # A copy/paste command uses PowerShell literal quoting, not shell execution.
        entry['powershell_command'] = '& ' + ' '.join("'" + s.replace("'", "''") + "'" for s in command)
        try:
            completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                                       text=True, encoding='utf-8', errors='replace', timeout=timeout,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            (logs / f'{name}.stdout.txt').write_text(completed.stdout, encoding='utf-8')
            (logs / f'{name}.stderr.txt').write_text(completed.stderr, encoding='utf-8')
            entry.update(exit_code=completed.returncode, passed=completed.returncode == expected)
        except Exception as error:
            entry.update(passed=False, error=f'{type(error).__name__}: {error}')
        entry['duration_s'] = time.perf_counter() - started
        report['commands'].append(entry)
        if not entry['passed']:
            report['failures'].append(name)
        save(output / 'acceptance.json', report)
        print(f"{name}: {'passed' if entry['passed'] else 'FAILED'} ({entry['duration_s']:.2f}s)", flush=True)
        return entry['passed']

    def check(name, function):
        try:
            result = function()
            if result is False:
                raise AssertionError(name)
            report['checks'][name] = {'passed': True, 'result': result}
        except Exception as error:
            report['checks'][name] = {'passed': False, 'error': f'{type(error).__name__}: {error}'}
            report['failures'].append(name)

    def locked_versions():
        versions = {}
        for line in (ROOT / 'requirements-data.lock.txt').read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            name, expected = line.split('==', 1)
            actual = importlib.metadata.version(name)
            if actual != expected:
                raise AssertionError(f'{name}: expected {expected}, installed {actual}')
            versions[name] = actual
        if sys.version_info[:2] != (3, 12):
            raise AssertionError('This acceptance recipe is verified with Python 3.12')
        return versions
    step('pip_check', ['-m', 'pip', 'check'])
    check('installed_distributions_match_lock', locked_versions)
    resources_ok = step('resources', ['scripts/prepare_resources.py', '--output-dir', str(output / 'resources'), *(['--offline'] if args.offline else [])])
    environment_ok = step('environment', ['scripts/check_environment.py', '--output-dir', str(output / 'inspection')])
    if environment_ok:
        check('independent_environment_and_imports', lambda: read(output / 'inspection/environment_check.json')['python']['is_project_virtual_environment']
              and read(output / 'inspection/environment_check.json')['data_dependencies_import_ok'])
    if resources_ok:
        data_ok = step('data_inspection', ['scripts/inspect_data.py', '--output-dir', str(output / 'inspection')])
        if data_ok:
            def data_integrity():
                info = read(output / 'inspection/data_inspection.json')
                raw = info['raw_data']
                return bool(info['source_manifest_check']['all_passed'] and raw['numeric_arrays_finite']
                    and info['simulator_parameters']['geometry']['declared_outline_matches_union']
                    and all(t['frame_indices_contiguous'] and t['timestamp_monotonic']
                            and t['timestamp_max_error_from_frame_index_over_fps'] < 1e-4
                            for t in raw['timestamp_checks']))
            check('raw_data_time_and_geometry', data_integrity)
        replay_ok = step('replay_3_episodes', ['scripts/replay_stage1.py', '--episodes', '3', '--output-dir', str(output / 'replay')])
        if replay_ok:
            step('contact_visualization', ['scripts/visualize_contacts.py', '--input',
                 str(output / 'replay/data/derived/probe_3_episodes/episode_000000.npz'),
                 '--output-dir', str(output / 'visualization')])
            check('fresh_replay_audited', lambda: read(output / 'replay/fresh_audit.json')['status'] == 'passed')
            def visual_alignment():
                metrics = read(output / 'replay/reports/probe_3_episodes/replay_validation.json')
                # Coarse regression alarm only; never filters demonstrations or
                # certifies exact object pose/contact truth.
                if metrics['visible_mask_iou_mean'] < 0.90 or metrics['visible_mask_iou_min'] < 0.70:
                    raise AssertionError('Large visible replay mismatch; inspect paired images')
                return {'mean_iou': metrics['visible_mask_iou_mean'], 'minimum_iou': metrics['visible_mask_iou_min'],
                        'minimum_mean_threshold': 0.90, 'minimum_frame_threshold': 0.70,
                        'purpose': 'Coarse regression check, not exact pose/contact ground truth; no data filtering.'}
            check('video_alignment_regression', visual_alignment)
    demo_a = step('simple_push_a', ['scripts/simple_push.py', '--output-dir', str(output / 'simple_push_a')])
    demo_b = step('simple_push_b', ['scripts/simple_push.py', '--output-dir', str(output / 'simple_push_b')])
    if demo_a and demo_b:
        def compare_demos():
            with np.load(output / 'simple_push_a/simple_push.npz', allow_pickle=False) as a, np.load(output / 'simple_push_b/simple_push.npz', allow_pickle=False) as b:
                if set(a.files) != set(b.files):
                    raise AssertionError('Different output fields')
                mismatched = [key for key in a.files if not np.array_equal(a[key], b[key])]
                if mismatched:
                    raise AssertionError('Different trajectories: ' + ', '.join(mismatched))
                return {'identical_fields': len(a.files), 'comparison': 'exact array equality in two fresh processes'}
        check('demo_reproducibility', compare_demos)
        demo = read(output / 'simple_push_a/simple_push.json')
        check('real_contact_and_object_motion', lambda: demo['observed']['contact_point_events'] > 0
              and demo['observed']['block_cog_displacement_norm'] > 1
              and demo['observed']['no_contact_labels_are_zero'])
        unchanged = sha(output / 'simple_push_a/simple_push.npz')
        step('refuse_existing_demo_output', ['scripts/simple_push.py', '--output-dir', str(output / 'simple_push_a')], expected=1)
        check('existing_demo_output_preserved', lambda: sha(output / 'simple_push_a/simple_push.npz') == unchanged)
        check('existing_output_error_is_explicit', lambda: 'FileExistsError' in (logs / 'refuse_existing_demo_output.stderr.txt').read_text(encoding='utf-8'))
    if args.official_check:
        evidence = args.official_check.resolve()
        def current_official_sources():
            earlier = read(evidence)
            if earlier['status'] != 'passed':
                raise AssertionError('Official source check did not pass')
            manifest = read(ROOT / 'resource_manifest.json')
            required = {'dataset_metadata': (f"https://huggingface.co/datasets/lerobot/pusht/resolve/{manifest['dataset_revision']}/meta/info.json", ROOT / 'data/raw/lerobot_pusht/meta/info.json'),
                        'project_website': ('https://diffusion-policy.cs.columbia.edu/', None)}
            for name, (repo, revision) in manifest['repositories'].items():
                required[name] = (f'https://raw.githubusercontent.com/{repo}/{revision}/README.md', ROOT / f'external/{name}/README.md')
            entries = {entry['name']: entry for entry in earlier['sources']}
            if set(entries) != set(required) or len(entries) != len(earlier['sources']):
                raise AssertionError('Missing or duplicate official sources')
            for name, (url, local) in required.items():
                item = entries[name]
                if item['status'] != 'passed' or item['url'] != url or item['http_status'] != 200:
                    raise AssertionError(f'Unexpected source/version: {name}')
                if local and (item['sha256'] != sha(local) or not item['matches_local']):
                    raise AssertionError(f'Official record differs from current local file: {name}')
            return {'required_sources': len(required), 'pinned_versions_and_local_hashes_match': True}
        check('official_sources_verified', current_official_sources)
        report['official_check'] = {'path': str(evidence), 'sha256': sha(evidence),
                                    'scope': 'Separate earlier network process; not re-fetched by offline acceptance.'}
        save(output / 'official_sources.json', read(evidence))
    def preserved():
        changed = [path for path, digest in before.items() if not (ROOT / path).is_file() or sha(ROOT / path) != digest]
        if changed:
            raise AssertionError('Existing files changed: ' + ', '.join(changed))
        return {'existing_files_verified_unchanged': len(before)}
    check('existing_project_files_preserved', preserved)
    report['source_files'] = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'scripts').glob('*.py')}
    report['requirements_sha256'] = sha(ROOT / 'requirements-data.lock.txt')
    report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['status'] = 'passed' if not report['failures'] else 'failed'
    save(output / 'acceptance.json', report)
    print(json.dumps({'status': report['status'], 'output': str(output), 'commands': len(report['commands']),
                      'checks': len(report['checks']), 'failures': report['failures']}), flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
