"""Re-run a small public replay in a NEW directory and independently audit it.

The previously implemented replay/audit kernels are reused, but old derived
datasets and reports are never read as evidence and are never overwritten.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import numpy as np
import pyarrow.parquet as pq

import reconstruct_contacts as replay
import audit_dataset as audit
import prepare_resources

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--points', type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 10:
        raise ValueError('Stage 1 accepts 1 to 10 episodes; full replay is outside this command scope')
    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f'Use a new replay directory: {destination}')
    destination.mkdir(parents=True)
    prepare_resources.run(destination / 'source_check', offline=True)
    shutil.copy2(ROOT / 'resource_manifest.json', destination / 'resource_manifest.json')
    original_root = replay.ROOT
    # RAW remains the original, checksum-verified public source directory.
    replay.ROOT = destination
    old_argv = sys.argv
    try:
        sys.argv = ['reconstruct_contacts.py', '--max-episodes', str(args.episodes), '--points', str(args.points)]
        replay.main()
    finally:
        replay.ROOT = original_root
        sys.argv = old_argv
    subfolder = f'probe_{args.episodes}_episodes'
    report = json.loads((destination / f'reports/{subfolder}/replay_validation.json').read_text(encoding='utf-8'))
    source = pq.read_table(replay.RAW / 'data/chunk-000/file-000.parquet')
    metadata = pq.read_table(replay.RAW / 'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    actions = np.asarray(source['action'].to_pylist(), dtype=float)
    positions = np.asarray(source['observation.state'].to_pylist(), dtype=float)
    results, common_points = [], None
    for meta in metadata[:args.episodes]:
        episode = int(meta['episode_index'])
        result, starts, common_points = audit.audit_episode(
            destination / f'data/derived/{subfolder}/episode_{episode:06d}.npz',
            meta, actions, positions, report['configuration']['sigma'], common_points)
        result['eligible_window_starts'] = starts
        results.append(result)
    result = {'status': 'passed', 'episodes': args.episodes, 'points': args.points,
              'fresh_replay': True, 'read_old_derived_data': False,
              'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'replay_kernel_sha256': report['script_sha256'],
              'audit_kernel_sha256': audit.sha256(Path(audit.__file__)),
              'frames': sum(r['frames'] for r in results),
              'contact_events': sum(r['contact_point_events'] for r in results),
              'no_contact_intervals': sum(r['intervals_without_exterior_contacts'] for r in results),
              'mean_visible_mask_iou': report['visible_mask_iou_mean'],
              'maximum_pusher_error': report['agent_position_error_max'],
              'max_rigid_error': max(r['rigid_point_max_absolute_error'] for r in results),
              'max_label_error': max(r['soft_label_max_absolute_error'] for r in results),
              'per_episode': results,
              'interpretation': 'Simulator reconstruction and internal consistency; not independent contact ground truth.'}
    (destination / 'fresh_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_episode'}), flush=True)


if __name__ == '__main__':
    main()
