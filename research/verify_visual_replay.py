"""Compare the first visual-DP test episode with an independent new process.

The policy uses one seed-0 torch.Generator continuously within an evaluation.
Therefore only the FIRST episode can be repeated with a freshly seeded policy
without restoring a later generator state. This check adds no test samples.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--repeated', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--atol', type=float, default=1e-9)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    original_summary = json.loads((args.original / 'summary.json').read_text(encoding='utf-8'))
    repeated_summary = json.loads((args.repeated / 'summary.json').read_text(encoding='utf-8'))
    first = original_summary['episodes'][0]
    if repeated_summary['episodes'] != [first]:
        raise AssertionError('Repeat exactly the first episode; later episodes need saved RNG state')
    name = f'episode_{first:06d}.npz'
    a_path, b_path = args.original / name, args.repeated / name
    comparisons = {}
    with np.load(a_path) as a, np.load(b_path) as b:
        if set(a.files) != set(b.files):
            raise AssertionError('Different trajectory fields')
        for key in a.files:
            x, y = a[key], b[key]
            shape_equal = x.shape == y.shape
            dtype_equal = x.dtype == y.dtype
            exact = bool(shape_equal and np.array_equal(x, y))
            floating = np.issubdtype(x.dtype, np.floating)
            within_tolerance = bool(shape_equal and (np.allclose(x, y, atol=args.atol, rtol=0) if floating else exact))
            error = float(np.max(np.abs(x - y))) if floating and shape_equal and x.size else 0.0
            comparisons[key] = {'shape': list(x.shape), 'dtype': str(x.dtype),
                                'shape_equal': shape_equal, 'dtype_equal': dtype_equal,
                                'exact_equal': exact, 'within_tolerance': within_tolerance,
                                'max_abs_difference': error}
    same_checkpoint = original_summary['checkpoint_sha256'] == repeated_summary['checkpoint_sha256']
    same_evaluator = original_summary['evaluator_sha256'] == repeated_summary['evaluator_sha256']
    passed = same_checkpoint and same_evaluator and all(v['dtype_equal'] and v['within_tolerance'] for v in comparisons.values())
    report = {'passed': passed, 'episode_index': first, 'new_statistical_samples': 0,
              'original_directory': str(args.original.resolve()), 'repeated_directory': str(args.repeated.resolve()),
              'original_npz_sha256': sha256(a_path), 'repeated_npz_sha256': sha256(b_path),
              'original_summary_sha256': sha256(args.original / 'summary.json'),
              'repeated_summary_sha256': sha256(args.repeated / 'summary.json'),
              'checkpoint_sha256': original_summary['checkpoint_sha256'],
              'same_checkpoint': same_checkpoint, 'same_evaluator': same_evaluator,
              'seed': 0, 'generator_policy': 'One continuous seed-0 generator for an evaluation; first episode repeated from a fresh process.',
              'float_absolute_tolerance': args.atol, 'float_relative_tolerance': 0,
              'array_comparisons': comparisons,
              'all_arrays_exact_equal': all(v['exact_equal'] for v in comparisons.values()),
              'note': 'File SHA may differ because NPZ zip metadata contains timestamps; array comparisons determine numerical reproducibility.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
