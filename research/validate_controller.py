"""Focused physical/isolation checks; writes measurements instead of assuming clones exact."""
from pathlib import Path
import argparse
import json
import sys
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.controller import (ContactReplayEnv, PredictionEnv, ContactController,
                                 desired_flow, transform, boundary_points)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    env = ContactReplayEnv()
    env.reset(options={'reset_to_state': [180, 215, 260, 200, 0]})
    prediction = PredictionEnv()
    comparisons = []
    for step in range(15):
        before = env.get_obs().copy()
        prediction.copy_state(env)
        predicted = prediction.step(np.array([220., 215.]))
        assert np.array_equal(before, env.get_obs()), 'Private candidate rollout mutated the real environment'
        actual = env.step(np.array([220., 215.]))[0]
        error = np.abs(actual-predicted)
        comparisons.append({'step': step, 'contact_events': len(env.contact_events),
                            'max_xy_error': float(error[:4].max()), 'angle_error_rad': float(error[4])})
    # Exact body state is copied, but private worlds rebuild contact caches. The
    # measured nonzero contact error is expected; reject material (>0.1 unit) drift.
    assert max(x['max_xy_error'] for x in comparisons) < .1
    assert max(x['angle_error_rad'] for x in comparisons) < .01
    env.reset(options={'reset_to_state': [100, 100, 260, 200, 0]})
    prediction.copy_state(env)
    predicted = prediction.step(np.array([110., 100.]))
    actual = env.step(np.array([110., 100.]))[0]
    free_error = float(np.max(np.abs(actual-predicted)))
    assert free_error < 1e-10
    state = env.get_obs().copy()
    current_goal_flow = desired_flow(state, state[2:])
    assert np.max(np.abs(current_goal_flow)) < 1e-9
    flow = desired_flow(state, env.goal_pose)
    assert flow.shape == (8, 64, 2) and np.all(np.isfinite(flow))
    selections = []
    for high in (8, 16):
        def scorer(state, flow, points, high=high):
            scores = np.zeros(64)
            scores[high] = 100.
            return scores
        controller = ContactController(env, score_fn=scorer, candidate_count=1)
        before = env.get_obs().copy()
        with patch('numpy.load', side_effect=AssertionError('Must not read trajectory NPZ')):
            with patch('pyarrow.parquet.read_table', side_effect=AssertionError('Must not read future Parquet')):
                controller.plan(300)
        assert np.array_equal(before, env.get_obs())
        assert controller.last_contact == high
        selections.append({'highest_score_index': high, 'selected_index': controller.last_contact,
                           'candidate_contacts': controller.plans[-1]['candidate_contacts']})
        controller.close()
    result = {'passed': True, 'non_contact_clone_max_state_error': free_error,
              'contact_clone_comparisons': comparisons,
              'contact_clone_max_xy_error': max(x['max_xy_error'] for x in comparisons),
              'contact_clone_max_angle_error_rad': max(x['angle_error_rad'] for x in comparisons),
              'clone_limitation': 'Body state/velocities are copied; persistent collision arbiter warm-start caches are rebuilt. Candidate predictions are approximate.',
              'real_environment_unchanged_during_private_rollouts': True,
              'same_current_pose_as_goal_gives_zero_flow': True,
              'contact_scores_change_selected_candidates': selections,
              'npz_and_parquet_readers_blocked_during_planning': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    prediction.close()
    env.close()


if __name__ == '__main__':
    main()
