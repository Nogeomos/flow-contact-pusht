"""Fresh-process, equal-action-budget closed-loop evaluation.

Only initial-state seeds come from held-out episode IDs. Demonstration actions,
future states, contact labels, and future video are never opened by this file.
"""
from pathlib import Path
import argparse
import importlib
import json
import sys
import time
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from reconstruct_contacts import ContactReplayEnv, initial_state, boundary_points
from research.controller import ContactController, desired_flow, transform, angle_difference


def confidence_interval(successes, total):
    """Wilson 95% interval; experimental unit is an episode, never a window."""
    if not total:
        return [0., 1.]
    z, p = 1.95996398454, successes / total
    center = (p + z*z / (2*total)) / (1 + z*z / total)
    radius = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1+z*z/total)
    return [float(center-radius), float(center+radius)]


def render_panel(rgb, episode, method, step, coverage, selected=None):
    panel = Image.new('RGB', (640, 574), '#f2f5fa')
    panel.paste(Image.fromarray(rgb).resize((512, 512)), (64, 48))
    d = ImageDraw.Draw(panel)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
    d.text((16, 8), f'{method} | episode {episode} | t={step/10:.1f}s | coverage={coverage:.3f}',
           fill='#153047', font=font)
    return panel


def run_episode(episode, method, predictor=None, max_steps=300, save_frames=False):
    env = ContactReplayEnv()
    env.reset(options={'reset_to_state': initial_state(episode)})
    controller = ContactController(env, score_fn=predictor) if method in ('rule', 'learned_contact') else None
    states = [env.get_obs().copy()]
    coverages = [float(env._get_coverage())]
    actions, contacts, selected_contacts, planner_flows = [], [], [], []
    images = [env._render()]
    agents = [states[0][:2].copy()]
    frames = [render_panel(images[-1], episode, method, 0, coverages[-1])] if save_frames else []
    action_queue = []
    elapsed_start = time.perf_counter()
    for step in range(max_steps):
        if coverages[-1] >= env.success_threshold:
            break
        state = env.get_obs().copy()
        flow = desired_flow(state, env.goal_pose)
        planner_flows.append(flow)
        if controller is not None:
            action = controller.next_action(max_steps-step)
            selected_contacts.append(-1 if controller.last_contact is None else controller.last_contact)
        elif method == 'flow_action':
            predicted = np.asarray(predictor(state, flow, transform(boundary_points(64), state[2:])))
            action = predicted[0] if predicted.ndim == 2 else predicted
            selected_contacts.append(-1)
        elif method == 'visual_dp':
            if not action_queue:
                image_history = np.stack((images[max(0, len(images)-2)], images[-1]))
                agent_history = np.stack((agents[max(0, len(agents)-2)], agents[-1]))
                proposed = np.asarray(predictor.predict(image_history, agent_history))
                if proposed.shape != (8, 2):
                    raise ValueError(f'Expected DP action chunk (8,2); got {proposed.shape}')
                action_queue = list(proposed)
            action = action_queue.pop(0)
            selected_contacts.append(-1)
        else:
            raise ValueError(method)
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError(f'Invalid pusher action: {action}')
        action = np.clip(action, 0., 512.)
        env.action_index = step
        state, _, _, _, info = env.step(action)
        states.append(state.copy())
        actions.append(action.copy())
        coverages.append(float(info['coverage']))
        contacts.append(len(env.contact_events))
        agents.append(state[:2].copy())
        images.append(env._render())
        if save_frames and (step % 3 == 0 or info['is_success']):
            frames.append(render_panel(images[-1], episode, method, step+1, coverages[-1]))
    elapsed = time.perf_counter() - elapsed_start
    goal_points = transform(boundary_points(64), env.goal_pose)
    final_points = transform(boundary_points(64), states[-1][2:])
    result = {'episode_index': int(episode), 'method': method, 'steps': len(actions),
              'success': bool(max(coverages) >= env.success_threshold),
              'native_success': bool(max(coverages) > env.success_threshold),
              'initial_coverage': coverages[0], 'final_coverage': coverages[-1],
              'max_coverage': max(coverages), 'wall_time_seconds': elapsed,
              'final_point_rmse': float(np.sqrt(((goal_points-final_points)**2).sum(1).mean())),
              'final_angle_error_deg': float(np.degrees(abs(angle_difference(env.goal_pose[2], states[-1][4])))),
              'contact_intervals': int(np.count_nonzero(contacts)),
              'contact_events': int(np.sum(contacts)), 'goal_pose': env.goal_pose.tolist(),
              'initial_state': states[0].tolist(), 'final_state': states[-1].tolist(),
              'candidate_rollouts': controller.rollout_count if controller is not None else 0,
              'candidate_physics_substeps': controller.rollout_physics_substeps if controller is not None else 0,
              'candidate_rollout_wall_time_seconds': controller.rollout_wall_time_seconds if controller is not None else 0.,
              'plans': controller.plans if controller is not None else []}
    arrays = {'episode_index': np.array(episode), 'states': np.asarray(states),
              'actions': np.asarray(actions).reshape(-1, 2), 'coverage': np.asarray(coverages),
              'contact_event_counts': np.asarray(contacts), 'selected_contact_indices': np.asarray(selected_contacts),
              'desired_flow_world': np.asarray(planner_flows), 'observations_rgb': np.asarray(images),
              'goal_pose': env.goal_pose.copy()}
    if controller is not None:
        controller.close()
    env.close()
    return result, arrays, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['rule','learned_contact','flow_action','visual_dp'], default='rule')
    parser.add_argument('--split', choices=['validation','test'], default='validation')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--episode-ids', type=int, nargs='+')
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--factory', help='module:function checkpoint loader (returns callable or DP policy)')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--videos', type=int, default=3)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'Preserve existing results: {args.output_dir}')
    splits_path = ROOT / 'data/derived/splits.json'
    splits = json.loads(splits_path.read_text())['splits']
    episodes = args.episode_ids if args.episode_ids else splits[args.split][:args.episodes]
    if not set(episodes).issubset(splits[args.split]):
        raise ValueError('Every requested seed must belong to the declared held-out split')
    predictor = None
    if args.method != 'rule':
        if not args.factory or not args.checkpoint:
            raise ValueError('Learned methods require checkpoint and explicit loader factory')
        module, function = args.factory.split(':')
        predictor = getattr(importlib.import_module(module), function)(args.checkpoint, device=args.device, seed=args.seed)
    args.output_dir.mkdir(parents=True)
    results = []
    for position, episode in enumerate(episodes):
        result, arrays, frames = run_episode(episode, args.method, predictor, args.max_steps,
                                             save_frames=position < args.videos)
        prefix = args.output_dir / f'episode_{episode:06d}'
        np.savez_compressed(prefix.with_suffix('.npz'), **arrays)
        prefix.with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        if frames:
            frames[0].save(prefix.with_suffix('.gif'), save_all=True, append_images=frames[1:],
                           duration=300, loop=0)
            frames[-1].save(prefix.with_suffix('.png'))
        results.append({k: v for k, v in result.items() if k != 'plans'})
        print(json.dumps(results[-1]), flush=True)
    successes = sum(r['success'] for r in results)
    summary = {'schema_version': 1, 'method': args.method, 'split': args.split,
               'episodes': episodes, 'max_steps_per_episode': args.max_steps,
               'success_criterion': 'coverage >= 0.95 at any executed state; native_success separately uses > 0.95',
               'successes': successes, 'count': len(results), 'success_rate': successes/len(results),
               'success_rate_wilson_95': confidence_interval(successes, len(results)),
               'mean_max_coverage': float(np.mean([r['max_coverage'] for r in results])),
               'mean_final_coverage': float(np.mean([r['final_coverage'] for r in results])),
               'mean_final_point_rmse': float(np.mean([r['final_point_rmse'] for r in results])),
               'mean_steps': float(np.mean([r['steps'] for r in results])),
               'total_wall_time_seconds': sum(r['wall_time_seconds'] for r in results),
               'total_candidate_rollouts': sum(r['candidate_rollouts'] for r in results),
               'total_candidate_physics_substeps': sum(r['candidate_physics_substeps'] for r in results),
               'total_candidate_rollout_wall_time_seconds': sum(r['candidate_rollout_wall_time_seconds'] for r in results),
               'compute_budget_note': 'Equal real action budget, not equal computation: contact controllers simulate candidates; direct policies do not.',
               'replanning_schedule': ('Replan after completing the queued approach, push and hold; executed states are observed every action.'
                                        if args.method in ('rule','learned_contact') else
                                        ('Every 8 actions' if args.method == 'visual_dp' else 'Every action')),
               'flow_record_note': 'NPZ desired_flow_world is recomputed from each observed state for diagnostics. Contact-score inputs are the frozen-at-replan flows stored in each JSON plan.',
               'checkpoint': str(args.checkpoint) if args.checkpoint else None,
               'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
               'split_sha256': hashlib.sha256(splits_path.read_bytes()).hexdigest(),
               'controller_sha256': hashlib.sha256((ROOT/'research/controller.py').read_bytes()).hexdigest(),
               'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'future_leakage_guard': 'No trajectory data loaded. Planner uses current state and fixed goal only.',
               'observations': ('two RGB images and pusher xy' if args.method=='visual_dp' else
                                'privileged simulator pose, fixed geometry, actual pusher xy; target-generated flow'),
               'controller_note': ('8 contact candidates x 2 push lengths, shared simulator-dynamics candidate model and visibility-graph paths; collision warm-start caches are rebuilt'
                                   if args.method in ('rule','learned_contact') else 'direct learned action policy'),
               'results': results}
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
