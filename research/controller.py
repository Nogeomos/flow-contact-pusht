"""Current-state-only PushT flow planner and independent contact controller.

The controller knows the simulator geometry/dynamics. This is a privileged
state-feedback baseline, not a vision-only policy or a claim about real robots.
Candidate rollouts happen in separate physics worlds; the real object is only
moved through the official pusher action API. No demonstration is read here.
"""
from pathlib import Path
import heapq
import sys
import math
import time
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from reconstruct_contacts import OUTLINE, boundary_points, ContactReplayEnv


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def transform(local, pose):
    return np.asarray(local) @ rotation(pose[2]).T + pose[:2]


def angle_difference(target, current):
    return (target - current + np.pi) % (2 * np.pi) - np.pi


def desired_flow(state, goal, horizon=8, points=64):
    """Rigid interpolation toward goal, bounded to 24 units / 0.24 rad in .8s.

    Translation is measured at the physical CoG, not the offset body origin.
    Returned cumulative displacements remain in the WORLD frame.
    """
    pose = np.asarray(state[2:5], dtype=float)
    goal = np.asarray(goal, dtype=float)
    local = boundary_points(points)
    cog_local = np.array([0., 45.])
    cog = transform(cog_local[None], pose)[0]
    goal_cog = transform(cog_local[None], goal)[0]
    delta = goal_cog - cog
    delta *= min(1., 24. / max(np.linalg.norm(delta), 1e-9))
    da = np.clip(angle_difference(goal[2], pose[2]), -.24, .24)
    current = transform(local, pose)
    future = []
    for fraction in np.arange(1, horizon + 1) / horizon:
        theta = pose[2] + fraction * da
        origin = cog + fraction * delta - rotation(theta) @ cog_local
        future.append(transform(local, np.r_[origin, theta]) - current)
    return np.asarray(future, dtype=np.float32)


def boundary_normals(n=64):
    edge = np.roll(OUTLINE, -1, axis=0) - OUTLINE
    lengths = np.linalg.norm(edge, axis=1)
    segment = np.searchsorted(np.cumsum(lengths), np.arange(n) * lengths.sum() / n,
                              side='right')
    tangent = edge[segment] / lengths[segment, None]
    return np.c_[-tangent[:, 1], tangent[:, 0]]


def collision_free_path(start, end, pose, clearance=18.):
    """Visibility graph around the T dilated by the pusher radius + margin."""
    obstacle = Polygon(transform(OUTLINE, pose)).buffer(clearance, join_style=2)
    interior = obstacle.buffer(-.05)
    start, end = np.asarray(start, float), np.asarray(end, float)
    if interior.contains(Point(end)):
        return None
    escape = []
    if obstacle.contains(Point(start)):
        nearest = np.asarray(nearest_points(Point(start), obstacle.boundary)[1].coords[0])
        direction = nearest - start
        if np.linalg.norm(direction) < 1e-9:
            return None
        escaped = nearest + direction / np.linalg.norm(direction) * .3
        escape = [escaped]
        start = escaped
    vertices = np.asarray(obstacle.exterior.coords[:-1])
    vertices = vertices[np.all((vertices >= 21) & (vertices <= 491), axis=1)]
    nodes = np.vstack([start, end, vertices])
    edges = [[] for _ in nodes]
    for i in range(len(nodes)):
        for j in range(i):
            if not LineString([nodes[i], nodes[j]]).intersects(interior):
                distance = float(np.linalg.norm(nodes[i] - nodes[j]))
                edges[i].append((j, distance))
                edges[j].append((i, distance))
    queue, distances, previous = [(0., 0)], {0: 0.}, {}
    while queue:
        cost, i = heapq.heappop(queue)
        if i == 1:
            indices = [1]
            while indices[-1] != 0:
                indices.append(previous[indices[-1]])
            return escape + [nodes[k] for k in indices[::-1][1:]]
        if cost > distances.get(i, np.inf):
            continue
        for j, distance in edges[i]:
            new_cost = cost + distance
            if new_cost < distances.get(j, np.inf):
                distances[j], previous[j] = new_cost, i
                heapq.heappush(queue, (new_cost, j))
    return None


def waypoint_actions(start, waypoints, speed=120.):
    result = []
    previous = np.asarray(start)
    for waypoint in waypoints:
        count = max(1, int(np.ceil(np.linalg.norm(waypoint - previous) / (speed / 10))))
        result.extend(previous + fraction * (waypoint - previous)
                      for fraction in np.arange(1, count + 1) / count)
        previous = waypoint
    return result


class PredictionEnv(ContactReplayEnv):
    """Skip costly reward geometry in private candidate rollouts only."""
    def step(self, action):
        self.contact_events = []
        self.substep = -1
        self.manifold_index = 0
        for _ in range(10):
            acceleration = self.k_p * (action - self.agent.position) - self.k_v * np.array(self.agent.velocity)
            self.agent.velocity += acceleration * self.dt
            self.space.step(self.dt)
        return self.get_obs()

    def copy_state(self, source):
        # Angle before position avoids the legacy reset's offset-origin rotation.
        # Collision warm-start caches are rebuilt, so this is an approximate
        # predictor even though it uses the same physical equations/parameters.
        self.reset(options={'reset_to_state': [100., 100., 256., 256., 0.]})
        self.agent.position = tuple(source.agent.position)
        self.agent.velocity = tuple(source.agent.velocity)
        self.block.angle = source.block.angle
        self.block.position = tuple(source.block.position)
        self.block.velocity = tuple(source.block.velocity)
        self.block.angular_velocity = source.block.angular_velocity
        self.space.reindex_shapes_for_body(self.agent)
        self.space.reindex_shapes_for_body(self.block)


class ContactController:
    """Contact scoring -> obstacle-aware approach -> short physics-tested push.

    score_fn(state5, cumulative_world_flow[8,N,2], world_points[N,2])->scores[N].
    Both learned and rule selectors use identical candidate count, simulation
    model, path speed, and action budget. Model scores select 8 spatially diverse
    candidate contacts; physics picks the executable path among those contacts.
    """
    def __init__(self, env, score_fn=None, points=64, candidate_count=8):
        self.env, self.score_fn = env, score_fn
        self.local = boundary_points(points)
        self.local_normals = boundary_normals(points)
        self.candidate_count = candidate_count
        self.queue = []
        self.plans = []
        self.prediction = PredictionEnv()
        self.last_contact = None
        self.rollout_count = 0
        self.rollout_physics_substeps = 0
        self.rollout_wall_time_seconds = 0.

    def rule_scores(self, state, flow, world_points):
        normals = self.local_normals @ rotation(state[4]).T
        cog = transform(np.array([[0., 45.]]), state[2:])[0]
        arms = world_points - cog
        torque = arms[:, 0] * normals[:, 1] - arms[:, 1] * normals[:, 0]
        twist = np.c_[-arms[:, 1], arms[:, 0]]
        responses = normals[:, None, :] + torque[:, None, None] / 3000. * twist[None]
        scores = (responses * flow[-1][None]).sum((1, 2))
        scores /= np.sqrt((responses ** 2).sum((1, 2))) + 1e-9
        return scores

    def plan(self, remaining_steps=300):
        state = self.env.get_obs().copy()
        points = transform(self.local, state[2:])
        flow = desired_flow(state, self.env.goal_pose, points=len(self.local))
        scores = (self.rule_scores(state, flow, points) if self.score_fn is None
                  else np.asarray(self.score_fn(state.copy(), flow.copy(), points.copy()), float))
        if scores.shape != (len(self.local),) or not np.all(np.isfinite(scores)):
            raise ValueError('Contact scorer must return finite scores for every material point')
        normals = self.local_normals @ rotation(state[4]).T
        chosen, candidates = [], []
        for index in np.argsort(-scores, kind='stable'):
            if any(np.linalg.norm(self.local[index] - self.local[old]) < 20. for old in chosen):
                continue
            before = points[index] - normals[index] * 19.
            if np.any(before < 21) or np.any(before > 491):
                continue
            path = collision_free_path(state[:2], before, state[2:])
            if path is None:
                continue
            chosen.append(int(index))
            approach = waypoint_actions(state[:2], path)
            approach += [before.copy()] * 3
            # Long pushes move toward distant goals; short pushes permit alignment.
            for distance in (12., 40.):
                pushing = [before + normals[index] * distance * f for f in np.arange(1, 9) / 8]
                actions = approach + pushing + [pushing[-1].copy()] * 2
                actions = [np.clip(a, 21., 491.) for a in actions[:remaining_steps]]
                if len(actions) <= len(approach):
                    continue
                prediction_start = time.perf_counter()
                self.prediction.copy_state(self.env)
                for action in actions:
                    self.prediction.step(action)
                self.rollout_count += 1
                # Includes the single reset step performed by the upstream env.
                self.rollout_physics_substeps += 1 + 10 * len(actions)
                self.rollout_wall_time_seconds += time.perf_counter() - prediction_start
                predicted = self.prediction.get_obs()
                predicted_points = transform(self.local, predicted[2:])
                goal_points = transform(self.local, self.env.goal_pose)
                error = float(np.sqrt(((predicted_points - goal_points) ** 2).sum(1).mean()))
                # Small travel cost discourages repeatedly orbiting the object.
                objective = error + .035 * len(approach)
                candidates.append((objective, index, actions, predicted, distance))
            if len(chosen) >= self.candidate_count:
                break
        if not candidates:
            self.queue = [state[:2].copy()]
            self.last_contact = None
            self.plans.append({'kind': 'no_feasible_contact', 'state': state.tolist()})
            return
        objective, index, actions, predicted, distance = min(candidates, key=lambda x: x[0])
        self.queue = actions
        self.last_contact = int(index)
        self.plans.append({'kind': 'contact', 'contact_index': int(index), 'state': state.tolist(),
                           'desired_flow_world': flow.tolist(), 'scores': scores.tolist(),
                           'candidate_contacts': chosen, 'candidate_rollouts': len(candidates),
                           'cumulative_rollout_count': self.rollout_count,
                           'cumulative_rollout_physics_substeps': self.rollout_physics_substeps,
                           'cumulative_rollout_wall_time_seconds': self.rollout_wall_time_seconds,
                           'selected_push_distance': distance, 'planned_actions': len(actions),
                           'predicted_endpoint_state': predicted.tolist(), 'objective': objective})

    def next_action(self, remaining_steps=300):
        if not self.queue:
            self.plan(remaining_steps)
        return np.asarray(self.queue.pop(0), dtype=np.float64)

    def close(self):
        self.prediction.close()
