"""
eval_bc_obs.py — 장애물(기둥) 회피 BC 정책 롤아웃 + 성공률.

eval_bc.py 에 기둥(장애물)을 추가한 버전.
- obs 16D (기둥 위치 포함), 웨이포인트 9개 (detour 포함)
- 기둥 배치 규칙 + 충돌회피 제약을 collect(g1_collect_demos_obs.py)와 동일하게 (BC 일관성)
- 출력에 박스/목표/기둥 위치 + 기둥과의 최소거리(회피 확인) 표시
*** 순수 시뮬. 실물 로봇 명령 없음. ***

실행:
    python eval_bc_obs.py --n 1 --view
    python eval_bc_obs.py --n 50
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import mujoco
import mujoco.viewer
import mink
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter

import logging
logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)

XML = r"C:\Users\USER\g1bc\mink-main\examples\unitree_g1\scene_g1_pickplace_obs.xml"
BLOCK_QADR = 50
APPROACH_H = 0.12
GRASP_OFF = 0.02
TABLE_TOP_Z = 0.835
SUCCESS_DIST = 0.05
N_WAYPOINTS = 9

# 수집(g1_collect_demos_obs.py)과 동일하게 맞춰야 평가가 의미 있음
WS_X = (0.25, 0.55); WS_Y = (0.00, 0.40)
BLOCK_SIZE_RANGE = (0.020, 0.035)
ELLIPSE_CX = 0.40; ELLIPSE_CY = 0.16; ELLIPSE_A = 0.12; ELLIPSE_B = 0.17

OBS_R = 0.03; OBS_Z = TABLE_TOP_Z + 0.10
MIN_SEP_OBS = 0.14; OBS_JITTER = 0.02; CLEAR = 0.06


def in_reach(x, y):
    return ((x - ELLIPSE_CX) / ELLIPSE_A) ** 2 + ((y - ELLIPSE_CY) / ELLIPSE_B) ** 2 < 1.0


def detour_point(box_xy, target_xy, obs_xy):
    box = np.array(box_xy, float); tgt = np.array(target_xy, float); obs = np.array(obs_xy, float)
    d = tgt - box; n = np.linalg.norm(d)
    if n < 1e-6:
        return None
    d /= n
    perp = np.array([-d[1], d[0]])
    off = OBS_R + CLEAR
    cands = [obs + perp * off, obs - perp * off]
    SHOULDER_XY = np.array([0.0, 0.10])
    cands.sort(key=lambda c: np.linalg.norm(c - SHOULDER_XY))
    for c in cands:
        if in_reach(c[0], c[1]):
            return (float(c[0]), float(c[1]))
    return None

MAX_STEPS = 1700
WP_THRESH = 0.05
DWELL_MIN = 15
GRASP_GATE = 0.08


def above(p): return p + np.array([0, 0, APPROACH_H])
def on(p):    return p + np.array([0, 0, GRASP_OFF])
def quat_wxyz(m): q = R.from_matrix(m).as_quat(); return np.array([q[3], q[0], q[1], q[2]])


class ResBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
    def forward(self, x): return x + self.net(x)

class Policy(nn.Module):
    def __init__(self, obs_dim=16, act_dim=4, hidden=256, n_wp=N_WAYPOINTS, emb=16):
        super().__init__()
        self.emb = nn.Embedding(n_wp, emb)
        self.inp = nn.Sequential(nn.Linear(obs_dim + emb, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.blocks = nn.Sequential(ResBlock(hidden), ResBlock(hidden), ResBlock(hidden))
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, act_dim))
    def forward(self, obs, wp):
        return self.out(self.blocks(self.inp(torch.cat([obs, self.emb(wp)], dim=-1))))


def build_mink(model, configuration, data):
    feet = ["right_foot", "left_foot"]; hands = ["right_palm", "left_palm"]
    tasks = [
        pelvis := mink.FrameTask("pelvis", "body", position_cost=0.0, orientation_cost=1.0, lm_damping=1.0),
        torso := mink.FrameTask("torso_link", "body", position_cost=0.0, orientation_cost=1.0, lm_damping=1.0),
        posture := mink.PostureTask(model, cost=1e-1),
        com := mink.ComTask(cost=10.0),
    ]
    feet_tasks = [mink.FrameTask(f, "site", position_cost=10.0, orientation_cost=1.0, lm_damping=1.0) for f in feet]
    hand_tasks = [
        mink.FrameTask("right_palm", "site", position_cost=2.0,  orientation_cost=1.0, lm_damping=1.0),
        mink.FrameTask("left_palm",  "site", position_cost=20.0, orientation_cost=1.0, lm_damping=1.0),
    ]
    tasks.extend(feet_tasks + hand_tasks)
    limits = [mink.ConfigurationLimit(model)]
    # 충돌 회피: collect와 동일 (BC 일관성)
    limits.append(mink.CollisionAvoidanceLimit(
        model,
        geom_pairs=[(["left_hand_collision"], ["obstacle_geom"])],
        minimum_distance_from_collisions=0.02,
        collision_detection_distance=0.10,
    ))
    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{f}_target").mocapid[0] for f in feet]
    hands_mid = [model.body(f"{h}_target").mocapid[0] for h in hands]
    posture.set_target_from_configuration(configuration)
    pelvis.set_target_from_configuration(configuration)
    torso.set_target_from_configuration(configuration)
    for h, f in zip(hands, feet):
        mink.move_mocap_to_frame(model, data, f"{f}_target", f, "site")
        mink.move_mocap_to_frame(model, data, f"{h}_target", h, "site")
    data.mocap_pos[com_mid] = data.subtree_com[1]
    com.set_target(data.mocap_pos[com_mid])
    for i, ft in enumerate(feet_tasks):
        ft.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
    hand_tasks[0].set_target(mink.SE3.from_mocap_id(data, hands_mid[0]))
    return tasks, limits, hand_tasks, hands_mid


def rollout(model, pol, stats, block_xy, target_xy, obstacle_xy, detour_xy, block_size, view=False):
    configuration = mink.Configuration(model)
    data = configuration.data
    gid = model.geom("block_geom").id
    model.geom_size[gid] = [block_size, block_size, block_size]

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    q0 = model.key_qpos[key].copy()
    binit = np.array([block_xy[0], block_xy[1], TABLE_TOP_Z + block_size])
    q0[BLOCK_QADR:BLOCK_QADR + 3] = binit
    q0[BLOCK_QADR + 3:BLOCK_QADR + 7] = [1, 0, 0, 0]
    configuration.update(q0)

    tasks, limits, hand_tasks, hands_mid = build_mink(model, configuration, data)
    L = 1
    left_quat = data.mocap_quat[hands_mid[L]].copy()

    block_bid = model.body("block").id
    palm_sid = model.site("left_palm").id
    target_mid = model.body("target_marker").mocapid[0]
    obstacle_mid = model.body("obstacle").mocapid[0]
    target_pos = np.array([target_xy[0], target_xy[1], TABLE_TOP_Z + block_size])
    data.mocap_pos[target_mid] = target_pos
    obstacle_pos = np.array([obstacle_xy[0], obstacle_xy[1], OBS_Z])
    data.mocap_pos[obstacle_mid] = obstacle_pos
    mujoco.mj_forward(model, data)

    carry_z = above(binit)[2]
    detour_goal = np.array([detour_xy[0], detour_xy[1], carry_z])
    # 9개 웨이포인트 목표 (collect/train과 동일 순서)
    goals = [above(binit), on(binit), on(binit), above(binit),
             detour_goal,
             above(target_pos), on(target_pos), on(target_pos), above(target_pos)]

    om, osd = stats["obs_mean"], stats["obs_std"]
    am, asd = stats["act_mean"], stats["act_std"]

    rate = RateLimiter(frequency=30.0)
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) if view else None

    wp_idx = 0; dwell = 0; grasped = False
    min_clear = 1e9                      # 운반 중 손-기둥 최소 xy거리 (회피 확인)
    for _ in range(MAX_STEPS):
        if view and not viewer.is_running():
            break
        palm = data.site_xpos[palm_sid].copy()
        bpos = data.xpos[block_bid].copy()
        # obs 16D: 손3 블록3 목표3 박스초기3 크기1 기둥3
        obs = np.concatenate([palm, bpos, target_pos, binit, [block_size], obstacle_pos]).astype(np.float32)
        obs_n = ((obs - om) / osd).astype(np.float32)
        with torch.no_grad():
            pred = pol(torch.from_numpy(obs_n)[None], torch.tensor([wp_idx]))[0].numpy()
        pred = pred * asd + am
        goal = pred[:3]; grasp_pred = float(pred[3])

        data.mocap_pos[hands_mid[L]] = goal
        data.mocap_quat[hands_mid[L]] = left_quat
        data.mocap_pos[obstacle_mid] = obstacle_pos
        hand_tasks[L].set_target(mink.SE3.from_mocap_id(data, hands_mid[L]))
        vel = mink.solve_ik(configuration, tasks, rate.dt, "daqp", limits=limits)
        configuration.integrate_inplace(vel, rate.dt)

        new_palm = data.site_xpos[palm_sid].copy()
        # 운반 단계(grasp~place)에서 손-기둥 xy 최소거리 기록
        if grasped:
            cd = np.linalg.norm(new_palm[:2] - obstacle_pos[:2])
            min_clear = min(min_clear, cd)
        if grasp_pred > 0.5 and (grasped or np.linalg.norm(new_palm - data.xpos[block_bid]) < GRASP_GATE):
            grasped = True
        if grasp_pred <= 0.5:
            grasped = False
        if grasped:
            q = configuration.q.copy()
            q[BLOCK_QADR:BLOCK_QADR + 3] = new_palm
            q[BLOCK_QADR + 3:BLOCK_QADR + 7] = quat_wxyz(data.site_xmat[palm_sid].reshape(3, 3))
            configuration.update(q)

        if wp_idx < N_WAYPOINTS - 1:
            dwell = dwell + 1 if np.linalg.norm(new_palm - goals[wp_idx]) < WP_THRESH else 0
            if dwell >= DWELL_MIN:
                wp_idx += 1; dwell = 0

        if view:
            viewer.sync(); rate.sleep()

    q = configuration.q.copy()
    q[BLOCK_QADR + 2] = TABLE_TOP_Z + block_size
    configuration.update(q)
    mujoco.mj_forward(model, data)
    dist = float(np.linalg.norm(data.xpos[block_bid].copy() - target_pos))
    if viewer is not None:
        viewer.close()
    return dist < SUCCESS_DIST, dist, (min_clear if min_clear < 1e8 else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    stats = dict(np.load("bc_stats_obs.npz"))
    pol = Policy(hidden=int(stats["hidden"]))
    pol.load_state_dict(torch.load("bc_policy_obs.pt", map_location="cpu"))
    pol.eval()

    model = mujoco.MjModel.from_xml_path(XML)
    rng = np.random.default_rng(args.seed)

    succ = done = 0
    while done < args.n:
        # 배치 규칙 collect와 동일
        if rng.random() < 0.5:
            by = rng.uniform(WS_Y[0], ELLIPSE_CY); ty = rng.uniform(ELLIPSE_CY, WS_Y[1])
        else:
            by = rng.uniform(ELLIPSE_CY, WS_Y[1]); ty = rng.uniform(WS_Y[0], ELLIPSE_CY)
        bx, tx = rng.uniform(*WS_X), rng.uniform(*WS_X)
        bsize = rng.uniform(*BLOCK_SIZE_RANGE)
        if np.hypot(bx - tx, by - ty) < MIN_SEP_OBS:
            continue
        if not (in_reach(bx, by) and in_reach(tx, ty)):
            continue
        ox = (bx + tx) / 2 + rng.uniform(-OBS_JITTER, OBS_JITTER)
        oy = (by + ty) / 2 + rng.uniform(-OBS_JITTER, OBS_JITTER)
        if not in_reach(ox, oy):
            continue
        detour = detour_point((bx, by), (tx, ty), (ox, oy))
        if detour is None:
            continue
        ok, dist, clr = rollout(model, pol, stats, (bx, by), (tx, ty), (ox, oy), detour, bsize, view=args.view)
        done += 1; succ += int(ok)
        print(f"  [{done:3d}] {'성공' if ok else '실패'}  "
              f"박스({bx:.2f},{by:.2f}) → 목표({tx:.2f},{ty:.2f})  기둥({ox:.2f},{oy:.2f})  "
              f"| 블록-타겟 {dist:.3f}m  손-기둥최소 {clr:.3f}m")

    print(f"\n[V] 성공률: {succ}/{done} = {succ/done*100:.1f}%")
    print(f"    (손-기둥최소: 기둥반경 {OBS_R}m보다 충분히 크면 회피 성공. 작으면 통과한 것)")


if __name__ == "__main__":
    main()