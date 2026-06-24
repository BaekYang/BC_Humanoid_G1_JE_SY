"""
eval_bc.py (v2) — 학습된 BC 정책으로 pick-and-place 롤아웃 + 성공률.

정책 출력 = '목표 손위치(절대)' + grasp.  매 스텝 그 목표를 mink IK에 넣어 손을 보냄.
단계(waypoint)는 손이 그 단계 목표에 가까워지면 자동 진행 (정책엔 단계 인덱스만 제공).
*** 순수 시뮬. 실물 로봇 명령 없음. ***

실행:
    python3 eval_bc.py --n 1 --view
    python3 eval_bc.py --n 50
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

XML = "/home/computer/mink/examples/unitree_g1/scene_g1_pickplace.xml"
BLOCK_QADR = 50
APPROACH_H = 0.12
GRASP_OFF = 0.02
TABLE_TOP_Z = 0.835
SUCCESS_DIST = 0.05
N_WAYPOINTS = 8

WS_X = (0.38, 0.45); WS_Y = (0.07, 0.23); MIN_SEP = 0.08
BLOCK_SIZE_RANGE = (0.020, 0.035)

MAX_STEPS = 1500
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
    def __init__(self, obs_dim=13, act_dim=4, hidden=256, n_wp=N_WAYPOINTS, emb=16):
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


def rollout(model, pol, stats, block_xy, target_xy, block_size, view=False):
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
    target_pos = np.array([target_xy[0], target_xy[1], TABLE_TOP_Z + block_size])
    data.mocap_pos[target_mid] = target_pos
    mujoco.mj_forward(model, data)

    goals = [above(binit), on(binit), on(binit), above(binit),
             above(target_pos), on(target_pos), on(target_pos), above(target_pos)]

    om, osd = stats["obs_mean"], stats["obs_std"]
    am, asd = stats["act_mean"], stats["act_std"]

    rate = RateLimiter(frequency=100.0)
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) if view else None

    wp_idx = 0; dwell = 0; grasped = False
    for _ in range(MAX_STEPS):
        if view and not viewer.is_running():
            break
        palm = data.site_xpos[palm_sid].copy()
        bpos = data.xpos[block_bid].copy()
        obs = np.concatenate([palm, bpos, target_pos, binit, [block_size]]).astype(np.float32)
        obs_n = ((obs - om) / osd).astype(np.float32)
        with torch.no_grad():
            pred = pol(torch.from_numpy(obs_n)[None], torch.tensor([wp_idx]))[0].numpy()
        pred = pred * asd + am
        goal = pred[:3]; grasp_pred = float(pred[3])

        data.mocap_pos[hands_mid[L]] = goal
        data.mocap_quat[hands_mid[L]] = left_quat
        hand_tasks[L].set_target(mink.SE3.from_mocap_id(data, hands_mid[L]))
        vel = mink.solve_ik(configuration, tasks, rate.dt, "daqp", limits=limits)
        configuration.integrate_inplace(vel, rate.dt)

        new_palm = data.site_xpos[palm_sid].copy()
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
    return dist < SUCCESS_DIST, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    stats = dict(np.load("bc_stats.npz"))
    pol = Policy(hidden=int(stats["hidden"]))
    pol.load_state_dict(torch.load("bc_policy.pt", map_location="cpu"))
    pol.eval()

    model = mujoco.MjModel.from_xml_path(XML)
    rng = np.random.default_rng(args.seed)

    succ = done = 0
    while done < args.n:
        bx, by = rng.uniform(*WS_X), rng.uniform(*WS_Y)
        tx, ty = rng.uniform(*WS_X), rng.uniform(*WS_Y)
        bsize = rng.uniform(*BLOCK_SIZE_RANGE)
        if np.hypot(bx - tx, by - ty) < MIN_SEP:
            continue
        ok, dist = rollout(model, pol, stats, (bx, by), (tx, ty), bsize, view=args.view)
        done += 1; succ += int(ok)
        print(f"  [{done:3d}] {'성공' if ok else '실패'}  (블록-타겟 {dist:.3f} m)")

    print(f"\n[V] 성공률: {succ}/{done} = {succ/done*100:.1f}%")


if __name__ == "__main__":
    main()
