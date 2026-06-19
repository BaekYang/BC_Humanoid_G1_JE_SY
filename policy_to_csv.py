"""
policy_to_csv.py — 학습된 BC 정책을 sim에서 롤아웃해 '왼손 궤적'을 ik_traj.py용 CSV로 저장.

CSV 형식 (ik_traj.py가 읽는 그대로):
  [timestamp, LH_x,LH_y,LH_z, LH_R,LH_P,LH_Y, RH_x,RH_y,RH_z, RH_R,RH_P,RH_Y]
  - RPY는 라디안 (ik_traj.py가 R.from_euler('xyz', ...) 라디안으로 읽음)
  - 오른손은 시작 자세로 고정, 왼손만 정책 궤적을 따름
*** 이 스크립트 자체는 순수 시뮬레이션. 실물 명령 없음. ***

실행:
    python3 policy_to_csv.py                       # 기본 블록/타겟으로 arm_traj.csv 생성
    python3 policy_to_csv.py --bx 0.42 --by 0.20 --tx 0.42 --ty 0.08
    python3 policy_to_csv.py --view                # 뽑으면서 sim으로 확인
"""

import csv
import time
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
N_WAYPOINTS = 8
MAX_STEPS = 1500
WP_THRESH = 0.05
DWELL_MIN = 15
GRASP_GATE = 0.08


def above(p): return p + np.array([0, 0, APPROACH_H])
def on(p):    return p + np.array([0, 0, GRASP_OFF])
def quat_wxyz(m): q = R.from_matrix(m).as_quat(); return np.array([q[3], q[0], q[1], q[2]])
def wxyz_to_rpy(q): return R.from_quat([q[1], q[2], q[3], q[0]]).as_euler('xyz')  # radians


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bx", type=float, default=0.42)
    ap.add_argument("--by", type=float, default=0.20)
    ap.add_argument("--tx", type=float, default=0.42)
    ap.add_argument("--ty", type=float, default=0.08)
    ap.add_argument("--size", type=float, default=0.025)
    ap.add_argument("--out", type=str, default="arm_traj.csv")
    ap.add_argument("--repeat", type=int, default=2, help="각 점을 N번 반복 = 실물 재생속도 늦춤(안전)")
    ap.add_argument("--slowmo", type=float, default=5.0, help="뷰어 보기 속도만 늦춤(저장 CSV엔 영향 없음). 클수록 느림")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    stats = dict(np.load("bc_stats.npz"))
    pol = Policy(hidden=int(stats["hidden"]))
    pol.load_state_dict(torch.load("bc_policy.pt", map_location="cpu"))
    pol.eval()

    model = mujoco.MjModel.from_xml_path(XML)
    configuration = mink.Configuration(model)
    data = configuration.data
    model.geom_size[model.geom("block_geom").id] = [args.size] * 3

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    q0 = model.key_qpos[key].copy()
    binit = np.array([args.bx, args.by, TABLE_TOP_Z + args.size])
    q0[BLOCK_QADR:BLOCK_QADR + 3] = binit
    q0[BLOCK_QADR + 3:BLOCK_QADR + 7] = [1, 0, 0, 0]
    configuration.update(q0)

    tasks, limits, hand_tasks, hands_mid = build_mink(model, configuration, data)
    L = 1
    left_quat = data.mocap_quat[hands_mid[L]].copy()
    left_rpy = wxyz_to_rpy(left_quat)

    block_bid = model.body("block").id
    palm_sid = model.site("left_palm").id
    rpalm_sid = model.site("right_palm").id
    target_mid = model.body("target_marker").mocapid[0]
    target_pos = np.array([args.tx, args.ty, TABLE_TOP_Z + args.size])
    data.mocap_pos[target_mid] = target_pos
    mujoco.mj_forward(model, data)

    # 오른손 고정 자세 (시작 시점 그대로)
    rh_pos = data.site_xpos[rpalm_sid].copy()
    rh_rpy = R.from_matrix(data.site_xmat[rpalm_sid].reshape(3, 3)).as_euler('xyz')

    goals = [above(binit), on(binit), on(binit), above(binit),
             above(target_pos), on(target_pos), on(target_pos), above(target_pos)]
    om, osd = stats["obs_mean"], stats["obs_std"]
    am, asd = stats["act_mean"], stats["act_std"]

    rate = RateLimiter(frequency=100.0)
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) if args.view else None

    traj = [data.site_xpos[palm_sid].copy()]   # 시작점 = 현재 휴식 자세 (실물 시작 점프 방지)
    wp_idx = 0; dwell = 0; grasped = False; final_dwell = 0
    for _ in range(MAX_STEPS):
        if args.view and not viewer.is_running():
            break
        palm = data.site_xpos[palm_sid].copy()
        bpos = data.xpos[block_bid].copy()
        obs = np.concatenate([palm, bpos, target_pos, binit, [args.size]]).astype(np.float32)
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
        traj.append(new_palm.copy())

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
        else:
            if np.linalg.norm(new_palm - goals[-1]) < WP_THRESH:
                final_dwell += 1
                if final_dwell >= DWELL_MIN:
                    break

        if args.view:
            viewer.sync(); time.sleep(args.slowmo / 100.0)

    if viewer is not None:
        viewer.close()

    # CSV 작성 (ik_traj.py 형식). repeat로 점을 늘려 실물 재생속도 늦춤.
    header = ['timestamp',
              'LH_x', 'LH_y', 'LH_z', 'LH_R', 'LH_P', 'LH_Y',
              'RH_x', 'RH_y', 'RH_z', 'RH_R', 'RH_P', 'RH_Y']
    dt = 1.0 / 200.0
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        t = 0.0
        for p in traj:
            for _ in range(max(1, args.repeat)):
                w.writerow([t,
                            p[0], p[1], p[2], left_rpy[0], left_rpy[1], left_rpy[2],
                            rh_pos[0], rh_pos[1], rh_pos[2], rh_rpy[0], rh_rpy[1], rh_rpy[2]])
                t += dt
    n_rows = len(traj) * max(1, args.repeat)
    print(f"[V] 저장: {args.out}  ({n_rows} 프레임, 약 {n_rows/200:.1f}초 재생)")
    print(f"    왼손 시작 {np.round(traj[0],3)}  →  끝 {np.round(traj[-1],3)}")


if __name__ == "__main__":
    main()
