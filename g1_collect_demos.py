"""
g1_collect_demos.py — 왼손 pick-and-place 데모를 대량 수집 (BC 학습용 데이터셋).

검증된 워크스페이스 안에서 블록/타겟을 랜덤화 → scripted mink expert 실행 →
성공한 에피소드만 demos/demo_XXXX.npz 로 저장.
*** 순수 시뮬. 실물 로봇 명령 없음. ***

실행:
    python3 g1_collect_demos.py --n 100          # 100개 수집
    python3 g1_collect_demos.py --n 5 --view     # 5개를 눈으로 보며 (점검용)
"""

import os
import sys
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import mink
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter

import logging
logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)

XML = "/home/computer/mink/examples/unitree_g1/scene_g1_pickplace.xml"


class ViewerClosed(Exception):
    """뷰어 창을 닫으면 수집을 깔끔히 중단하기 위한 신호."""
    pass

BLOCK_QADR = 50
BLOCK_SIZE = 0.025
APPROACH_H = 0.12
GRASP_OFF  = 0.02
STEPS_PER_WP = 80
SETTLE_STEPS = 50
TABLE_TOP_Z = 0.835
SUCCESS_DIST = 0.05

# 검증된 왼손 워크스페이스 (테이블 위, 왼쪽 가까운 영역)
WS_X = (0.38, 0.45)
WS_Y = (0.07, 0.23)
MIN_SEP = 0.08           # 블록-타겟 최소 거리 (의미있는 이동 보장)
BLOCK_SIZE_RANGE = (0.020, 0.035)   # 블록 half-extent 랜덤 범위 (크기 일반화)


def quat_wxyz(mat3):
    q = R.from_matrix(mat3).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def collect_one(model, block_xy, target_xy, block_size=BLOCK_SIZE, view=False):
    """한 에피소드 실행 → (기록 dict, 성공여부, block_init, target_pos)."""
    configuration = mink.Configuration(model)
    data = configuration.data

    # 블록 크기 적용 (geom size 변경)
    gid = model.geom("block_geom").id
    model.geom_size[gid] = [block_size, block_size, block_size]

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    q0 = model.key_qpos[key].copy()
    block_init = np.array([block_xy[0], block_xy[1], TABLE_TOP_Z + block_size])
    q0[BLOCK_QADR:BLOCK_QADR + 3] = block_init
    q0[BLOCK_QADR + 3:BLOCK_QADR + 7] = [1, 0, 0, 0]
    configuration.update(q0)

    feet = ["right_foot", "left_foot"]
    hands = ["right_palm", "left_palm"]
    tasks = [
        pelvis_task := mink.FrameTask("pelvis", "body", position_cost=0.0, orientation_cost=1.0, lm_damping=1.0),
        torso_task := mink.FrameTask("torso_link", "body", position_cost=0.0, orientation_cost=1.0, lm_damping=1.0),
        posture_task := mink.PostureTask(model, cost=1e-1),
        com_task := mink.ComTask(cost=10.0),
    ]
    feet_tasks = [mink.FrameTask(f, "site", position_cost=10.0, orientation_cost=1.0, lm_damping=1.0) for f in feet]
    # 왼손(추적 대상)은 정밀도 위해 높은 비용, 오른손은 현 위치 유지용 낮은 비용
    hand_tasks = [
        mink.FrameTask("right_palm", "site", position_cost=2.0,  orientation_cost=1.0, lm_damping=1.0),
        mink.FrameTask("left_palm",  "site", position_cost=20.0, orientation_cost=1.0, lm_damping=1.0),
    ]
    tasks.extend(feet_tasks + hand_tasks)
    limits = [mink.ConfigurationLimit(model)]

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{f}_target").mocapid[0] for f in feet]
    hands_mid = [model.body(f"{h}_target").mocapid[0] for h in hands]
    L = 1

    posture_task.set_target_from_configuration(configuration)
    pelvis_task.set_target_from_configuration(configuration)
    torso_task.set_target_from_configuration(configuration)
    for h, f in zip(hands, feet):
        mink.move_mocap_to_frame(model, data, f"{f}_target", f, "site")
        mink.move_mocap_to_frame(model, data, f"{h}_target", h, "site")
    data.mocap_pos[com_mid] = data.subtree_com[1]
    com_task.set_target(data.mocap_pos[com_mid])
    for i, ft in enumerate(feet_tasks):
        ft.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
    hand_tasks[0].set_target(mink.SE3.from_mocap_id(data, hands_mid[0]))

    left_quat = data.mocap_quat[hands_mid[L]].copy()
    left_pos = data.mocap_pos[hands_mid[L]].copy()

    block_bid = model.body("block").id
    palm_sid = model.site("left_palm").id
    target_mid = model.body("target_marker").mocapid[0]
    target_pos = np.array([target_xy[0], target_xy[1], TABLE_TOP_Z + block_size])
    data.mocap_pos[target_mid] = target_pos
    mujoco.mj_forward(model, data)
    block_pos = data.xpos[block_bid].copy()

    above = lambda p: np.array([p[0], p[1], p[2] + APPROACH_H])
    on    = lambda p: np.array([p[0], p[1], p[2] + GRASP_OFF])
    waypoints = [
        ("approach", above(block_pos),  0), ("descend", on(block_pos),    0),
        ("grasp",    on(block_pos),     1), ("lift",    above(block_pos), 1),
        ("move",     above(target_pos), 1), ("place",   on(target_pos),   1),
        ("release",  on(target_pos),    0), ("retreat", above(target_pos),0),
    ]

    rate = RateLimiter(frequency=100.0)
    rec = {k: [] for k in ["hand_pos", "hand_quat", "block_pos", "block_quat",
                           "target_pos", "block_size", "waypoint_idx", "waypoint_grip",
                           "act_hand_target", "act_grasp"]}
    state = {"grasped": False}
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) if view else None

    def step_once(cmd, grip, wi):
        if grip == 1:
            state["grasped"] = True
        data.mocap_pos[hands_mid[L]] = cmd
        data.mocap_quat[hands_mid[L]] = left_quat
        hand_tasks[L].set_target(mink.SE3.from_mocap_id(data, hands_mid[L]))
        vel = mink.solve_ik(configuration, tasks, rate.dt, "daqp", limits=limits)
        configuration.integrate_inplace(vel, rate.dt)
        palm_pos = data.site_xpos[palm_sid].copy()
        palm_q = quat_wxyz(data.site_xmat[palm_sid].reshape(3, 3))
        if state["grasped"]:
            q = configuration.q.copy()
            q[BLOCK_QADR:BLOCK_QADR + 3] = palm_pos
            q[BLOCK_QADR + 3:BLOCK_QADR + 7] = palm_q
            configuration.update(q)
        rec["hand_pos"].append(palm_pos)
        rec["hand_quat"].append(palm_q)
        rec["block_pos"].append(data.xpos[block_bid].copy())
        rec["block_quat"].append(data.xquat[block_bid].copy())
        rec["target_pos"].append(target_pos.copy())
        rec["block_size"].append(block_size)
        rec["waypoint_idx"].append(wi)
        rec["waypoint_grip"].append(grip)
        rec["act_hand_target"].append(cmd.copy())
        rec["act_grasp"].append(grip)
        if view:
            if not viewer.is_running():
                viewer.close()
                raise ViewerClosed()
            viewer.sync(); rate.sleep()

    cur = left_pos.copy()
    for wi, (name, wp_pos, grip) in enumerate(waypoints):
        if name == "release":
            state["grasped"] = False
        start = cur.copy()
        for s in range(STEPS_PER_WP):
            a = (s + 1) / STEPS_PER_WP
            step_once((1 - a) * start + a * wp_pos, grip, wi)
        for _ in range(SETTLE_STEPS):
            step_once(wp_pos, grip, wi)
        cur = wp_pos.copy()

    q = configuration.q.copy()
    q[BLOCK_QADR + 2] = TABLE_TOP_Z + block_size
    configuration.update(q)
    mujoco.mj_forward(model, data)
    dist = float(np.linalg.norm(data.xpos[block_bid].copy() - target_pos))
    if viewer is not None:
        viewer.close()
    return rec, dist < SUCCESS_DIST, block_init, target_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model = mujoco.MjModel.from_xml_path(XML)
    os.makedirs("demos", exist_ok=True)

    saved = tried = 0
    while saved < args.n:
        tried += 1
        bx, by = rng.uniform(*WS_X), rng.uniform(*WS_Y)
        tx, ty = rng.uniform(*WS_X), rng.uniform(*WS_Y)
        bsize = rng.uniform(*BLOCK_SIZE_RANGE)
        if np.hypot(bx - tx, by - ty) < MIN_SEP:
            continue
        rec, ok, binit, tpos = (None, False, None, None)
        try:
            rec, ok, binit, tpos = collect_one(model, (bx, by), (tx, ty), block_size=bsize, view=args.view)
        except ViewerClosed:
            print("뷰어를 닫아 수집을 중단합니다.")
            break
        if not ok:
            if tried > max(30, args.n * 4):
                print("[!] 실패율이 너무 높음 — 워크스페이스(WS_X/WS_Y)를 줄여야 할 수 있어.")
                break
            continue
        np.savez(f"demos/demo_{saved:04d}.npz",
                 **{k: np.asarray(v) for k, v in rec.items()},
                 block_init_xpos=binit, target_xpos=tpos, block_size_scalar=np.float64(bsize))
        saved += 1
        if saved % 10 == 0:
            print(f"  {saved}/{args.n} 저장 (시도 {tried}회)")

    print(f"[V] 완료: {saved}개 저장, 시도 {tried}회 "
          f"(성공률 {saved/max(tried,1)*100:.0f}%)")


if __name__ == "__main__":
    main()
