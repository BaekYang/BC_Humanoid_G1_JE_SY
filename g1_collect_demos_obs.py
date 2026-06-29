"""
g1_collect_demos_obs.py — 장애물(기둥) 회피 데모 수집 [1단계: 기둥 배치 검증용].

기존 g1_collect_demos.py 에 장애물(기둥) + 우회를 추가한 버전.
*** 2단계: 기둥을 박스-목표 '사이'에 랜덤 배치 + expert가 기둥을 우회(detour).
    우회 방향 = 박스→목표 직선의 수직 양옆 중 '도달 가능 & 경로 짧은' 쪽 자동 선택.
    기둥 위치를 obs에 기록 → 3단계에서 train_bc(N_WAYPOINTS=9)로 학습. ***
*** 순수 시뮬. 실물 로봇 명령 없음. ***

실행:
    python g1_collect_demos_obs.py --n 5 --view     # 우회 동작 눈으로 확인
    python g1_collect_demos_obs.py --n 20           # 성공률 확인
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

# 장애물 씬 (기둥 포함). 머신에 맞게 경로 수정.
XML = r"C:\Users\USER\g1bc\mink-main\examples\unitree_g1\scene_g1_pickplace_obs.xml"
# 우분투: "/home/computer/mink/examples/unitree_g1/scene_g1_pickplace_obs.xml"


class ViewerClosed(Exception):
    pass

BLOCK_QADR = 50
BLOCK_SIZE = 0.025
APPROACH_H = 0.12
GRASP_OFF  = 0.02
STEPS_PER_WP = 80
SETTLE_STEPS = 50
TABLE_TOP_Z = 0.835
SUCCESS_DIST = 0.05

# 넓힌 워크스페이스 (사용자 설정)
WS_X = (0.25, 0.55)
WS_Y = (0.00, 0.40)
BLOCK_SIZE_RANGE = (0.020, 0.035)

# 타원 도달 범위 (넓힌 값)
ELLIPSE_CX = 0.40
ELLIPSE_CY = 0.16
ELLIPSE_A  = 0.12
ELLIPSE_B  = 0.17

# --- 장애물(기둥) 설정 ---
OBS_R       = 0.03                    # 기둥 반지름[m] (씬 geom과 일치)
OBS_Z       = TABLE_TOP_Z + 0.10      # 기둥 중심 z (높이 20cm → 중심 0.935)
MIN_SEP_OBS = 0.14                    # 박스-목표 최소거리 (기둥+우회여유 확보 위해 키움)
OBS_JITTER  = 0.02                    # 기둥을 중점에서 흔드는 정도[m]
CLEAR       = 0.06                    # 손 우회 여유[m]. 전완은 IK 충돌회피가 처리하므로 손만 비키면 됨(작게)

# --- 기둥 '없음' 케이스 (일반화: 기둥 있으면 피하고 없으면 직진) ---
NO_OBS_RATIO = 0.30                   # 이 비율만큼은 기둥을 경로 밖 먼 곳에 치움(=없음)
OBS_FAR_XY   = (0.40, -0.30)          # 기둥 치워둘 먼 위치 (경로/워크스페이스 밖, y 음수쪽)


def in_reach(x, y):
    return ((x - ELLIPSE_CX) / ELLIPSE_A) ** 2 + ((y - ELLIPSE_CY) / ELLIPSE_B) ** 2 < 1.0


def detour_point(box_xy, target_xy, obs_xy):
    """
    기둥을 비껴가는 경유점을 양옆(박스→목표 직선의 수직)으로 계산하고,
    '몸쪽(어깨에 가까운 = 왼팔이 편한)' 쪽을 우선 선택. 단 도달 가능(타원 안)할 때만.
    둘 다 도달 불가면 None.  (전완 회피는 IK 충돌회피가 따로 보장)
    """
    box = np.array(box_xy, dtype=float)
    tgt = np.array(target_xy, dtype=float)
    obs = np.array(obs_xy, dtype=float)
    d = tgt - box
    n = np.linalg.norm(d)
    if n < 1e-6:
        return None
    d /= n
    perp = np.array([-d[1], d[0]])              # 직선에 수직 (xy 90도 회전)
    off = OBS_R + CLEAR
    cands = [obs + perp * off, obs - perp * off]
    # 왼쪽 어깨 위치(대략) — '몸쪽'은 여기에 가까운 쪽. 가까운 순으로 정렬.
    SHOULDER_XY = np.array([0.0, 0.10])
    cands.sort(key=lambda c: np.linalg.norm(c - SHOULDER_XY))
    for c in cands:                              # 몸쪽(어깨 가까운)부터 → 도달 가능한 첫 번째 채택
        if in_reach(c[0], c[1]):
            return (float(c[0]), float(c[1]))
    return None


def quat_wxyz(mat3):
    q = R.from_matrix(mat3).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def collect_one(model, block_xy, target_xy, obstacle_xy, detour_xy, block_size=BLOCK_SIZE, view=False):
    """한 에피소드 → (기록 dict, 성공여부, block_init, target_pos, obstacle_pos)."""
    configuration = mink.Configuration(model)
    data = configuration.data

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
    hand_tasks = [
        mink.FrameTask("right_palm", "site", position_cost=2.0,  orientation_cost=1.0, lm_damping=1.0),
        mink.FrameTask("left_palm",  "site", position_cost=20.0, orientation_cost=1.0, lm_damping=1.0),
    ]
    tasks.extend(feet_tasks + hand_tasks)
    limits = [mink.ConfigurationLimit(model)]
    # --- (A) 충돌 회피: 왼팔 전완 캡슐 ↔ 기둥 이 일정 거리 안으로 못 들어오게 ---
    # IK가 팔 '자세'를 비껴서 풀어 → 손뿐 아니라 전완도 기둥 안 뚫음.
    # geom 쌍: (왼팔 링크들) ↔ (기둥). minimum_distance=2cm 여유.
    obstacle_avoid = mink.CollisionAvoidanceLimit(
        model,
        geom_pairs=[(["left_hand_collision"], ["obstacle_geom"])],
        minimum_distance_from_collisions=0.02,
        collision_detection_distance=0.10,
    )
    limits.append(obstacle_avoid)

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{f}_target").mocapid[0] for f in feet]
    hands_mid = [model.body(f"{h}_target").mocapid[0] for h in hands]
    obstacle_mid = model.body("obstacle").mocapid[0]
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

    # --- 기둥 배치 (mocap 이동) ---
    obstacle_pos = np.array([obstacle_xy[0], obstacle_xy[1], OBS_Z])
    data.mocap_pos[obstacle_mid] = obstacle_pos

    mujoco.mj_forward(model, data)
    block_pos = data.xpos[block_bid].copy()

    above = lambda p: np.array([p[0], p[1], p[2] + APPROACH_H])
    on    = lambda p: np.array([p[0], p[1], p[2] + GRASP_OFF])
    # [2단계] 우회: lift와 move 사이에 '기둥 옆 경유점'을 든 상태(grip=1)로 삽입.
    carry_z = above(block_pos)[2]                      # 박스 든 높이 유지
    detour_wp = np.array([detour_xy[0], detour_xy[1], carry_z])
    waypoints = [
        ("approach", above(block_pos),  0), ("descend", on(block_pos),    0),
        ("grasp",    on(block_pos),     1), ("lift",    above(block_pos), 1),
        ("detour",   detour_wp,         1),                               # ← 기둥 우회 경유점
        ("move",     above(target_pos), 1), ("place",   on(target_pos),   1),
        ("release",  on(target_pos),    0), ("retreat", above(target_pos),0),
    ]

    rate = RateLimiter(frequency=100.0)
    rec = {k: [] for k in ["hand_pos", "hand_quat", "block_pos", "block_quat",
                           "target_pos", "obstacle_pos", "block_size", "waypoint_idx",
                           "waypoint_grip", "act_hand_target", "act_grasp"]}
    state = {"grasped": False}
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) if view else None

    def step_once(cmd, grip, wi):
        if grip == 1:
            state["grasped"] = True
        data.mocap_pos[hands_mid[L]] = cmd
        data.mocap_quat[hands_mid[L]] = left_quat
        data.mocap_pos[obstacle_mid] = obstacle_pos   # 기둥 위치 유지
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
        rec["obstacle_pos"].append(obstacle_pos.copy())
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
    return rec, dist < SUCCESS_DIST, block_init, target_pos, obstacle_pos, detour_wp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model = mujoco.MjModel.from_xml_path(XML)
    os.makedirs("demos_obs", exist_ok=True)   # 장애물 데모는 별도 폴더
    print(f"[워크스페이스] 사각형 x{WS_X} y{WS_Y} ∩ 타원(중심 {ELLIPSE_CX},{ELLIPSE_CY} 반경 {ELLIPSE_A},{ELLIPSE_B})")
    print(f"[장애물] 기둥 r={OBS_R}, 박스-목표 최소거리 {MIN_SEP_OBS}, 우회 ON. 기둥없음 비율 {NO_OBS_RATIO*100:.0f}%")

    saved = tried = ran = skipped = 0
    while saved < args.n:
        tried += 1
        # 박스/목표를 y축 반대편에 배치 (기둥이 그 사이에 오게)
        if rng.random() < 0.5:
            by = rng.uniform(WS_Y[0], ELLIPSE_CY)   # 박스 아래쪽
            ty = rng.uniform(ELLIPSE_CY, WS_Y[1])   # 목표 위쪽
        else:
            by = rng.uniform(ELLIPSE_CY, WS_Y[1])   # 박스 위쪽
            ty = rng.uniform(WS_Y[0], ELLIPSE_CY)   # 목표 아래쪽
        bx, tx = rng.uniform(*WS_X), rng.uniform(*WS_X)
        bsize = rng.uniform(*BLOCK_SIZE_RANGE)

        if np.hypot(bx - tx, by - ty) < MIN_SEP_OBS:
            skipped += 1; continue
        if not (in_reach(bx, by) and in_reach(tx, ty)):
            skipped += 1; continue

        if rng.random() < NO_OBS_RATIO:
            # [기둥 없음] 기둥을 경로 밖 먼 곳으로 치우고, detour = 박스-목표 직선 중점(직진 경유점)
            ox, oy = OBS_FAR_XY
            detour = ((bx + tx) / 2, (by + ty) / 2)   # 우회 안 함 (직진 위의 점)
        else:
            # [기둥 있음] 기둥 = 박스-목표 중점 + 흔들림, 우회 경유점 계산
            ox = (bx + tx) / 2 + rng.uniform(-OBS_JITTER, OBS_JITTER)
            oy = (by + ty) / 2 + rng.uniform(-OBS_JITTER, OBS_JITTER)
            if not in_reach(ox, oy):
                skipped += 1; continue
            detour = detour_point((bx, by), (tx, ty), (ox, oy))
            if detour is None:
                skipped += 1; continue

        ran += 1
        try:
            rec, ok, binit, tpos, opos, detour_wp = collect_one(
                model, (bx, by), (tx, ty), (ox, oy), detour, block_size=bsize, view=args.view)
        except ViewerClosed:
            print("뷰어를 닫아 수집을 중단합니다.")
            break
        if not ok:
            if ran - saved > max(30, args.n * 4):
                print("[!] 실제 IK 실패가 너무 많음.")
                break
            continue
        np.savez(f"demos_obs/demo_{saved:04d}.npz",
                 **{k: np.asarray(v) for k, v in rec.items()},
                 block_init_xpos=binit, target_xpos=tpos, obstacle_xpos=opos,
                 detour_xpos=detour_wp, block_size_scalar=np.float64(bsize))
        saved += 1
        if saved % 10 == 0:
            print(f"  {saved}/{args.n} 저장 (실행 {ran}회 / 샘플 버림 {skipped}회)")

    print(f"[V] 완료: {saved}개 저장 (demos_obs/)")
    print(f"    실제 IK 성공률: {saved}/{ran} = {saved/max(ran,1)*100:.0f}%")
    print(f"    (전체 시도 {tried}회 중 샘플 버림 {skipped}회)")


if __name__ == "__main__":
    main()