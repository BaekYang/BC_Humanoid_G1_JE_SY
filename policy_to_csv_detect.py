"""
policy_to_csv.py — 학습된 BC 정책을 sim에서 롤아웃해 '왼손 궤적'을 ik_traj.py용 CSV로 저장.
이번 버전: 차렷(초기자세)에서 출발 → pick&place → 차렷 복귀 하도록 리드인/리드아웃 추가.

CSV 형식 (ik_traj.py가 읽는 그대로):
  [timestamp, LH_x,LH_y,LH_z, LH_R,LH_P,LH_Y, RH_x,RH_y,RH_z, RH_R,RH_P,RH_Y]   (RPY 라디안)
  - 오른손은 시작 자세로 고정, 왼손만 궤적을 따름.
*** 이 스크립트 자체는 순수 시뮬레이션. 실물 명령 없음. ***

실행:
    python3 policy_to_csv.py --repeat 8           # 차렷→pick&place→차렷, 느린 CSV 저장
    python3 policy_to_csv.py --view --slowmo 8    # 천천히 미리보기 (차렷부터 전체)
    python3 policy_to_csv.py --no-return          # 복귀(리드아웃) 빼기
    --bx --by --tx --ty --size 로 블록/타겟 변경
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
from scipy.spatial.transform import Rotation as R, Slerp
from loop_rate_limiters import RateLimiter

import logging
logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)

XML = r"C:\Users\USER\g1bc\mink-main\examples\unitree_g1\scene_g1_pickplace.xml"
BLOCK_QADR = 50
LEFT_ARM_QADR = 22          # 왼팔 7관절 qpos 시작 (22~28): 차렷 = 전부 0
RIGHT_ARM_QADR = 36         # 오른팔 7관절 qpos 시작 (36~42): 차렷 = 전부 0
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
def rpy_to_wxyz(rpy): q = R.from_euler('xyz', rpy).as_quat(); return np.array([q[3], q[0], q[1], q[2]])
def wxyz_to_rpy(q): return R.from_quat([q[1], q[2], q[3], q[0]]).as_euler('xyz')


def resample(frames, max_step):
    """연속 점 간격이 max_step(m)을 넘으면 중간 점을 보간해 넣음 (위치 lerp + 방향 slerp).
    grasp는 보간하지 않고 계단식: 중간 점은 시작 프레임 값(g0)을 유지, 도착 프레임에서 g1로 전환."""
    out = [frames[0]]
    for (p0, r0, g0), (p1, r1, g1) in zip(frames[:-1], frames[1:]):
        p0 = np.asarray(p0); p1 = np.asarray(p1)
        d = np.linalg.norm(p1 - p0)
        n = max(1, int(np.ceil(d / max_step)))
        if np.allclose(r0, r1):
            for k in range(1, n + 1):
                a = k / n
                g = g1 if k == n else g0          # 도착 프레임에서만 g1
                out.append(((1 - a) * p0 + a * p1, np.asarray(r0), g))
        else:
            sl = Slerp([0, 1], R.concatenate([R.from_euler('xyz', r0), R.from_euler('xyz', r1)]))
            for k in range(1, n + 1):
                a = k / n
                g = g1 if k == n else g0
                out.append(((1 - a) * p0 + a * p1, sl(a).as_euler('xyz'), g))
    return out


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
    ap.add_argument("--out", type=str, default=None,
                    help="출력 CSV 이름. 안 주면 좌표로 자동: traj_bx_by_to_tx_ty.csv")
    ap.add_argument("--repeat", type=int, default=2, help="각 점 N번 반복 = 실물 재생속도 늦춤(안전)")
    ap.add_argument("--slowmo", type=float, default=5.0, help="뷰어 보기속도만 늦춤(CSV 무관)")
    ap.add_argument("--maxstep", type=float, default=0.005, help="점 사이 최대 간격(m). 작을수록 더 촘촘/부드러움")
    ap.add_argument("--clear", type=float, default=0.12, help="박스 위로 띄울 여유 높이(m). 팔 올릴 때 박스 안 걸리게")
    ap.add_argument("--leadin", type=int, default=100, help="차렷→시작점 보간 프레임 수")
    ap.add_argument("--leadout", type=int, default=100, help="끝점→차렷 복귀 보간 프레임 수")
    ap.add_argument("--no-return", action="store_true", help="차렷 복귀(리드아웃) 생략")
    ap.add_argument("--grip-rpy", type=float, nargs=3, default=None, metavar=("R", "P", "Y"),
                    help="악수(정면 보기) 손목 방향 rpy[rad]. 주면 집기 구간 손목을 이 방향으로 고정. 없으면 기존 방향 유지")
    ap.add_argument("--palm-offset", type=float, default=0.04,
                    help="손목→손바닥 중앙 거리[m]. --grip-rpy 줄 때만 적용. 손목을 정면 축으로 이만큼 뒤로 빼서 손바닥 중앙이 박스에 닿게")
    ap.add_argument("--palm-axis", type=str, default="x", choices=["x", "y", "z", "-x", "-y", "-z"],
                    help="손 로컬 '정면(손목→손바닥)' 축. --view로 확인 후 틀리면 부호/축 변경")
    ap.add_argument("--view", action="store_true")
    # --- 비전 게이트 (실로봇 카메라 연결 시에만 사용. 시뮬만 돌릴 땐 안 켜면 됨) ---
    ap.add_argument("--vision-gate", action="store_true",
                    help="카메라 시야에 박스가 있을 때만 실행. 없으면 롤아웃 안 하고 즉시 종료")
    ap.add_argument("--net", type=str, default=None,
                    help="--vision-gate 용 네트워크 인터페이스 (예: eth0). 생략 가능")
    ap.add_argument("--min-pixels", type=int, default=500,
                    help="--vision-gate 판정 임계값(박스 색 픽셀 수). vision_gate.py로 맞춤")
    args = ap.parse_args()

    # --out 안 주면 좌표로 자동 파일명: traj_b<bx>_<by>_to_t<tx>_<ty>.csv
    if args.out is None:
        args.out = f"traj_{args.bx*100.0:.0f}_{args.by*100.0:.0f}_to_{args.tx*100.0:.0f}_{args.ty*100.0:.0f}.csv"

    # ---------- 비전 게이트: 카메라 시야에 박스가 있을 때만 진행 ----------
    # --vision-gate 줬을 때만 동작. 박스가 안 보이면 정책 로드/IK/CSV 전부 건너뛰고 종료.
    # (vision_gate.py 와 box_detector.py 가 같은 폴더에 있어야 함)
    if args.vision_gate:
        from box_detector import VisionGate
        gate = VisionGate(net=args.net, min_pixels=args.min_pixels)
        if not gate.object_visible(verbose=True):
            print("[비전] 시야에 박스 없음 → 롤아웃/CSV 생성 안 함. 종료.")
            return
        print("[비전] 시야에 박스 확인 → 진행.")

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
    left_rpy = wxyz_to_rpy(left_quat)              # grasp(준비) 자세 손목 방향

    # --- 악수(정면 보기) 방향 + 손목→손바닥 4cm 오프셋 (옵트인) ---
    # --grip-rpy 를 주면 집기 구간 손목을 그 방향으로 고정하고,
    # 손목을 '정면 축'으로 palm_offset 만큼 뒤로 빼서 손바닥 중앙이 박스에 닿게 함.
    if args.grip_rpy is not None:
        grip_rpy = np.array(args.grip_rpy, dtype=float)
        _axmap = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1],
                  "-x": [-1, 0, 0], "-y": [0, -1, 0], "-z": [0, 0, -1]}
        _fwd_local = np.array(_axmap[args.palm_axis], dtype=float)
        _fwd_world = R.from_euler('xyz', grip_rpy).apply(_fwd_local)   # 손 정면 방향(월드)
        palm_off_vec = -args.palm_offset * _fwd_world                  # 손목을 정면 반대로 빼기
        print(f"[악수모드] grip_rpy={np.round(grip_rpy,3)}, 손목→손바닥 {args.palm_offset*100:.0f}cm "
              f"오프셋 방향(월드)={np.round(_fwd_world,2)}")
    else:
        grip_rpy = left_rpy
        palm_off_vec = np.zeros(3)

    block_bid = model.body("block").id
    palm_sid = model.site("left_palm").id
    rpalm_sid = model.site("right_palm").id
    target_mid = model.body("target_marker").mocapid[0]
    target_pos = np.array([args.tx, args.ty, TABLE_TOP_Z + args.size])
    data.mocap_pos[target_mid] = target_pos
    mujoco.mj_forward(model, data)

    # --- 차렷(초기자세): 양팔 관절을 0으로 둔 FK (오른팔도 안 들도록) ---
    home = mujoco.MjData(model)
    home.qpos[:] = q0
    home.qpos[LEFT_ARM_QADR:LEFT_ARM_QADR + 7] = 0.0
    home.qpos[RIGHT_ARM_QADR:RIGHT_ARM_QADR + 7] = 0.0
    mujoco.mj_forward(model, home)
    start_pos = home.site_xpos[palm_sid].copy()
    start_rot = R.from_matrix(home.site_xmat[palm_sid].reshape(3, 3))
    grasp_rot = R.from_euler('xyz', left_rpy)

    # 오른손은 차렷 자세로 고정 (들리지 않게) — CSV·IK 목표 모두 여기로
    rh_pos = home.site_xpos[rpalm_sid].copy()
    rh_rpy = R.from_matrix(home.site_xmat[rpalm_sid].reshape(3, 3)).as_euler('xyz')
    data.mocap_pos[hands_mid[0]] = rh_pos
    data.mocap_quat[hands_mid[0]] = rpy_to_wxyz(rh_rpy)
    hand_tasks[0].set_target(mink.SE3.from_mocap_id(data, hands_mid[0]))

    goals = [above(binit), on(binit), on(binit), above(binit),
             above(target_pos), on(target_pos), on(target_pos), above(target_pos)]
    om, osd = stats["obs_mean"], stats["obs_std"]
    am, asd = stats["act_mean"], stats["act_std"]
    rate = RateLimiter(frequency=100.0)

    # ---------- 1) 정책 롤아웃 (헤드리스, traj 생성) ----------
    traj = [data.site_xpos[palm_sid].copy()]
    traj_grasp = [0]                          # 시작점은 손 벌림
    wp_idx = 0; dwell = 0; grasped = False; final_dwell = 0
    for _ in range(MAX_STEPS):
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
        traj_grasp.append(1 if grasp_pred > 0.5 else 0)   # 정책의 grasp 결정 기록

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

    # ---------- 2) 전체 프레임 구성: 차렷 → 수직상승 → pick&place → 복귀 ----------
    start_rpy = start_rot.as_euler('xyz')
    LIFTZ = TABLE_TOP_Z + args.size + args.clear          # 박스 위 안전 높이
    up_home = np.array([start_pos[0], start_pos[1], LIFTZ])  # 차렷 위로 수직 상승한 지점

    # rest→approach 저공 스윙 제거: traj에서 approach(박스 위) 도달 지점부터 사용
    approach0 = goals[0]
    idx0 = 0
    for i, p in enumerate(traj):
        if np.linalg.norm(np.asarray(p) - approach0) < 0.06:
            idx0 = i
            break
    main_traj = traj[idx0:]
    main_grasp = traj_grasp[idx0:]

    frames = [(start_pos, start_rpy, 0),   # 차렷 (손 벌림, 오프셋 없음 — 실제 홈자세 보존)
              (up_home, start_rpy, 0)]      # 박스 높이 위로 수직 상승
    for p, g in zip(main_traj, main_grasp):  # 고공으로 이동 후 pick & place (악수 방향 + 손목 오프셋)
        frames.append((np.asarray(p) + palm_off_vec, grip_rpy, g))
    if not args.no_return:                 # 복귀: 끝점 → 차렷 위 고공 → 차렷 하강 (손 벌림)
        frames.append((up_home + palm_off_vec, grip_rpy, 0))
        frames.append((start_pos, start_rpy, 0))

    # 일정 간격으로 촘촘하게 보간 (전 구간 부드럽게)
    frames = resample(frames, args.maxstep)

    # ---------- 3) (선택) 뷰어로 차렷부터 전체 미리보기 ----------
    if args.view:
        configuration.update(home.qpos.copy())   # 차렷에서 시작
        viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        for pos, rpy, _g in frames:
            if not viewer.is_running():
                break
            data.mocap_pos[hands_mid[L]] = pos
            data.mocap_quat[hands_mid[L]] = rpy_to_wxyz(rpy)
            hand_tasks[L].set_target(mink.SE3.from_mocap_id(data, hands_mid[L]))
            vel = mink.solve_ik(configuration, tasks, rate.dt, "daqp", limits=limits)
            configuration.integrate_inplace(vel, rate.dt)
            viewer.sync()
            time.sleep(args.slowmo / 100.0)
        viewer.close()

    # ---------- 4) CSV 저장 ----------
    header = ['timestamp',
              'LH_x', 'LH_y', 'LH_z', 'LH_R', 'LH_P', 'LH_Y',
              'RH_x', 'RH_y', 'RH_z', 'RH_R', 'RH_P', 'RH_Y',
              'grasp']
    dt = 1.0 / 200.0
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        t = 0.0
        for pos, rpy, g in frames:
            for _ in range(max(1, args.repeat)):
                w.writerow([t,
                            pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2],
                            rh_pos[0], rh_pos[1], rh_pos[2], rh_rpy[0], rh_rpy[1], rh_rpy[2],
                            int(g)])
                t += dt
    n_rows = len(frames) * max(1, args.repeat)
    print(f"[V] 저장: {args.out}  ({n_rows} 프레임, 약 {n_rows/200:.1f}초 재생)")
    print(f"    차렷 시작 {np.round(start_pos,3)} → 수직상승(z={LIFTZ:.2f}) → pick&place → {'차렷 복귀' if not args.no_return else '복귀 없음'}")
    print(f"    박스 위 {LIFTZ:.2f}m 로 올린 뒤 이동 (rest 저공 {idx0}프레임 건너뜀)")
    g_frames = sum(1 for _, _, g in frames if g)
    g_transitions = sum(1 for a, b in zip(frames[:-1], frames[1:]) if a[2] != b[2])
    print(f"    grasp 켜진 프레임 {g_frames}/{len(frames)} (열림→닫힘/닫힘→열림 전환 {g_transitions}회) → CSV 14번째 열")


import os

print("XML =", XML)
print("exists =", os.path.exists(XML))

if __name__ == "__main__":
    main()