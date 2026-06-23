import sys
import time
import csv
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp  # [추가됨] 회전 보간을 위한 Slerp 임포트
from scipy.interpolate import interp1d

# Unitree SDK 필수 임포트
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

# Dex3 왼손 그리퍼 (같은 폴더의 dex3_gripper.py)
from dex3_gripper import Dex3Gripper

import logging

_HERE = Path(__file__).parent
_XML = Path("/home/computer/mink/examples/unitree_g1/scene_table.xml")

JOINT_MAP_ARM = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

# 그리퍼 세기 (물체 두께에 맞게 조절)
GRASP_STRENGTH = 2.0   # 쥐는 힘(kp). 너무 크면 과부하 — 2.0 권장
GRASP_EXTENT   = 1.0   # 쥐는 정도 0~1 (물체 있으면 0.6~0.8 권장, 공중 테스트는 1.0 OK)

class RealRobotInterface:
    def __init__(self):
        self.low_state = None

    def low_state_callback(self, msg: LowState_):
        self.low_state = msg

def get_current_robot_joints(low_state_msg):
    joints = np.zeros(29)
    for i in range(29):
        joints[i] = low_state_msg.motor_state[i].q
    return joints

def get_mapped_qpos(low_state_msg):
    q = np.zeros(50)
    q[2] = 0.76
    q[3] = 1.0
    if hasattr(low_state_msg, 'imu_state') and hasattr(low_state_msg.imu_state, 'quaternion'):
        q[3:7] = low_state_msg.imu_state.quaternion
    for i in range(22):
        q[i + 7] = low_state_msg.motor_state[i].q
    for i in range(22, 29):
        q[i + 14] = low_state_msg.motor_state[i].q
    return q

def send_robot_command(pub, low_cmd, crc, joints, control_mode, weight):
    if control_mode == "ARM_SDK":
        low_cmd.motor_cmd[29].q = weight
        for j in JOINT_MAP_ARM:
            low_cmd.motor_cmd[j].mode = 1
            low_cmd.motor_cmd[j].kp = 60.0
            low_cmd.motor_cmd[j].kd = 1.5
            low_cmd.motor_cmd[j].dq = 0.0
            low_cmd.motor_cmd[j].tau = 0.0
            low_cmd.motor_cmd[j].q = joints[j]
    else:
        for j in range(29):
            low_cmd.motor_cmd[j].mode = 1
            low_cmd.motor_cmd[j].kp = 140.0 if j < 12 else 60.0
            low_cmd.motor_cmd[j].kd = 2.5 if j < 12 else 1.5
            low_cmd.motor_cmd[j].dq = 0.0
            low_cmd.motor_cmd[j].tau = 0.0
            low_cmd.motor_cmd[j].q = joints[j]
            
    low_cmd.crc = crc.Crc(low_cmd)
    pub.Write(low_cmd)

def load_trajectory_from_csv(file_path):
    trajectory = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 스킵
        for row in reader:
            if not row: continue
            traj_point = {
                'lh_pos': np.array(row[1:4], dtype=float),
                'lh_rpy': np.array(row[4:7], dtype=float),
                'rh_pos': np.array(row[7:10], dtype=float),
                'rh_rpy': np.array(row[10:13], dtype=float),
                'grasp': int(float(row[13])) if len(row) > 13 else 0   # 14번째 열 (없으면 0)
            }
            trajectory.append(traj_point)
    return trajectory

def resample_trajectory(trajectory, slowdown_factor=2.0):
    """
    궤적의 재생 속도를 slowdown_factor 배만큼 늦추고,
    빈 공간을 부드럽게 보간(Interpolation)하여 새로운 궤적 리스트를 반환합니다.
    """
    if slowdown_factor <= 1.0 or len(trajectory) < 2:
        return trajectory

    N_orig = len(trajectory)
    N_new = int(N_orig * slowdown_factor)
    
    # 0부터 1까지의 가상 시간 생성
    t_orig = np.linspace(0, 1, N_orig)
    t_new = np.linspace(0, 1, N_new)

    # 1. 궤적에서 데이터 추출 및 변환
    lh_pos = np.array([pt['lh_pos'] for pt in trajectory])
    rh_pos = np.array([pt['rh_pos'] for pt in trajectory])
    
    # Slerp 적용을 위해 RPY를 Scipy Rotation 객체로 변환
    lh_rot = R.from_euler('xyz', [pt['lh_rpy'] for pt in trajectory])
    rh_rot = R.from_euler('xyz', [pt['rh_rpy'] for pt in trajectory])

    # 2. 위치(XYZ) 선형 보간
    lh_pos_interp = interp1d(t_orig, lh_pos, axis=0)(t_new)
    rh_pos_interp = interp1d(t_orig, rh_pos, axis=0)(t_new)

    # 3. 회전(RPY) Slerp 보간
    lh_slerp = Slerp(t_orig, lh_rot)
    rh_slerp = Slerp(t_orig, rh_rot)
    
    # 보간된 회전값을 다시 RPY(Euler)로 변환
    lh_rpy_interp = lh_slerp(t_new).as_euler('xyz')
    rh_rpy_interp = rh_slerp(t_new).as_euler('xyz')

    # 3b. grasp(0/1)는 보간하지 않고 최근접(계단식)으로 운반
    grasp_arr = np.array([pt.get('grasp', 0) for pt in trajectory], dtype=float)
    grasp_interp = interp1d(t_orig, grasp_arr, kind='nearest')(t_new)

    # 4. 새로운 궤적 리스트 재조립
    new_trajectory = []
    for i in range(N_new):
        new_trajectory.append({
            'lh_pos': lh_pos_interp[i],
            'lh_rpy': lh_rpy_interp[i],
            'rh_pos': rh_pos_interp[i],
            'rh_rpy': rh_rpy_interp[i],
            'grasp': int(round(grasp_interp[i]))
        })
        
    return new_trajectory

if __name__ == "__main__":
    CONTROL_MODE = "ARM_SDK"  # "ARM_SDK" 또는 "FULL_BODY"
    
    if len(sys.argv) < 3:
        print("사용법: python3 g1_traj_playback.py [네트워크_인터페이스(예: eth0)] [CSV_궤적파일_경로]")
        sys.exit(1)

    # 1. 파일 데이터 로드 (CSV)
    csv_path = Path(sys.argv[2])
    if not csv_path.exists():
        print(f"[오류] 데이터 파일({csv_path})을 찾을 수 없습니다.")
        sys.exit(1)
        
    print(f"[{csv_path.name}] 파일에서 궤적 데이터를 불러옵니다...")
    trajectory_data = load_trajectory_from_csv(csv_path)
    
    # ================= [추가된 부분] =================
    SLOWDOWN_FACTOR = 6.0  # 💡 여기에 원하는 n배수 입력 (예: 3.0 = 3배 느리게 재생)
    print(f"궤적을 {SLOWDOWN_FACTOR}배 느리게 설정하고 부드럽게 보간합니다...")
    trajectory_data = resample_trajectory(trajectory_data, slowdown_factor=SLOWDOWN_FACTOR)
    # =================================================
    
    print(f"[V] 총 {len(trajectory_data)} 프레임의 궤적 로드 및 보간 완료.")

    # 2. SDK 초기화
    ChannelFactoryInitialize(0, sys.argv[1])
    robot = RealRobotInterface()
    crc = CRC()
    low_cmd = unitree_hg_msg_dds__LowCmd_()
    
    topic_name = "rt/arm_sdk" if CONTROL_MODE == "ARM_SDK" else "rt/lowcmd"
    pub = ChannelPublisher(topic_name, LowCmd_)
    pub.Init()
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(robot.low_state_callback, 10)

    # Dex3 왼손 그리퍼 — ik_traj가 이미 연 채널을 공유 (init_network=False)
    gripper = Dex3Gripper(hand_side="L", init_network=False)
    print("[V] Dex3 왼손 그리퍼 준비 완료 (초기: 힘 빠짐).")

    print("실제 로봇 텔레메트리 수신 대기 중...")
    while robot.low_state is None:
        time.sleep(0.1)
    print("[V] 로봇 동기화 완료. 시뮬레이터 창이 열립니다.")

    # 3. MuJoCo / Mink 모델 초기화
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    configuration = mink.Configuration(model)
    
    initial_robot_joints = get_current_robot_joints(robot.low_state)
    initial_qpos = get_mapped_qpos(robot.low_state)
    configuration.update(initial_qpos)
    data = configuration.data

    rate = RateLimiter(frequency=200.0)

    # 4. 태스크 및 IK 설정
    hands = ["right_palm", "left_palm"]
    feet = ["right_foot", "left_foot"]
    tasks = [
        pelvis_orientation_task := mink.FrameTask("pelvis", "body", position_cost=0.0, orientation_cost=1.0, lm_damping=1.0),
        torso_orientation_task := mink.FrameTask("torso_link", "body", position_cost=0.0, orientation_cost=1.0, lm_damping=1.0),
        posture_task := mink.PostureTask(model, cost=1e-1),
        com_task := mink.ComTask(cost=10.0),
    ]
    feet_tasks = [mink.FrameTask(f, "site", position_cost=10.0, orientation_cost=1.0, lm_damping=1.0) for f in feet]
    hand_tasks = [mink.FrameTask(h, "site", position_cost=2.0, orientation_cost=1.0, lm_damping=1.0) for h in hands]
    tasks.extend(feet_tasks + hand_tasks)
    limits = [mink.ConfigurationLimit(model)]
    
    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{f}_target").mocapid[0] for f in feet]
    hands_mid = [model.body(f"{h}_target").mocapid[0] for h in hands]

    # ========== [추가된 부분: 초기 자세 부드러운 보간(Interpolation) 궤적 생성] ==========
    print("\n[보간] 로봇의 현재 자세에서 CSV 첫 프레임까지 부드럽게 이동하는 궤적을 생성합니다.")
    
    # FK를 업데이트하여 현재 로봇 자세 기준의 site 프레임 위치를 계산
    mujoco.mj_kinematics(model, data)
    for hand, foot in zip(hands, feet):
        mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
        mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")

    # 시작(현재) 양손 위치/회전 추출 (MuJoCo 쿼터니언 [w,x,y,z] -> Scipy [x,y,z,w])
    rh_pos_start = data.mocap_pos[hands_mid[0]].copy()
    rh_q = data.mocap_quat[hands_mid[0]]
    rh_rot_start = R.from_quat([rh_q[1], rh_q[2], rh_q[3], rh_q[0]])

    lh_pos_start = data.mocap_pos[hands_mid[1]].copy()
    lh_q = data.mocap_quat[hands_mid[1]]
    lh_rot_start = R.from_quat([lh_q[1], lh_q[2], lh_q[3], lh_q[0]])

    # 목표(CSV 첫 프레임) 위치/회전 추출
    first_frame = trajectory_data[0]
    rh_pos_end = first_frame['rh_pos']
    rh_rot_end = R.from_euler('xyz', first_frame['rh_rpy'])
    
    lh_pos_end = first_frame['lh_pos']
    lh_rot_end = R.from_euler('xyz', first_frame['lh_rpy'])

    # Slerp 객체 생성
    rh_slerp = Slerp([0, 1], R.from_quat([rh_rot_start.as_quat(), rh_rot_end.as_quat()]))
    lh_slerp = Slerp([0, 1], R.from_quat([lh_rot_start.as_quat(), lh_rot_end.as_quat()]))

    # 2.0초(200Hz 기준 400스텝) 동안 보간 수행
    INTERP_TIME = 2.0
    interp_steps = int(INTERP_TIME * (1.0 / rate.dt))
    prep_trajectory = []

    for i in range(interp_steps):
        alpha = i / float(interp_steps)
        # S-Curve(코사인 보간) 적용: 시작과 끝에서 부드럽게 가감속
        alpha_smooth = 0.5 * (1.0 - np.cos(np.pi * alpha))
        
        rh_pos_curr = rh_pos_start + (rh_pos_end - rh_pos_start) * alpha_smooth
        lh_pos_curr = lh_pos_start + (lh_pos_end - lh_pos_start) * alpha_smooth
        
        rh_rpy_curr = rh_slerp(alpha_smooth).as_euler('xyz')
        lh_rpy_curr = lh_slerp(alpha_smooth).as_euler('xyz')
        
        prep_trajectory.append({
            'lh_pos': lh_pos_curr,
            'lh_rpy': lh_rpy_curr,
            'rh_pos': rh_pos_curr,
            'rh_rpy': rh_rpy_curr,
            'grasp': 0
        })

    # 기존 궤적 데이터의 제일 앞에 보간 궤적 연결
    trajectory_data = prep_trajectory + trajectory_data
    print(f"[V] {interp_steps} 프레임({INTERP_TIME}초)의 초기 보간 궤적 추가 완료.")
    # =========================================================================

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.sync()
        
        print("\n==========================================================")
        print(" >> 터미널에서 [Enter]를 누르면 시뮬레이터 내에서 '가상 프리뷰 주행'을 시작합니다.")
        print("==========================================================")
        input("Press [Enter] to start Virtual Preview...")

        # -------------------------------------------------------------
        # 🏃 [PHASE 1] 시뮬레이션 가상 주행 (Preview) Loop
        # -------------------------------------------------------------
        print("\n[프리뷰] 기록된 궤적을 따라 가상 주행 중... 창을 확인하세요.")
        
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)
        torso_orientation_task.set_target_from_configuration(configuration)
        
        for hand, foot in zip(hands, feet):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
        data.mocap_pos[com_mid] = data.subtree_com[1]

        com_task.set_target(data.mocap_pos[com_mid])
        for i, f_task in enumerate(feet_tasks):
            f_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))

        # CSV 궤적의 모든 프레임을 순차 재생
        prev_grasp_preview = 0
        for step_data in trajectory_data:
            if not viewer.is_running(): sys.exit(0)

            g = step_data.get('grasp', 0)
            if g != prev_grasp_preview:
                print(f"  [프리뷰] grasp {'닫힘(집기)' if g == 1 else '열림(놓기)'} 지점 — 실제 손은 PHASE 2에서만 작동")
                prev_grasp_preview = g

            q_r = R.from_euler('xyz', step_data['rh_rpy']).as_quat()
            q_l = R.from_euler('xyz', step_data['lh_rpy']).as_quat()

            data.mocap_pos[hands_mid[0]] = step_data['rh_pos']
            data.mocap_quat[hands_mid[0]] = np.array([q_r[3], q_r[0], q_r[1], q_r[2]])
            
            data.mocap_pos[hands_mid[1]] = step_data['lh_pos']
            data.mocap_quat[hands_mid[1]] = np.array([q_l[3], q_l[0], q_l[1], q_l[2]])

            hand_tasks[0].set_target(mink.SE3.from_mocap_id(data, hands_mid[0]))
            hand_tasks[1].set_target(mink.SE3.from_mocap_id(data, hands_mid[1]))

            vel = mink.solve_ik(configuration, tasks, rate.dt, "daqp", limits=limits)
            configuration.integrate_inplace(vel, rate.dt)
            
            send_robot_command(pub, low_cmd, crc, initial_robot_joints, CONTROL_MODE, weight=0.5)
            
            viewer.sync()
            rate.sleep()

        # -------------------------------------------------------------
        # 🛑 2단계 대기 (실제 로봇 실행 대기)
        # -------------------------------------------------------------
        print("\n==========================================================")
        print(" >> 가상 프리뷰 주행이 종료되었습니다.")
        print(" >> 터미널에서 [Enter]를 누르면 실제 로봇이 동일한 궤적으로 구동됩니다.")
        print("==========================================================")
        input("Press [Enter] to execute on REAL ROBOT...")

        # -------------------------------------------------------------
        # 🤖 [PHASE 2] 실제 로봇 구동 루프
        # -------------------------------------------------------------
        print("\n[구동] 실제 로봇 전이 시작! 로봇 궤적 추종을 시작합니다.")
        
        configuration.update(initial_qpos) 
        for hand, foot in zip(hands, feet):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")

        arm_sdk_weight = 0.0

        # 손은 벌린 상태로 시작
        gripper.open()
        prev_grasp = 0

        for step_data in trajectory_data:
            if not viewer.is_running(): break
            
            cmd_joints = np.zeros(29)

            q_r = R.from_euler('xyz', step_data['rh_rpy']).as_quat()
            q_l = R.from_euler('xyz', step_data['lh_rpy']).as_quat()
            
            data.mocap_pos[hands_mid[0]] = step_data['rh_pos']
            data.mocap_quat[hands_mid[0]] = np.array([q_r[3], q_r[0], q_r[1], q_r[2]])
            
            data.mocap_pos[hands_mid[1]] = step_data['lh_pos']
            data.mocap_quat[hands_mid[1]] = np.array([q_l[3], q_l[0], q_l[1], q_l[2]])

            hand_tasks[0].set_target(mink.SE3.from_mocap_id(data, hands_mid[0]))
            hand_tasks[1].set_target(mink.SE3.from_mocap_id(data, hands_mid[1]))

            vel = mink.solve_ik(configuration, tasks, rate.dt, "daqp", limits=limits)
            configuration.integrate_inplace(vel, rate.dt)
            
            for j in range(29):
                sim_idx = (j + 7) if j < 22 else (j + 14)
                cmd_joints[j] = configuration.q[sim_idx]

            if arm_sdk_weight < 1.0:
                arm_sdk_weight = min(1.0, arm_sdk_weight + 0.01)

            send_robot_command(pub, low_cmd, crc, cmd_joints, CONTROL_MODE, weight=arm_sdk_weight)

            # grasp 신호 전환 시 손 구동 (0→1 닫음, 1→0 폄)
            g = step_data.get('grasp', 0)
            if g != prev_grasp:
                if g == 1:
                    gripper.close(strength=GRASP_STRENGTH, extent=GRASP_EXTENT)
                    print("  [grip] 손 닫음 (집기)")
                else:
                    gripper.open()
                    print("  [grip] 손 폄 (놓기)")
                prev_grasp = g

            viewer.sync()
            rate.sleep()

        # -------------------------------------------------------------
        # ✅ 종료: 차렷 자세에서 2초 안정 후 자동 정지
        # -------------------------------------------------------------
        print("[V] 궤적 재생 완료. 차렷 자세에서 2초 안정 후 자동 종료합니다.")
        try:
            hold = int(2.0 / rate.dt)          # 2초간 마지막(차렷) 자세 유지 (원하면 2.0을 조절)
            for _ in range(hold):
                if not viewer.is_running():
                    break
                send_robot_command(pub, low_cmd, crc, cmd_joints, CONTROL_MODE, weight=1.0)
                viewer.sync()
                rate.sleep()
        except KeyboardInterrupt:
            print("\n[!] 사용자 종료 명령 감지.")
        finally:
            print("[!] 제어권 이양 및 모터 댐핑 종료 처리를 시작합니다.")
            # Dex3 손 먼저 안전 종료
            try:
                gripper.shutdown()
                print("[V] Dex3 그리퍼 종료 완료.")
            except Exception as e:
                print(f"[!] 그리퍼 종료 중 경고: {e}")
            if CONTROL_MODE == "ARM_SDK" and robot.low_state is not None:
                steps = int(1.0 / rate.dt)
                for i in range(steps):
                    low_cmd.motor_cmd[29].q = 1.0 - (i / float(steps))
                    low_cmd.crc = crc.Crc(low_cmd)
                    pub.Write(low_cmd)
                    time.sleep(rate.dt)
                low_cmd.motor_cmd[29].q = 0.0
                low_cmd.crc = crc.Crc(low_cmd)
                pub.Write(low_cmd)
            else:
                for j in range(29):
                    low_cmd.motor_cmd[j].mode = 0
                    low_cmd.motor_cmd[j].kp = 0.0
                    low_cmd.motor_cmd[j].kd = 2.0
                low_cmd.crc = crc.Crc(low_cmd)
                pub.Write(low_cmd)
            print("[V] 안전하게 제어가 종료되었습니다.")
