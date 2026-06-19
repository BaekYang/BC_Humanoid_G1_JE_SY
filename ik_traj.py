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

# Unitree SDK 필수 임포트
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

# 로깅 경고 무시 (선택사항 - 터미널 깔끔하게 유지)
import logging


_HERE = Path(__file__).parent
_XML = Path("/home/computer/mink/examples/unitree_g1/scene_table.xml")

JOINT_MAP_ARM = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

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
    """CSV 파일에서 손의 궤적(XYZ, RPY)을 로드하여 리스트로 반환"""
    trajectory = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 스킵
        for row in reader:
            if not row: continue
            # 인덱싱 구조:
            # [1:4] Left Hand XYZ, [4:7] Left Hand RPY
            # [7:10] Right Hand XYZ, [10:13] Right Hand RPY
            traj_point = {
                'lh_pos': np.array(row[1:4], dtype=float),
                'lh_rpy': np.array(row[4:7], dtype=float),
                'rh_pos': np.array(row[7:10], dtype=float),
                'rh_rpy': np.array(row[10:13], dtype=float)
            }
            trajectory.append(traj_point)
    return trajectory


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
    print(f"[V] 총 {len(trajectory_data)} 프레임의 궤적 로드 완료.")

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
        for step_data in trajectory_data:
            if not viewer.is_running(): sys.exit(0)
            
            # Scipy R.from_euler는 기본적으로 라디안(radians)을 사용 (이전 코드에서 라디안으로 저장함 가정)
            q_r = R.from_euler('xyz', step_data['rh_rpy']).as_quat() # [x, y, z, w]
            q_l = R.from_euler('xyz', step_data['lh_rpy']).as_quat()

            # MuJoCo의 쿼터니언 순서는 [w, x, y, z] 이므로 재배치
            data.mocap_pos[hands_mid[0]] = step_data['rh_pos']
            data.mocap_quat[hands_mid[0]] = np.array([q_r[3], q_r[0], q_r[1], q_r[2]])
            
            data.mocap_pos[hands_mid[1]] = step_data['lh_pos']
            data.mocap_quat[hands_mid[1]] = np.array([q_l[3], q_l[0], q_l[1], q_l[2]])

            hand_tasks[0].set_target(mink.SE3.from_mocap_id(data, hands_mid[0]))
            hand_tasks[1].set_target(mink.SE3.from_mocap_id(data, hands_mid[1]))

            vel = mink.solve_ik(configuration, tasks, rate.dt, "daqp", limits=limits)
            configuration.integrate_inplace(vel, rate.dt)
            
            # 프리뷰 동안 실제 로봇은 시작 자세를 안전하게 유지
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
        
        # 모델을 로봇의 현재 (시작) 위치로 안전하게 복원 후 시작
        configuration.update(initial_qpos) 
        for hand, foot in zip(hands, feet):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")

        arm_sdk_weight = 0.0

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
            
            # IK 결과를 실제 로봇 전송용 인덱스 매핑으로 변환
            for j in range(29):
                sim_idx = (j + 7) if j < 22 else (j + 14)
                cmd_joints[j] = configuration.q[sim_idx]

            # 초반 구동 충격을 줄이기 위해 제어 가중치를 서서히 1.0까지 증가시킴
            if arm_sdk_weight < 1.0:
                arm_sdk_weight = min(1.0, arm_sdk_weight + 0.01)

            send_robot_command(pub, low_cmd, crc, cmd_joints, CONTROL_MODE, weight=arm_sdk_weight)

            viewer.sync()
            rate.sleep()

        print("[V] 궤적 재생이 완료되었습니다. 최종 자세를 유지합니다.")
        try:
            while viewer.is_running():
                send_robot_command(pub, low_cmd, crc, cmd_joints, CONTROL_MODE, weight=1.0)
                rate.sleep()
        except KeyboardInterrupt:
            print("\n[!] 사용자 종료 명령 감지.")
        finally:
            print("[!] 제어권 이양 및 모터 댐핑 종료 처리를 시작합니다.")
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
