import sys
import time
import csv
import threading
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink
import logging

# loop_rate_limiters 라이브러리의 경고 메시지 무시 설정
logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)

# Unitree SDK 임포트 (초기 자세 동기화용)
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

# 파일 경로 설정
_HERE = Path(__file__).parent
_XML = Path("/home/computer/mink/examples/unitree_g1/scene_table.xml")

class RealRobotInterface:
    def __init__(self):
        self.low_state = None

    def low_state_callback(self, msg: LowState_):
        self.low_state = msg

def get_mapped_qpos(low_state_msg):
    """실제 로봇의 현재 상태를 시뮬레이터 qpos 구조에 맞게 매핑"""
    q = np.zeros(50)
    q[2] = 0.76  # 초기 골반 높이 기본값
    q[3] = 1.0   # 쿼터니언 w
    
    if hasattr(low_state_msg, 'imu_state') and hasattr(low_state_msg.imu_state, 'quaternion'):
        q[3:7] = low_state_msg.imu_state.quaternion

    for i in range(22):
        q[i + 7] = low_state_msg.motor_state[i].q
    for i in range(22, 29):
        q[i + 14] = low_state_msg.motor_state[i].q
    return q

def mat2rpy(R):
    """3x3 회전 행렬을 Roll, Pitch, Yaw (라디안)로 변환"""
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    if np.isclose(np.cos(pitch), 0.0):
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])

# 전역 제어 변수 (터미널 입력 스레드와 메인 루프 간 공유)
recording = False
exit_flag = False

def console_input_thread():
    """사용자 엔터 입력을 감지하는 비동기 스레드"""
    global recording, exit_flag
    input(">> [준비] 엔터를 누르면 시뮬레이션 데이터 기록이 시작됩니다...\n")
    recording = True
    print("● [기록 중] 데이터 수집을 시작했습니다. 시뮬레이터에서 마커를 움직이세요.")
    
    input(">> [기록 중] 엔터를 누르면 기록을 중지하고 종료합니다...\n")
    recording = False
    exit_flag = True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 g1_sim_to_data_capture.py [네트워크_인터페이스(예: eth0)]")
        sys.exit(1)
    
    # 1. 로봇 데이터 수신 (시작 자세 동기화 목적)
    ChannelFactoryInitialize(0, sys.argv[1])
    robot = RealRobotInterface()
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(robot.low_state_callback, 10)

    print("실제 로봇으로부터 현재 자세(Telemetry)를 받아오는 중...")
    while robot.low_state is None:
        time.sleep(0.1)
    print("[V] 로봇 데이터 수신 완료. 실제 자세 그대로 시뮬레이터를 초기화합니다.")

    # 2. MuJoCo 모델 및 Mink IK 설정
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    configuration = mink.Configuration(model)
    
    initial_q = get_mapped_qpos(robot.low_state)
    configuration.update(initial_q)

    feet = ["right_foot", "left_foot"]
    hands = ["right_palm", "left_palm"]
    tasks = [
        pelvis_orientation_task := mink.FrameTask(frame_name="pelvis", frame_type="body", position_cost=0.0, orientation_cost=1.0, lm_damping=1.0),
        torso_orientation_task := mink.FrameTask(frame_name="torso_link", frame_type="body", position_cost=0.0, orientation_cost=1.0, lm_damping=1.0),
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

    data = configuration.data

    posture_task.set_target_from_configuration(configuration)
    pelvis_orientation_task.set_target_from_configuration(configuration)
    torso_orientation_task.set_target_from_configuration(configuration)
    for hand, foot in zip(hands, feet):
        mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
        mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
    data.mocap_pos[com_mid] = data.subtree_com[1]

    rate = RateLimiter(frequency=100.0)

    # 3. CSV 저장을 위한 헤더 정의 및 파일 오픈
    # 타임스탬프, 양손 XYZ RPY (12개 차원), 전체 qpos 관절 상태 (50개 차원)
    csv_filename = f"g1_sim_trajectory_{int(time.time())}.csv"
    csv_file = open(csv_filename, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    
    header = ['timestamp', 
              'left_hand_x', 'left_hand_y', 'left_hand_z', 'left_hand_R', 'left_hand_P', 'left_hand_Y',
              'right_hand_x', 'right_hand_y', 'right_hand_z', 'right_hand_R', 'right_hand_P', 'right_hand_Y']
    # MuJoCo의 전체 내부 관절 상태 차원(qpos_0 ~ qpos_49) 추가
    header.extend([f'qpos_{i}' for i in range(model.nq)])
    csv_writer.writerow(header)

    # 4. 사용자 입력 감지용 백그라운드 스레드 시작
    input_thread = threading.Thread(target=console_input_thread)
    input_thread.daemon = True
    input_thread.start()

    print("\n[V] 모니터링 모드 활성화 (⚠️ 실제 로봇으로 명령을 전송하지 않습니다.)")
    start_time = None

    try:
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            while viewer.is_running() and not exit_flag:
                # IK 타겟 계산 및 시뮬레이터 업데이트
                com_task.set_target(data.mocap_pos[com_mid])
                for i, (h_task, f_task) in enumerate(zip(hand_tasks, feet_tasks)):
                    f_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
                    h_task.set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))

                vel = mink.solve_ik(configuration, tasks, rate.dt, "daqp", limits=limits)
                configuration.integrate_inplace(vel, rate.dt)

                # 데이터 기록 파트
                if recording:
                    if start_time is None:
                        start_time = time.time()
                    
                    current_time = time.time() - start_time
                    
                    # 왼손/오른손 site의 현재 주소(데이터 ID) 추출
                    lh_id = model.site("left_palm").id
                    rh_id = model.site("right_palm").id
                    
                    # 왼손 XYZ 및 RPY 추출
                    lh_pos = data.site_xpos[lh_id]
                    lh_rot = data.site_xmat[lh_id].reshape(3, 3)
                    lh_rpy = mat2rpy(lh_rot)
                    
                    # 오른손 XYZ 및 RPY 추출
                    rh_pos = data.site_xpos[rh_id]
                    rh_rot = data.site_xmat[rh_id].reshape(3, 3)
                    rh_rpy = mat2rpy(rh_rot)
                    
                    # 현재 시뮬레이터 상의 전체 관절 상태 배열 (configuration.q)
                    current_qpos = configuration.q.tolist()
                    
                    # 로우 데이터 생성
                    row = [current_time] + \
                          lh_pos.tolist() + lh_rpy.tolist() + \
                          rh_pos.tolist() + rh_rpy.tolist() + \
                          current_qpos
                          
                    csv_writer.writerow(row)

                viewer.sync()
                rate.sleep()

    finally:
        csv_file.close()
        print(f"\n[V] 데이터 기록 완료 및 파일 저장 성공: {csv_filename}")
