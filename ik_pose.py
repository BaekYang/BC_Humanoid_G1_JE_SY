import sys
import time
import json
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# Unitree SDK 필수 임포트
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

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

if __name__ == "__main__":
    CONTROL_MODE = "ARM_SDK"  # "ARM_SDK" 또는 "FULL_BODY"
    TRAJECTORY_MODE = "CARTESIAN"  # "CARTESIAN" 또는 "JOINT"
    MOTION_DURATION = 3.0  # 전이 시간 (초)
    
    if len(sys.argv) < 2:
        print("사용법: python3 g1_trajectory_control.py [네트워크_인터페이스(예: eth0)]")
        sys.exit(1)

    # 1. 파일 데이터 로드
    json_path = Path("captured_pose.json")
    if not json_path.exists():
        print(f"[오류] 데이터 캡처본(captured_pose.json)이 없습니다.")
        sys.exit(1)
    with open(json_path, "r") as f:
        pose_data = json.load(f)

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
    total_steps = int(MOTION_DURATION / rate.dt)

    # 4. 수동 및 시각화 관련 설정 (원본 코드 방식 완벽 복원)
    if TRAJECTORY_MODE == "CARTESIAN":
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
        
        # Mocap ID 사전 추출 (원본 코드의 안전한 패턴)
        com_mid = model.body("com_target").mocapid[0]
        feet_mid = [model.body(f"{f}_target").mocapid[0] for f in feet]
        hands_mid = [model.body(f"{h}_target").mocapid[0] for h in hands]

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.sync()
        
        print("\n==========================================================")
        print(f" 현재 모드: [{TRAJECTORY_MODE}]")
        print(" >> 터미널에서 [Enter]를 누르면 시뮬레이터 내에서 '가상 프리뷰 주행'을 시작합니다.")
        print("==========================================================")
        input("Press [Enter] to start Virtual Preview...")

        # -------------------------------------------------------------
        # 🏃 [PHASE 1] 시뮬레이션 가상 주행 (Preview) Loop
        # -------------------------------------------------------------
        print("\n[프리뷰] 가상 시뮬레이션 주행 중... 창을 확인하세요.")
        
        if TRAJECTORY_MODE == "CARTESIAN":
            posture_task.set_target_from_configuration(configuration)
            pelvis_orientation_task.set_target_from_configuration(configuration)
            torso_orientation_task.set_target_from_configuration(configuration)
            
            # 발, 손 등의 마커를 원본 코드처럼 프레임 위치로 안전하게 동기화
            for hand, foot in zip(hands, feet):
                mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
                mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
            data.mocap_pos[com_mid] = data.subtree_com[1]

            # 고정 부위 타겟 주입
            com_task.set_target(data.mocap_pos[com_mid])
            for i, f_task in enumerate(feet_tasks):
                f_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))

            r_hand_idx = model.site("right_palm").id
            l_hand_idx = model.site("left_palm").id
            start_r_pos, start_r_mat = data.site_xpos[r_hand_idx].copy(), data.site_xmat[r_hand_idx].reshape(3, 3).copy()
            start_l_pos, start_l_mat = data.site_xpos[l_hand_idx].copy(), data.site_xmat[l_hand_idx].reshape(3, 3).copy()

            goal_r_pos = np.array(pose_data["CARTESIAN"]["RIGHT_HAND"]["pos"])
            goal_r_rot = R.from_euler('xyz', pose_data["CARTESIAN"]["RIGHT_HAND"]["rpy"], degrees=True)
            goal_l_pos = np.array(pose_data["CARTESIAN"]["LEFT_HAND"]["pos"])
            goal_l_rot = R.from_euler('xyz', pose_data["CARTESIAN"]["LEFT_HAND"]["rpy"], degrees=True)

            slerp_r = Slerp([0.0, MOTION_DURATION], R.from_matrix(np.array([start_r_mat, goal_r_rot.as_matrix()])))
            slerp_l = Slerp([0.0, MOTION_DURATION], R.from_matrix(np.array([start_l_mat, goal_l_rot.as_matrix()])))
            
        elif TRAJECTORY_MODE == "JOINT":
            start_joints = initial_robot_joints.copy()
            goal_joints = np.array(pose_data["JOINT"]["JOINT_LIST"])

        for step in range(total_steps):
            if not viewer.is_running(): sys.exit(0)
            current_time = step * rate.dt
            alpha = current_time / MOTION_DURATION

            if TRAJECTORY_MODE == "CARTESIAN":
                interp_r_pos = (1 - alpha) * start_r_pos + alpha * goal_r_pos
                interp_l_pos = (1 - alpha) * start_l_pos + alpha * goal_l_pos
                q_r, q_l = slerp_r(current_time).as_quat(), slerp_l(current_time).as_quat()
                
                # [핵심 수정 파트] 원본의 안전한 Mocap 강제 제어 방식으로 롤백
                data.mocap_pos[hands_mid[0]] = interp_r_pos
                data.mocap_quat[hands_mid[0]] = np.array([q_r[3], q_r[0], q_r[1], q_r[2]])
                
                data.mocap_pos[hands_mid[1]] = interp_l_pos
                data.mocap_quat[hands_mid[1]] = np.array([q_l[3], q_l[0], q_l[1], q_l[2]])

                # Mocap ID를 통한 타겟 주입 (100% 동작 보장)
                hand_tasks[0].set_target(mink.SE3.from_mocap_id(data, hands_mid[0]))
                hand_tasks[1].set_target(mink.SE3.from_mocap_id(data, hands_mid[1]))

                vel = mink.solve_ik(configuration, tasks, rate.dt, "daqp", limits=limits)
                configuration.integrate_inplace(vel, rate.dt)
                
            elif TRAJECTORY_MODE == "JOINT":
                preview_joints = (1 - alpha) * start_joints + alpha * goal_joints
                sim_q = initial_qpos.copy()
                for j in range(29):
                    sim_idx = (j + 7) if j < 22 else (j + 14)
                    sim_q[sim_idx] = preview_joints[j]
                configuration.update(sim_q)

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
        print("\n[구동] 실제 로봇 전이 시작! 안전 거리를 유지하세요.")
        
        # 모델 초기 위치로 복원
        configuration.update(initial_qpos) 
        if TRAJECTORY_MODE == "CARTESIAN":
            for hand, foot in zip(hands, feet):
                mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
                mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")

        arm_sdk_weight = 0.0

        for step in range(total_steps):
            if not viewer.is_running(): break
            current_time = step * rate.dt
            alpha = current_time / MOTION_DURATION

            cmd_joints = np.zeros(29)

            if TRAJECTORY_MODE == "CARTESIAN":
                interp_r_pos = (1 - alpha) * start_r_pos + alpha * goal_r_pos
                interp_l_pos = (1 - alpha) * start_l_pos + alpha * goal_l_pos
                q_r, q_l = slerp_r(current_time).as_quat(), slerp_l(current_time).as_quat()
                
                # 동일한 Mocap 제어 방식 적용
                data.mocap_pos[hands_mid[0]] = interp_r_pos
                data.mocap_quat[hands_mid[0]] = np.array([q_r[3], q_r[0], q_r[1], q_r[2]])
                
                data.mocap_pos[hands_mid[1]] = interp_l_pos
                data.mocap_quat[hands_mid[1]] = np.array([q_l[3], q_l[0], q_l[1], q_l[2]])

                hand_tasks[0].set_target(mink.SE3.from_mocap_id(data, hands_mid[0]))
                hand_tasks[1].set_target(mink.SE3.from_mocap_id(data, hands_mid[1]))

                vel = mink.solve_ik(configuration, tasks, rate.dt, "daqp", limits=limits)
                configuration.integrate_inplace(vel, rate.dt)
                
                for j in range(29):
                    sim_idx = (j + 7) if j < 22 else (j + 14)
                    cmd_joints[j] = configuration.q[sim_idx]
                    
            elif TRAJECTORY_MODE == "JOINT":
                cmd_joints = (1 - alpha) * start_joints + alpha * goal_joints
                sim_q = initial_qpos.copy()
                for j in range(29):
                    sim_idx = (j + 7) if j < 22 else (j + 14)
                    sim_q[sim_idx] = cmd_joints[j]
                configuration.update(sim_q)

            if arm_sdk_weight < 1.0:
                arm_sdk_weight = min(1.0, arm_sdk_weight + 0.01)

            send_robot_command(pub, low_cmd, crc, cmd_joints, CONTROL_MODE, weight=arm_sdk_weight)

            viewer.sync()
            rate.sleep()

        print("[V] 실제 로봇이 목표 자세에 도달했습니다. 자세를 유지합니다.")
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
