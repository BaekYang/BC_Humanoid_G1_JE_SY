import sys
import time
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

# Unitree SDK 필수 임포트
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

_HERE = Path(__file__).parent
_XML = Path("/home/computer/mink/examples/unitree_g1/scene_table.xml")

class RealRobotInterface:
    def __init__(self):
        self.low_state = None

    def low_state_callback(self, msg: LowState_):
        self.low_state = msg

def get_mapped_qpos(low_state_msg):
    """로봇 텔레메트리를 MuJoCo 50차원 qpos로 매핑 (정기구학 연산용)"""
    q = np.zeros(50)
    q[2] = 0.76  # 골반 높이 기본값
    q[3] = 1.0   # 쿼터니언 w
    if hasattr(low_state_msg, 'imu_state') and hasattr(low_state_msg.imu_state, 'quaternion'):
        q[3:7] = low_state_msg.imu_state.quaternion
    for i in range(22):
        q[i + 7] = low_state_msg.motor_state[i].q
    for i in range(22, 29):
        q[i + 14] = low_state_msg.motor_state[i].q
    return q

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 g1_pose_capture.py [네트워크_인터페이스(예: eth0)]")
        sys.exit(1)

    # 1. SDK 초기화 및 구독자 설정
    ChannelFactoryInitialize(0, sys.argv[1])
    robot = RealRobotInterface()
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(robot.low_state_callback, 10)

    # 2. 정기구학(FK) 계산을 위한 가상 MuJoCo 모델 로드
    if not _XML.exists():
        print(f"[오류] MuJoCo 모델 파일을 찾을 수 없습니다: {_XML}")
        sys.exit(1)
        
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    print("실제 로봇으로부터 실시간 Telemetry 데이터를 수신하는 중...")
    while robot.low_state is None:
        time.sleep(0.1)
    print("[V] 로봇과 연결되었습니다.")
    print("==========================================================")
    print(">> 로봇을 가만히 세워두거나 손으로 티칭 포즈를 취해준 뒤,")
    print(">> [Enter] 키를 누르면 두 모드용 데이터가 통합 저장됩니다.")
    print("==========================================================")
    
    input("Press [Enter] to capture...")

    # 3. 데이터 캡처 타임스탬프 고정
    captured_state = robot.low_state

    # [모드 2 데이터 추출] 순수 29개 관절 값 수집
    joint_list = [float(captured_state.motor_state[i].q) for i in range(29)]

    # [모드 1 데이터 추출] MuJoCo 정기구학(FK)을 이용한 손끝 XYZRPY 계산
    sim_q = get_mapped_qpos(captured_state)
    data.qpos = sim_q
    mujoco.mj_forward(model, data)  # 현재 조인트 기반으로 링크 위치 강제 업데이트

    r_hand_idx = model.site("right_palm").id
    l_hand_idx = model.site("left_palm").id

    r_pos = data.site_xpos[r_hand_idx].tolist()
    r_mat = data.site_xmat[r_hand_idx].reshape(3, 3)
    l_pos = data.site_xpos[l_hand_idx].tolist()
    l_mat = data.site_xmat[l_hand_idx].reshape(3, 3)

    # Rotation Matrix -> RPY (Euler Angle XYZ, Degree 단위) 변환
    r_rpy = R.from_matrix(r_mat).as_euler('xyz', degrees=True).tolist()
    l_rpy = R.from_matrix(l_mat).as_euler('xyz', degrees=True).tolist()

    # 4. 하나의 통합 구조로 패킹
    unified_data = {
        "CARTESIAN": {
            "RIGHT_HAND": {"pos": r_pos, "rpy": r_rpy},
            "LEFT_HAND":  {"pos": l_pos, "rpy": l_rpy}
        },
        "JOINT": {
            "JOINT_LIST": joint_list
        }
    }

    # 5. 단일 파일 저장
    output_filename = "captured_pose.json"
    with open(output_filename, "w") as f:
        json.dump(unified_data, f, indent=4)

    print(f"\n[V] 통합 캡처 성공! '{output_filename}' 파일이 생성되었습니다.")
    print(f" - 저장된 XYZRPY (우): Pos {r_pos}, Rpy {r_rpy}")
    print(f" - 저장된 XYZRPY (좌): Pos {l_pos}, Rpy {l_rpy}")
    print(f" - 저장된 관절 개수: {len(joint_list)}개")
