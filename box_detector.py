"""
box_detector.py
================
Unitree G1 카메라(RGB) 기반 박스 위치 탐지 모듈.

카메라 연결(VideoClient)과 박스 탐지(OpenCV)를 합친 버전.

사용법:
  python box_detector.py --tune [net_interface]   # 박스 색 HSV 맞추기 (먼저!)
  python box_detector.py [net_interface]          # 실시간 탐지 루프
  python box_detector.py --image front_image.jpg  # 저장된 사진으로 오프라인 테스트

  net_interface: 실로봇이면 네트워크 인터페이스 이름 (예: eth0). 시뮬/생략 가능.

3D 좌표(깊이) 추정:
  깊이 카메라가 없으니 핀홀 모델로 근사.
      Z = fx * 실제폭 / 픽셀폭
      X = (u - cx) * Z / fx
      Y = (v - cy) * Z / fy
  -> 카메라 좌표계 기준. 로봇 월드 좌표로 바꾸려면 extrinsic 필요(camera_to_world).
"""

import cv2
import numpy as np
import sys


# =============================================================================
# 1. 설정값 (네 환경에 맞게 바꿔야 하는 부분)
# =============================================================================

# --- 카메라 내부 파라미터 (intrinsic) ---
# 반드시 G1 카메라 실제 값으로! 캘리브레이션 파일 있으면 그 fx,fy,cx,cy 사용.
# 없으면 아래는 640x480 추정값이라 거리 오차가 큼.
FX = 600.0
FY = 600.0
CX = 320.0
CY = 240.0

# --- 박스 실제 한 변 길이 (미터) ---
REAL_BOX_WIDTH = 0.05   # 예: 5cm -> 0.05

# --- 색상 프리셋 (--tune 으로 찾은 값을 여기 저장) ---
# H: 0~179, S: 0~255, V: 0~255
COLOR_PRESETS = {
    "red": [   # 빨강은 hue가 한 바퀴 돌아서 범위 2개
        (np.array([0,   120, 70]),  np.array([10,  255, 255])),
        (np.array([170, 120, 70]),  np.array([179, 255, 255])),
    ],
    "blue":  [(np.array([100, 120, 70]), np.array([130, 255, 255]))],
    "green": [(np.array([40,  80,  60]), np.array([80,  255, 255]))],
    "yellow":[(np.array([20,  120, 90]), np.array([35,  255, 255]))],
}

ACTIVE_COLOR = "red"   # 지금 쓸 박스 색
MIN_AREA = 300         # 이보다 작은 덩어리는 노이즈로 무시


# =============================================================================
# 2. 카메라 (Unitree VideoClient 래퍼) — 초기화는 1번, 프레임은 계속
# =============================================================================

class G1Camera:
    def __init__(self, net_interface=None):
        # SDK는 로봇 위에서만 import 되니까 여기서 lazy import
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.video.video_client import VideoClient

        if net_interface:
            ChannelFactoryInitialize(0, net_interface)
        else:
            ChannelFactoryInitialize(0)

        self.client = VideoClient()
        self.client.SetTimeout(3.0)
        self.client.Init()

    def get_frame(self):
        """BGR numpy 프레임 하나 반환. 실패 시 None."""
        code, data = self.client.GetImageSample()
        if code != 0:
            print("GetImageSample error. code:", code)
            return None
        buf = np.frombuffer(bytes(data), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# =============================================================================
# 3. 탐지 핵심 로직 (프레임 하나 -> 박스 위치)
# =============================================================================

def make_mask(frame_bgr, color_ranges):
    """BGR 프레임 -> 해당 색 마스크."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in color_ranges:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def estimate_3d(u, v, pixel_width):
    """픽셀 중심 + 픽셀 폭 -> 카메라 좌표계 3D (X,Y,Z) [m]. 핀홀 근사."""
    if pixel_width <= 0:
        return None
    Z = FX * REAL_BOX_WIDTH / pixel_width
    X = (u - CX) * Z / FX
    Y = (v - CY) * Z / FY
    return np.array([X, Y, Z], dtype=np.float32)


def detect(frame_bgr, color_ranges=None):
    """
    프레임 하나 받아서 박스 탐지.
    반환 dict: found, center_px(u,v), bbox(x,y,w,h), cam_xyz, mask
    """
    if color_ranges is None:
        color_ranges = COLOR_PRESETS[ACTIVE_COLOR]

    mask = make_mask(frame_bgr, color_ranges)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    result = {"found": False, "center_px": None, "bbox": None,
              "cam_xyz": None, "mask": mask}
    if not contours:
        return result

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return result

    x, y, w, h = cv2.boundingRect(c)
    u, v = x + w / 2.0, y + h / 2.0
    result.update(found=True, center_px=(u, v), bbox=(x, y, w, h),
                  cam_xyz=estimate_3d(u, v, w))
    return result


def draw_overlay(frame_bgr, result):
    """탐지 결과 시각화."""
    out = frame_bgr.copy()
    if not result["found"]:
        cv2.putText(out, "no box", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out
    x, y, w, h = result["bbox"]
    u, v = result["center_px"]
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(out, (int(u), int(v)), 4, (0, 0, 255), -1)
    if result["cam_xyz"] is not None:
        X, Y, Z = result["cam_xyz"]
        cv2.putText(out, f"X={X:+.3f} Y={Y:+.3f} Z={Z:.3f} m", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


# =============================================================================
# 4. HSV 튜닝 모드 (박스 색 맞추기) — 제일 먼저 이걸로
# =============================================================================

def tune_hsv(get_frame):
    """
    트랙바로 HSV 범위 실시간 조절. 마스크에 박스만 하얗게 나오게 맞춘 뒤
    q 누르면 콘솔에 lo/hi 찍힘 -> COLOR_PRESETS에 복사.
    get_frame: 호출하면 BGR 프레임 주는 함수
    """
    win = "tune (q to quit)"
    cv2.namedWindow(win)
    bars = [("Hmin", 0, 179), ("Hmax", 179, 179), ("Smin", 80, 255),
            ("Smax", 255, 255), ("Vmin", 60, 255), ("Vmax", 255, 255)]
    for name, val, mx in bars:
        cv2.createTrackbar(name, win, val, mx, lambda x: None)

    while True:
        frame = get_frame()
        if frame is None:
            continue
        g = lambda n: cv2.getTrackbarPos(n, win)
        lo = np.array([g("Hmin"), g("Smin"), g("Vmin")])
        hi = np.array([g("Hmax"), g("Smax"), g("Vmax")])
        mask = make_mask(frame, [(lo, hi)])
        view = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow(win, view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("lo =", lo.tolist())
            print("hi =", hi.tolist())
            break
    cv2.destroyAllWindows()


# =============================================================================
# 5. 카메라 좌표계 -> 로봇 월드 좌표계 (extrinsic 있으면 채우기)
# =============================================================================

def camera_to_world(cam_xyz):
    """world = R @ cam_xyz + t. 카메라 장착 위치/자세 알면 채워."""
    R = np.eye(3)      # TODO: 실제 회전행렬
    t = np.zeros(3)    # TODO: 실제 이동벡터
    return R @ cam_xyz + t


# =============================================================================
# 6. 메인
# =============================================================================

def run_loop(get_frame):
    print("탐지 시작. 창에서 q 누르면 종료.")
    while True:
        frame = get_frame()
        if frame is None:
            continue
        result = detect(frame)
        if result["found"]:
            print("cam_xyz:", result["cam_xyz"])
            # world = camera_to_world(result["cam_xyz"])  # extrinsic 있으면
        cv2.imshow("box detect (q to quit)", draw_overlay(frame, result))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def main():
    args = sys.argv[1:]
    tune = "--tune" in args

    # --image 모드: 저장된 사진 한 장으로 오프라인 테스트 (카메라 불필요)
    if "--image" in args:
        path = args[args.index("--image") + 1]
        img = cv2.imread(path)
        if img is None:
            print("이미지 못 읽음:", path); return
        if tune:
            tune_hsv(lambda: img.copy())
        else:
            cv2.imshow("result (q to quit)", draw_overlay(img, detect(img)))
            print("cam_xyz:", detect(img)["cam_xyz"])
            while cv2.waitKey(0) & 0xFF != ord("q"):
                pass
            cv2.destroyAllWindows()
        return

    # 실제 카메라 모드: --tune/--image 외의 첫 인자를 net interface로 사용
    net = next((a for a in args if not a.startswith("--")), None)
    cam = G1Camera(net_interface=net)

    if tune:
        tune_hsv(cam.get_frame)
    else:
        run_loop(cam.get_frame)


if __name__ == "__main__":
    main()