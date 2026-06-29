"""
box_detector.py
================
Unitree G1 카메라(RGB) 기반 박스 비전 모듈. (탐지 + 비전 게이트 통합)

이 파일 하나에 다 들어있음:
  - G1Camera   : SDK 카메라 연결 래퍼 (초기화 1번, 프레임 계속)
  - detect()   : 박스 위치 탐지 (bbox, 픽셀중심, 3D 추정)  ← 풀 기능
  - VisionGate : "시야에 박스 있냐 없냐"만 판정하는 가벼운 게이트  ← policy_to_csv용

실행 모드:
  python box_detector.py --tune [net]          # 박스 색 HSV 맞추기 (제일 먼저!)
  python box_detector.py --gate [net]          # 게이트 임계값(MIN_PIXELS) 맞추기
  python box_detector.py [net]                 # 박스 탐지 루프 (bbox + 3D 표시)
  python box_detector.py --image front.jpg     # 저장된 사진으로 오프라인 테스트

  net: 실로봇이면 네트워크 인터페이스 (예: eth0). 시뮬/생략 가능.

policy_to_csv.py 등 메인 파일에서 게이트로 쓰는 법:
    from box_detector import VisionGate
    gate = VisionGate(net="eth0")
    if gate.object_visible():     # ← 박스 보일 때만 True
        run_policy_and_save_csv()
"""

import cv2
import numpy as np
import sys


# =============================================================================
# 1. 설정값 (네 환경에 맞게)
# =============================================================================

# --- 카메라 내부 파라미터 (intrinsic) — 3D 추정(detect)에만 쓰임. 게이트는 무관 ---
# 캘리브레이션 값 있으면 넣고, 없으면 아래는 640x480 추정값(거리 오차 큼).
FX = 600.0
FY = 600.0
CX = 320.0
CY = 240.0

# --- 박스 실제 한 변 길이 (미터) — 3D 추정에만 쓰임 ---
REAL_BOX_WIDTH = 0.05

# --- 색상 프리셋 (--tune 으로 찾은 값 저장) ---
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
MIN_AREA = 300         # detect(): 이보다 작은 덩어리는 노이즈로 무시
MIN_PIXELS = 500       # VisionGate: 이 픽셀 수 이상이면 '박스 보임'으로 판정


# =============================================================================
# 2. 카메라 (Unitree VideoClient 래퍼)
# =============================================================================

class G1Camera:
    def __init__(self, net_interface=None):
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
        """BGR numpy 프레임 반환. 실패 시 None."""
        code, data = self.client.GetImageSample()
        if code != 0:
            print("GetImageSample error. code:", code)
            return None
        buf = np.frombuffer(bytes(data), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# =============================================================================
# 3. 공통: 색 마스크
# =============================================================================

def make_mask(frame_bgr, color_ranges):
    """BGR 프레임 -> 해당 색 마스크(이진)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in color_ranges:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# =============================================================================
# 4. 비전 게이트 — "박스 있냐 없냐"만 (policy_to_csv용)
# =============================================================================

class VisionGate:
    """
    카메라 시야에 박스가 있으면 True, 없으면 False.
    칼리브레이션/3D 불필요. 박스 색 픽셀 수만 센다.
    """
    def __init__(self, net=None, color=ACTIVE_COLOR, min_pixels=MIN_PIXELS):
        self.cam = G1Camera(net_interface=net)
        self.ranges = COLOR_PRESETS[color]
        self.min_pixels = min_pixels

    def object_visible(self, verbose=False):
        frame = self.cam.get_frame()
        if frame is None:
            return False                       # 프레임 못 받으면 안전하게 '안 보임'
        count = int(cv2.countNonZero(make_mask(frame, self.ranges)))
        if verbose:
            print(f"색 픽셀 수: {count}  ->  "
                  f"{'보임' if count >= self.min_pixels else '안보임'}")
        return count >= self.min_pixels


# =============================================================================
# 5. 풀 탐지 — 위치/3D까지 (게이트보다 무거움, 필요할 때만)
# =============================================================================

def estimate_3d(u, v, pixel_width):
    """픽셀 중심 + 픽셀 폭 -> 카메라 좌표계 3D (X,Y,Z)[m]. 핀홀 근사."""
    if pixel_width <= 0:
        return None
    Z = FX * REAL_BOX_WIDTH / pixel_width
    X = (u - CX) * Z / FX
    Y = (v - CY) * Z / FY
    return np.array([X, Y, Z], dtype=np.float32)


def detect(frame_bgr, color_ranges=None):
    """프레임 -> dict(found, center_px, bbox, cam_xyz, mask)."""
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


def camera_to_world(cam_xyz):
    """world = R @ cam_xyz + t. extrinsic 알면 채워."""
    R = np.eye(3)      # TODO
    t = np.zeros(3)    # TODO
    return R @ cam_xyz + t


# =============================================================================
# 6. HSV 튜닝 모드
# =============================================================================

def tune_hsv(get_frame):
    """트랙바로 HSV 조절. 박스만 하얗게 맞추고 q -> 콘솔 lo/hi를 COLOR_PRESETS에."""
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
            print("lo =", lo.tolist()); print("hi =", hi.tolist())
            break
    cv2.destroyAllWindows()


# =============================================================================
# 7. 메인
# =============================================================================

def _detect_loop(get_frame):
    print("탐지 시작. q 종료.")
    while True:
        frame = get_frame()
        if frame is None:
            continue
        result = detect(frame)
        if result["found"]:
            print("cam_xyz:", result["cam_xyz"])
        cv2.imshow("box detect (q to quit)", draw_overlay(frame, result))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def _gate_loop(net):
    """MIN_PIXELS 맞추기: 박스 넣었다 뺐다 하며 픽셀 수 관찰."""
    gate = VisionGate(net=net)
    print("박스를 넣었다 뺐다 하며 픽셀 수 관찰. Ctrl+C 종료.")
    print("있을 때/없을 때 숫자 사이 값으로 MIN_PIXELS를 정해.\n")
    try:
        while True:
            gate.object_visible(verbose=True)
            cv2.waitKey(100)
    except KeyboardInterrupt:
        print("\n종료.")


def main():
    args = sys.argv[1:]

    # 오프라인: 저장된 사진으로 테스트 (카메라 불필요)
    if "--image" in args:
        path = args[args.index("--image") + 1]
        img = cv2.imread(path)
        if img is None:
            print("이미지 못 읽음:", path); return
        if "--tune" in args:
            tune_hsv(lambda: img.copy())
        else:
            r = detect(img)
            print("cam_xyz:", r["cam_xyz"])
            cv2.imshow("result (q to quit)", draw_overlay(img, r))
            while cv2.waitKey(0) & 0xFF != ord("q"):
                pass
            cv2.destroyAllWindows()
        return

    net = next((a for a in args if not a.startswith("--")), None)

    if "--tune" in args:
        cam = G1Camera(net_interface=net)
        tune_hsv(cam.get_frame)
    elif "--gate" in args:
        _gate_loop(net)
    else:
        cam = G1Camera(net_interface=net)
        _detect_loop(cam.get_frame)


if __name__ == "__main__":
    main()
