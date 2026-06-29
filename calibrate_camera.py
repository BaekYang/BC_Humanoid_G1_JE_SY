"""
calibrate_camera.py
===================
G1 카메라 체스보드 캘리브레이션. intrinsic(fx, fy, cx, cy)과 왜곡계수를 구해서
box_detector.py에 넣을 값을 뽑아낸다.

준비물:
  - 체스보드 인쇄물 (평평한 판에 붙여. A4에 흔한 9x6 패턴이면 됨)
  - "내부 코너" 개수를 정확히 세서 CHECKERBOARD에 넣어
    (사각형 개수가 아니라 검/흰이 만나는 교차점 개수! 10x7칸이면 내부코너는 9x6)
  - 사각형 한 변 실제 길이(미터)를 SQUARE_SIZE에 입력 (자로 재서)

순서:
  1) python calibrate_camera.py capture eth0
       -> 체스보드를 카메라 앞에서 여러 각도/거리로 보여주면서 SPACE로 캡처.
          코너가 인식되면 화면에 컬러 점이 뜸. 15~25장 정도, 각도 다양하게.
          q 누르면 종료. calib_imgs/ 폴더에 저장됨.

  2) python calibrate_camera.py calibrate
       -> calib_imgs/ 안의 사진들로 캘리브레이션 실행.
          fx, fy, cx, cy 출력 + camera_calib.npz 저장 + 재투영 오차 출력.

  3) 출력된 fx/fy/cx/cy를 box_detector.py 상단에 복붙.
     (또는 box_detector에서 np.load('camera_calib.npz')로 불러와도 됨)
"""

import cv2
import numpy as np
import sys
import os
import glob


# ============================================================================
# 설정 — 네 체스보드에 맞게!
# ============================================================================
CHECKERBOARD = (9, 6)     # 내부 코너 (가로, 세로). 사각형 칸수 -1 씩.
SQUARE_SIZE  = 0.025      # 사각형 한 변 길이 [미터]. 예: 25mm -> 0.025
IMG_DIR      = "calib_imgs"
OUT_FILE     = "camera_calib.npz"


# ============================================================================
# 1단계: 체스보드 사진 캡처
# ============================================================================
def capture(net=None):
    from box_detector import G1Camera          # 카메라 래퍼 재사용
    os.makedirs(IMG_DIR, exist_ok=True)
    cam = G1Camera(net_interface=net)

    print("SPACE=캡처(코너 인식될 때만 저장), q=종료")
    print("팁: 가까이/멀리, 좌우상하 기울여서 다양한 각도로 15~25장")
    count = 0
    while True:
        frame = cam.get_frame()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, CHECKERBOARD, corners, found)
        cv2.putText(view, f"saved: {count}  corners: {'OK' if found else 'NO'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("capture (SPACE=save, q=quit)", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and found:
            path = os.path.join(IMG_DIR, f"calib_{count:02d}.jpg")
            cv2.imwrite(path, frame)
            print("saved:", path)
            count += 1
    cv2.destroyAllWindows()
    print(f"총 {count}장 저장됨. 이제: python calibrate_camera.py calibrate")


# ============================================================================
# 2단계: 캘리브레이션 계산
# ============================================================================
def calibrate():
    # 체스보드의 3D 좌표 (z=0 평면). 실제 단위(m)로 스케일.
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints, imgpoints = [], []
    images = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if not images:
        print(f"{IMG_DIR}/ 에 사진이 없어. 먼저 capture 단계부터.")
        return

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray_shape = None
    used = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]   # (width, height)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if not found:
            print("코너 못 찾음(건너뜀):", fname)
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        used += 1

    if used < 5:
        print(f"유효 사진이 {used}장뿐. 최소 10장 이상 권장. 더 찍어.")
        return

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None)

    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]

    # 재투영 오차 (작을수록 좋음. 보통 1픽셀 미만이면 양호)
    total_err = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        total_err += cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
    mean_err = total_err / len(objpoints)

    np.savez(OUT_FILE, mtx=mtx, dist=dist, fx=fx, fy=fy, cx=cx, cy=cy,
             img_size=gray_shape)

    print("\n" + "=" * 50)
    print(f"사용한 사진: {used}장")
    print(f"재투영 오차(mean): {mean_err:.4f} px  (1.0 미만이면 양호)")
    print("=" * 50)
    print("box_detector.py 상단에 이대로 붙여넣어:\n")
    print(f"FX = {fx:.2f}")
    print(f"FY = {fy:.2f}")
    print(f"CX = {cx:.2f}")
    print(f"CY = {cy:.2f}")
    print(f"\n왜곡계수 dist = {dist.ravel().tolist()}")
    print(f"\n전체 행렬은 {OUT_FILE} 에 저장됨 (np.load로 불러쓸 수 있음).")


# ============================================================================
if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("사용법:")
        print("  python calibrate_camera.py capture [net_interface]")
        print("  python calibrate_camera.py calibrate")
        sys.exit(0)

    mode = args[0]
    if mode == "capture":
        net = args[1] if len(args) > 1 else None
        capture(net)
    elif mode == "calibrate":
        calibrate()
    else:
        print("모드는 capture 또는 calibrate")