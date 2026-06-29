# 장애물(기둥) 회피 — Obstacle Avoidance

기존 pick-and-place BC 파이프라인에 **장애물(기둥) 회피**를 추가한 확장.
정책이 기둥 위치를 관측(obs)하고 **기둥이 경로를 막으면 우회, 없으면 직진**하도록 학습한다.

## 파이프라인

```
g1_collect_demos_obs.py   →   train_bc_obs.py   →   eval_bc_obs.py   →   (배포)
  (기둥 배치+우회 데모)         (16D obs 학습)         (회피 검증)        policy_to_csv + ik_traj_grip
        demos_obs/            bc_policy_obs.pt                          (충돌회피 제약 필요)
                             bc_stats_obs.npz
```

기존 무장애물 파일(`g1_collect_demos.py`, `train_bc.py`, `eval_bc.py`)은 **그대로 보존**.
장애물 버전은 전부 `_obs` 접미사로 분리.

## 씬

`scene_g1_pickplace.xml` 을 복사해 `scene_g1_pickplace_obs.xml` 생성 후, `worldbody`에 기둥 추가:

```xml
<body name="obstacle" pos="0.5 0.05 0.935" mocap="true">
    <geom name="obstacle_geom" type="cylinder" size="0.03 0.10" rgba="0.9 0.6 0.1 1"/>
</body>
```

- **mocap=true** → 런타임에 코드가 위치 이동(랜덤 배치). keyframe qpos 안 건드림.
- cylinder `size="0.03 0.10"` = 반지름 3cm, 반높이 10cm → **높이 20cm**.
- 바닥이 테이블 윗면(z=0.835)에 닿게 중심 z=0.935.

## 핵심 설계 — 두 종류의 회피

| 회피 대상 | 담당 | 이유 |
|---|---|---|
| **손 경로** (어디로 돌지) | 정책이 학습 (detour 웨이포인트) | 손 위치는 정책이 출력 → BC로 학습 가능 |
| **전완/팔 링크** (팔이 기둥 안 뚫게) | IK 충돌회피 (mink) | 같은 손 위치라도 팔 자세는 여러 개 → 정책이 제어 불가, IK가 처리 |

> **BC 일관성:** 손 우회는 정책이 배우고, 전완 회피는 IK가 한다. IK 충돌회피는
> 학습(collect)·검증(eval)·배포(policy_to_csv·ik_traj_grip)에 **동일하게** 들어가야 한다.

## 1. 기둥 배치 규칙 (g1_collect_demos_obs.py)

- 박스/목표를 **y축 반대편**에 배치 (박스 아래 / 목표 위, 50% 확률로 뒤바뀜) → 기둥이 그 사이에 오게.
- 기둥 = 박스–목표 **중점 + 흔들림(±2cm)**.
- 박스–목표 최소거리 `MIN_SEP_OBS=0.14` (기둥 + 우회 여유 확보 위해 키움).
- 박스·목표·기둥 모두 **타원 도달범위 안**이어야 함.

## 2. 손 우회 경유점 (detour)

`lift`(들기)와 `move`(이동) 사이에 **detour 웨이포인트**를 삽입 → 웨이포인트 **8개 → 9개**.

- 박스→목표 직선의 **수직 방향** 양옆으로 기둥에서 `OBS_R + CLEAR` 만큼 비낀 두 후보.
- **몸쪽(왼쪽 어깨 ≈ (0.0, 0.10)에 가까운) 쪽 우선** 선택 (왼팔이 편한 방향). 단 도달 가능(타원 안)할 때만.
- 둘 다 도달 불가면 그 에피소드 스킵.

## 3. IK 충돌회피 (전완)

mink `CollisionAvoidanceLimit` 로 **왼팔 전완 캡슐 ↔ 기둥**이 일정 거리 안으로 못 들어오게:

```python
mink.CollisionAvoidanceLimit(
    model,
    geom_pairs=[(["left_hand_collision"], ["obstacle_geom"])],
    minimum_distance_from_collisions=0.02,   # 2cm 여유
    collision_detection_distance=0.10,
)
```

→ IK가 팔 자세를 통째로 비껴서 풀어, 전완도 기둥을 안 뚫는다.
→ 충돌회피가 전완을 처리하므로 손 우회(`CLEAR`)는 작게(6cm)만 줘도 됨.

## 4. obs / 액션 (train_bc_obs.py)

- **obs 16D** = 손위치(3) + 블록위치(3) + 목표(3) + 블록초기(3) + 블록크기(1) + **기둥위치(3)** + waypoint_idx 임베딩
- **action 4D** = 목표 손위치(3) + grasp(1) — 기존과 동일
- **N_WAYPOINTS = 9** (detour 포함)
- detour 목표 좌표는 npz의 `detour_xpos` 사용 (재구성 오차 방지)
- 결과: `bc_policy_obs.pt`, `bc_stats_obs.npz`

## 5. 일반화 — 기둥 없는 경우 (직진)

"기둥 있으면 피하고 **없으면 직진**"을 배우게 하려고, 데모의 일부를 **기둥 없음**으로 섞음.

- `NO_OBS_RATIO = 0.30` → **30%는 기둥을 경로 밖 먼 곳**(`OBS_FAR_XY=(0.40, -0.30)`)으로 치움.
- 이때 detour = 박스–목표 **직선 중점**(우회 안 함, 직진 경유점) → 웨이포인트는 **항상 9개로 고정**.
- 정책은 "기둥이 경로 근처면 우회 / 멀면 직진"을 **기둥 위치(obs)로 구분**해 학습.

## 파라미터 정리 (현재 검증값)

| 항목 | 값 | 의미 |
|---|---|---|
| `WS_X` / `WS_Y` | (0.25, 0.55) / (0.00, 0.40) | 넓힌 워크스페이스 |
| 타원 `CX,CY / A,B` | 0.40, 0.16 / 0.12, 0.17 | 도달 범위(2차 제한) |
| `OBS_R` | 0.03 | 기둥 반지름 |
| `OBS_Z` | 0.935 | 기둥 중심 높이(20cm 기둥) |
| `MIN_SEP_OBS` | 0.14 | 박스–목표 최소거리 |
| `OBS_JITTER` | 0.02 | 기둥 위치 흔들림 |
| `CLEAR` | 0.06 | 손 우회 여유(전완은 IK가 처리) |
| `NO_OBS_RATIO` | 0.30 | 기둥 없음(직진) 비율 |
| `OBS_FAR_XY` | (0.40, −0.30) | 기둥 치워둘 먼 위치 |
| 충돌회피 min_dist | 0.02 | 전완–기둥 최소거리 |
| 충돌회피 detect_dist | 0.10 | 충돌 감지 거리 |
| `N_WAYPOINTS` | 9 | approach,descend,grasp,lift,**detour**,move,place,release,retreat |
| obs 차원 | 16 | 기존 13 + 기둥위치 3 |

## 실행

```bat
REM 노트북(윈도우). XML 경로는 scene_g1_pickplace_obs.xml (윈도우 raw string)
cd C:\Users\USER\g1bc

REM 1) 데모 수집 (기둥 있음 70% + 없음 30%)
rmdir /s /q demos_obs
python g1_collect_demos_obs.py --n 100
python g1_collect_demos_obs.py --n 5 --view     REM 눈으로 우회 확인

REM 2) 학습 → bc_policy_obs.pt, bc_stats_obs.npz
python train_bc_obs.py --epochs 200

REM 3) 검증 (기둥 회피 + 손-기둥 최소거리 표시)
python eval_bc_obs.py --n 1 --view
python eval_bc_obs.py --n 50
```

eval 출력의 `손-기둥최소` 가 기둥반경(0.03)보다 충분히 크면(예 0.06+) 회피 성공.

## 남은 작업 (배포)

- `policy_to_csv.py` → `_obs` 버전: obs 16D, 기둥 위치 인자(`--ox --oy`), 충돌회피 제약 추가.
- `ik_traj_grip.py`: 실로봇 IK에도 동일 충돌회피 제약 (BC 일관성).
- (선택) eval에도 기둥 없는 30% 케이스 넣어 "직진" 일반화까지 검증.

## 알려진 한계

- **시뮬 파지 = 부착(kinematic).** 실제 손가락 파지는 실물 Dex3에서만 검증됨.
- 시뮬은 **다리/허리 자가균형**을 가정 → 실배포 시 손끝 위치 미세 드리프트 가능(차후 실시간 상태 피드백으로 개선 여지).
- 기둥은 **고정 높이 20cm / 반지름 3cm 1개**만 가정. 여러 개·다양한 크기는 미지원.
