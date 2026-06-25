# Unitree G1 — Behavioral Cloning Pick-and-Place (왼팔 + Dex3 왼손)

Unitree **G1-EDU** 휴머노이드가 **임의 위치의 박스를 집어 임의의 목표 지점에 놓도록** Behavioral Cloning(BC)으로 학습하고, MuJoCo 시뮬레이션에서 검증한 뒤 실로봇(왼팔 + Dex3 왼손)으로 배포하는 프로젝트다. 전신 IK는 [mink](https://github.com/kevinzakka/mink), 물리는 [MuJoCo](https://github.com/google-deepmind/mujoco), 비전 없음(상태 기반).

# Video - https://youtu.be/w3wSNIc-Rjc?si=9RLIqycnpfSdSja4 - 켜는법

## 파이프라인

```
g1_collect_demos.py   →   train_bc.py   →   eval_bc.py   →   policy_to_csv.py   →   ik_traj_grip.py
 (전문가 데이터)          (BC 학습)        (시뮬 평가)       (궤적 CSV 추출)        (실로봇 배포: 팔+손)
       │                    │                 │                  │                      │
       └──── mink 전신 IK / scene_g1_pickplace.xml / bc_policy.pt+bc_stats.npz 공유 ────┘
```

BC의 기본 레시피(① 전문가 데이터 수집 → ② 지도학습 → ③ 일반화 평가 → ④ 배포)를 그대로 파일로 나눈 구조로, 각 단계를 독립적으로 검증·교체할 수 있다. G1은 정책 출력을 **CSV로 추출**한 뒤 실로봇 재생기(`ik_traj_grip.py`)가 전신 IK로 균형을 잡으며 재생하는 점이 특징이다.

> **경로 표기 규칙** — 절대경로 대신 두 기준점으로 표기한다:
> - `PROJECT_DIR` = 작업 폴더 (예: `~/unitree_sdk2_python/example/my/IK`) — 스크립트·모델·CSV가 들어있는 폴더
> - `MINK_DIR` = mink 씬 폴더 (예: `~/mink/examples/unitree_g1`) — `.xml` 씬·assets이 들어있는 폴더
> 각 스크립트 상단 `XML = "..."` 한 줄만 본인 `MINK_DIR` 경로로 바꾸면 실행할 수 있다.

### 📂 디렉토리 구조

```
PROJECT_DIR/                      # 작업 폴더 (IK)
├── g1_collect_demos.py           # [필수] 스크립트 전문가 데모 수집 (시뮬)
├── train_bc.py                   # [필수] demos/ → 정책 학습
├── eval_bc.py                    # [필수] 정책 시뮬 평가 (성공률)
├── policy_to_csv.py              # [필수] 학습 정책 → arm_traj.csv 추출 (시뮬)
├── ik_traj_grip.py               # [필수] CSV → 실로봇 재생 (팔 + Dex3 손)  ★최종 실행
├── ik_traj.py                    # CSV → 실로봇 재생 (팔만, 손 없음)
├── dex3_gripper.py               # [필수] Dex3 손 제어 클래스 (ik_traj_grip이 import)
├── test_grip.py                  # 손 단독 열기/닫기 테스트 (로봇)
├── bc_policy.pt                  # [필수·학습산출] 정책 가중치
├── bc_stats.npz                  # [필수·학습산출] obs 정규화 통계 (평균/표준편차)
├── arm_traj.csv                  # (생성물) policy_to_csv 출력 — ik_traj, ik_traj_grip가 읽음. 별도의 추가 파일은 traj_zip에 모아놓음
├── demos/                        # (생성물) g1_collect_demos 출력
│   ├── demo_0000.npz
│   └── ...
├── requirements.txt
└── README.md

MINK_DIR/                         # mink 씬 폴더 (별도 설치)
├── scene_g1_pickplace.xml        # [필수] BC 시뮬용 씬 (블록 포함) — collect/train·eval/policy_to_csv
├── scene_table.xml               # [필수] 로봇 재생용 씬 — ik_traj/ik_traj_grip
├── g1_with_hands.xml             # 로봇 모델 (손 포함)
├── g1.xml · scene.xml
└── assets/                       # 메시·텍스처
```

> `★ ik_traj_grip.py`가 최종 실행 파일. `dex3_gripper.py`와 **같은 폴더**에 있어야 import 된다.

## 목차

- [1. 코드 설명 — 설계 의도](#1-코드-설명--설계-의도)
  - [g1_collect_demos.py](#g1_collect_demospy--전문가-데이터-생성)
  - [train_bc.py](#train_bcpy--bc-정책과-학습)
  - [policy_to_csv.py](#policy_to_csvpy--궤적-csv-추출)
  - [ik_traj_grip.py](#ik_traj_grippy--실로봇-배포)
- [2. 파라미터 — 최적값](#2-파라미터--최적값)
- [3. 코드 실행 및 사용법](#3-코드-실행-및-사용법)

---

# 1. 코드 설명 — 설계 의도

## g1_collect_demos.py — 전문가 데이터 생성

BC의 **입력 데이터 D = {(상태, 행동)}** 를 만드는 가장 중요한 파일이다. 전문가 궤적을 **mink 전신 IK로 자체 생성**한다(사람 시범 대신 → 결정적·일관·대량 생성).

### waypoint & dense 보간
작업을 **8개 Cartesian waypoint**로 표현하고, 각 waypoint를 mink IK로 풀어 전신 관절각을 얻는다. 사이는 보간 + 정착으로 촘촘히(dense) 채운다.

| # | waypoint | 손 위치 | grip |
|---|---|---|---|
| 0 | approach | 박스 위(+0.12) | 0 (열림) |
| 1 | descend | 박스 위(+0.02) | 0 |
| 2 | grasp | 박스 위(+0.02) | 1 (닫힘) |
| 3 | lift | 박스 위(+0.12) | 1 |
| 4 | move | 목표 위(+0.12) | 1 |
| 5 | place | 목표 위(+0.02) | 1 |
| 6 | release | 목표 위(+0.02) | 0 |
| 7 | retreat | 목표 위(+0.12) | 0 |

- waypoint당 `STEPS_PER_WP`(=80) 보간 + `SETTLE_STEPS`(=50) 정착 → episode당 신호가 풍부 → rollout 중 분포 이탈(covariate shift)에 덜 취약.
- **grasp 게이팅**: `grasp=1`인 동안 블록을 손에 **kinematic attach**(블록 qpos = 손 위치)하여 시뮬에서 집기를 모델링한다.
  - **소명**: grasp는 연속이 아니라 **불연속 사건**이라, 보간 중 깜빡이지 않도록 waypoint 도착 시점에만 0/1로 전환해 타이밍을 명확히 고정한다.
### 무엇을 변주하고 무엇을 고정했나
- **박스 x,y → 랜덤** (`WS_X=[0.38,0.45]`, `WS_Y=[0.07,0.23]` 균등).
  - **소명**: BC의 목표는 "한 궤적 외우기"가 아니라 **"박스가 어디 있든 대응하는 매핑"** 을 배우는 것이다. 박스 위치를 무작위로 다양하게 줘야 상태공간이 골고루 덮이고, 학습에 없던 위치에도 일반화한다.
- **목표(goal) x,y → 랜덤** (같은 범위, 박스와 `MIN_SEP=0.08` 이상 떨어지게).
  - **소명**: 본 과제는 박스·목표가 **둘 다 랜덤**이다(Lite6는 목표 고정). 목표 좌표를 obs에 넣어, 어떤 목표가 와도 대응하는 매핑을 배우게 한다.
- **박스 size → 랜덤** (`[0.020,0.035]` half-extent).
  - **소명**: 다양한 박스 두께에 손 높이·grasp를 맞추도록 하는 크기 일반화.
- **박스 회전 → 고정** (쿼터니언 `[1,0,0,0]`).
  - **소명**: 위에서 수직으로 내려와 집는(top-down) 방식이라 박스 회전이 결과에 거의 영향이 없다. 불필요한 변수를 줄여 **학습 난이도를 낮추고 데이터 일관성을 높이기** 위함이다.
- **박스 z → 고정** (`TABLE_TOP_Z + size`, 테이블 높이).
  - **소명**: z는 테이블 높이로 항상 일정 → 불필요 변수 제거.

### 저장
성공(`SUCCESS_DIST`=5cm 이내)한 episode만 `demos/demo_XXXX.npz`로 저장. 키: `hand_pos, hand_quat, block_pos, block_quat, target_pos, block_size, waypoint_idx, waypoint_grip, act_hand_target, act_grasp, block_init_xpos, target_xpos`.

## train_bc.py — BC 정책과 학습

### 관측(obs) / 행동(action) 설계
- **obs 13차원** = 손 위치(3) + 블록 위치(3) + 목표 위치(3) + **블록 초기위치(3)** + 박스 size(1). 여기에 **waypoint_idx 임베딩**(8단계 → 16차원)을 이어붙인다.
  - **블록 초기위치를 넣는 이유**: 잡은 뒤엔 블록=손이 되어 현재 블록위치만으론 "원래 어디서 집었는지"를 모른다 → 초기위치를 같이 줘야 목표까지의 매핑이 일관된다.
- **action 4차원** = 현재 waypoint의 **목표 손위치(절대좌표 3)** + grasp(1).

### 네트워크 구조 (Residual 사용)
```
Linear(13+16 → 256) + LayerNorm + SiLU
  → ResBlock(256) × 3        # 각 블록: LayerNorm→Linear→SiLU→Linear + skip
  → LayerNorm + Linear(256 → 4)
```
- waypoint를 임베딩으로 조건화하고 단계가 8개로 나뉘어 함수가 구간별로 꺾이므로, 얕은 MLP보다 **ResBlock 3개**가 안정적으로 수렴했다(Lite6의 단순 MLP와 다른 점).

### 학습 작동 방식
1. `demos/*.npz`를 모아 obs/action 텐서로 만들고 **정규화**(평균0/표준편차1, `std+1e-6`).
2. `DataLoader`(batch 256, shuffle) → 정책 예측 → **HuberLoss(δ=1.0)** → 역전파.
3. **AdamW**(lr 1e-3, weight_decay 1e-4) + **CosineAnnealing** 스케줄, `epochs` 200.
4. 끝나면 `bc_policy.pt`(가중치)와 `bc_stats.npz`(정규화 통계)를 **각각** 저장.

> **정규화 통계는 `bc_stats.npz`에 저장된다.** 추론(eval/policy_to_csv)에서 이 파일을 읽어 **학습과 동일 기준으로 obs를 정규화**한다. → 배포에는 `bc_policy.pt`**와** `bc_stats.npz`가 **둘 다** 있어야 한다. (Lite6와 달리 CSV 재계산이 아니라 npz로 저장하는 방식.)

> **손실 설계**: 손 위치는 연속값이라 회귀(Huber는 이상치에 강한 MSE 변형). grasp(0/1)도 같은 회귀로 묶되, 배포 시에는 정책의 grasp 출력을 **0.5 임계값**으로 이진화해 손을 닫/연다.

## policy_to_csv.py — 궤적 CSV 추출

학습된 정책을 **한 config**(`--bx --by --tx --ty --size`)에 대해 closed-loop로 롤아웃하여, 왼손 Cartesian 궤적을 실로봇 재생기 포맷(`arm_traj.csv`)으로 추출한다. CSV 열: `[t, LH xyz, LH rpy, RH xyz, RH rpy, grasp]`.

### 후처리 (재학습 불필요 — 전부 이 파일 안에서)
- **차렷 시작/복귀**: 좌·우 팔 qpos를 0으로 둔 초기자세에서 출발해 pick&place 후 복귀.
- **박스 충돌 회피**: 박스 위 `LIFTZ`까지 **수직 상승** 후 고공으로 이동, 저공 rest 자세는 스킵.
- **오른팔 고정**: 오른손 IK 타깃을 차렷에 묶어 오른팔이 들리지 않게.
- **resample**(`--maxstep` 0.005m, 위치 lerp + 방향 slerp)로 전 구간을 dense화 → "순간이동" 방지.
- **grasp 운반**: 정책의 grasp 출력을 CSV **14번째 열**로 계단식(불연속)으로 기록 → 재생기가 그 열로 손을 구동.

> 왜 CSV를 경유하나: 실로봇 재생기(`ik_traj`)가 양손 Cartesian CSV를 받아 **전신 IK로 균형을 잡으며** 재생하는 구조라, 정책 출력을 그 포맷으로 변환하는 다리 역할이다.

### 악수(handshake) 파지 자세 — `policy_to_csv.py`
손바닥이 정면을 보고 손가락이 수평으로 박스를 향하는 자세. **이 값으로 실물 검증 완료.**

| 인자 | 값 | 의미 |
|---|---|---|
| `--grip-rpy` | `0 0 0` | 악수 손목 방향 (씬 기본 방향이 이미 악수라 0,0,0) |
| `--palm-offset` | `0.04` | 손목→손바닥 중앙 4cm. 손목을 정면 축으로 빼서 손바닥 중앙이 박스에 닿게 |
| `--palm-axis` | `x` | 손 정면(손목→손바닥) 로컬축 |
| `--repeat` | `1` | 재생기(`ik_traj_grip.py`)가 자체 6배 슬로우를 주므로 1 유지 |

### 실행 예시
```bash
# 1) 노트북: CSV 생성 (좌표는 워크스페이스 안에서 지정)
python policy_to_csv.py --bx 0.38 --by 0.12 --tx 0.43 --ty 0.20 \
    --grip-rpy 0 0 0 --palm-offset 0.04 --repeat 1

# 2) 우분투(로봇): 재생 (뷰어 GL 문제 시 unset)
unset MUJOCO_GL
python3 ik_traj_grip.py $NET arm_traj.csv
```

### 재생 속도 / 파지 세기 — `ik_traj_grip.py`
| 항목 | 값 | 의미 |
|---|---|---|
| `SLOWDOWN_FACTOR` | 6.0 | 클수록 느리고 부드러움 (8~12 권장 가능) |
| `GRASP_EXTENT` | 1.0 | 손 쥐는 정도 0~1 (실물에 물체 있으면 0.6~0.8 튜닝) |
| `GRASP_STRENGTH` | 2.0 | 쥐는 힘(kp) |
| 종료 동작 | 차렷 복귀 후 2초 유지 → 자동 정지 | Ctrl+C 불필요 |

### ⚠️ 안전 주의
- **좌표는 타원 안쪽 권장.** 가장자리로 갈수록 팔이 과도하게 접히거나 손목 방향(악수)이 깨질 수 있음 (IK가 위치 우선이라 방향 희생).
- **시뮬은 self-collision(링크 겹침)을 검사하지 않음.** 실물에선 팔이 몸/어깨에 닿을 수 있으니, 새 좌표는 `--view`로 자세 확인 후 사용.
- 실물 첫 구동은 **천천히(`SLOWDOWN_FACTOR` 크게) + E-stop 준비 + 동료와 함께.**

## ik_traj_grip.py — 실로봇 배포

`arm_traj.csv`를 읽어 **ARM_SDK**(`rt/arm_sdk`)로 왼팔(+허리)을 재생한다(다리는 자가 균형). Dex3 왼손을 같이 구동해 실제 집고-놓기를 수행한다.

- **내장 스무딩**: `resample_trajectory`(6배 보간) + `prep`(현재자세→CSV 첫 프레임까지 2초 S커브) → 시작 튐 방지. **CSV는 `--repeat 1`로 뽑아야** 6배와 겹쳐 과도하게 느려지지 않는다.
- **손 제어**: `Dex3Gripper(hand_side="L", init_network=False)`로 재생기가 이미 연 채널을 공유. grasp `0→1`에서 `close()`, `1→0`에서 `open()`. 진입 구간은 grasp 0(벌림).
- **2단계 실행**: 1차 `[Enter]` = 시뮬 프리뷰(팔만, 손은 안 움직이고 grasp 시점만 출력), 2차 `[Enter]` = 실제 로봇(팔+손). 종료 시 손 먼저 안전종료 후 팔 댐핑.
- **하이브리드 관점**: 정책이 접근·집기 위치까지 전 궤적을 만들고(BC가 잘하는 적응), grasp는 위치가 아니라 **정책 신호(열)로 게이팅**한다. 단 실제로 쥐려면 손이 내려가는 자리에 **물체가 물리적으로 있어야** 한다(없으면 공중 여닫힘만 확인).

> **안전**: ARM_SDK weight를 0→1로 서서히 올려(매 프레임 +0.01) 초기 충격을 줄이고, 종료 시 weight를 1→0으로 내려 제어권을 안전하게 이양한다.

---

# 2. 파라미터 — 최적값

### g1_collect_demos.py (전문가 데이터 생성)

| 파라미터 | 값 | 의미 / 왜 이 값 |
|---|---|---|
| `--n` | 100 | 수집할 **성공** episode 수 (`--n`으로 조절) |
| `STEPS_PER_WP` | 80 | waypoint 사이 보간 스텝 |
| `SETTLE_STEPS` | 50 | waypoint 도착 후 정착 스텝 |
| `APPROACH_H` | 0.12 | 박스/목표 **위**로 띄우는 높이[m] |
| `GRASP_OFF` | 0.02 | 집기/놓기 시 박스 위 오프셋[m] |
| `TABLE_TOP_Z` | 0.835 | 테이블 상단 z[m] |
| `BLOCK_SIZE` | 0.025 | 기본 블록 half-extent[m] |
| `BLOCK_SIZE_RANGE` | (0.020, 0.035) | 블록 크기 랜덤 범위 |
| `WS_X` / `WS_Y` | (0.38,0.45) / (0.07,0.23) | 박스·목표 랜덤 범위[m] (도달 가능 영역) |
| `MIN_SEP` | 0.08 | 박스-목표 최소 거리[m] (의미있는 이동 보장) |
| `SUCCESS_DIST` | 0.05 | 성공 판정(목표 5cm 이내) |

### train_bc.py (정책 & 학습)

| 파라미터 | 값 | 비고 |
|---|---|---|
| obs / action | 13 / 4 | obs 13 + waypoint 임베딩(8→16) |
| 네트워크 | `(13+16)→256 → ResBlock×3 → 4` | LayerNorm + SiLU, **Residual 사용** |
| 손실 | HuberLoss(δ=1.0) | 이상치에 강한 회귀 손실 |
| 옵티마이저 | AdamW (lr 1e-3, wd 1e-4) | |
| 스케줄 | CosineAnnealing | T_max = epochs |
| epochs / batch | 200 / 256 | |
| hidden | 256 | |
| 정규화 | 평균0/표준편차1 (`std+1e-6`) | `bc_stats.npz`에 저장 |
| 체크포인트 | `bc_policy.pt` + `bc_stats.npz` | **둘 다** 있어야 추론 가능 |

### eval_bc.py (시뮬 평가)

| 파라미터 | 값 | 의미 |
|---|---|---|
| `--n` | 50 | 무작위 held-out config 수 |
| `--seed` | 123 | 평가 재현성 |
| `SUCCESS_DIST` | 0.05 | 박스가 목표 5cm 이내면 성공 |
| `MAX_STEPS` | 1500 | episode당 최대 스텝 |
| `WP_THRESH` | 0.05 | waypoint 도달 판정[m] |
| `DWELL_MIN` | 15 | waypoint 최소 체류 스텝 |
| `GRASP_GATE` | 0.08 | 집기 트리거 거리[m] |

### policy_to_csv.py (궤적 추출)

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `--bx --by` | 0.42 / 0.20 | 박스 시작 x,y[m] (학습 범위 안) |
| `--tx --ty` | 0.42 / 0.08 | 목표 x,y[m] |
| `--size` | 0.025 | 박스 half-extent[m] |
| `--repeat` | 2 | 행 반복=실물 재생 늦춤 (**ik_traj_grip엔 1**) |
| `--maxstep` | 0.005 | 점 사이 최대 간격[m] (작을수록 부드러움) |
| `--clear` | 0.12 | 박스 위로 띄울 여유 높이[m] |
| `--slowmo` | 5.0 | 뷰어 보기 속도만 늦춤 |
| `--out` | arm_traj.csv | 출력 파일 |

### ik_traj_grip.py (실로봇 배포)

| 파라미터 | 값 | 의미 |
|---|---|---|
| `SLOWDOWN_FACTOR` | 6.0 | 재생 보간 배수 (느리고 부드럽게) |
| `INTERP_TIME` (prep) | 2.0 | 현재자세→첫 프레임 진입 시간[s] |
| `GRASP_STRENGTH` | 2.0 | 손 쥐는 힘(kp) |
| `GRASP_EXTENT` | 1.0 | 쥐는 정도 0~1 (물체 있으면 0.6~0.8) |
| arm kp / kd | 60 / 1.5 | ARM_SDK 팔 게인 |
| weight ramp | +0.01/step | 0→1 제어권 점증 |
| `JOINT_MAP_ARM` | 12~28 | 허리(12~14)+양팔(15~28) |

### 최적 설정 요약
- **데이터 일관성**(스크립트 전문가 = 결정적)이 성능의 핵심 → 시뮬 성공률 100%(5cm).
- 박스·목표는 학습 범위(`WS_X/WS_Y`) **안**에서만 신뢰. 넓히려면 재수집+재학습.
- 실로봇 재생은 `ik_traj_grip.py` + **CSV `--repeat 1`** 조합이 부드럽다.

---

## 운용 설정값 (현재 검증된 값)

### 워크스페이스 (박스·목표 랜덤 범위)
`g1_collect_demos.py` / `eval_bc.py` 공통:

| 항목 | 값 |
|---|---|
| `WS_X` (앞뒤) | (0.34, 0.48) [m] |
| `WS_Y` (좌우) | (0.05, 0.27) [m] |
| `MIN_SEP` (박스-목표 최소거리) | 0.08 [m] |
| `BLOCK_SIZE_RANGE` (half-extent) | (0.020, 0.035) [m] |

**2차 제한 — 타원형 도달 범위** (사각형의 못 가는 모서리 제거):
조건 `((x-CX)/A)² + ((y-CY)/B)² < 1` 을 만족하는 (x,y)만 사용.

| 항목 | 값 | 의미 |
|---|---|---|
| `ELLIPSE_CX` | 0.40 | 타원 중심 x (어깨 도달 중심) |
| `ELLIPSE_CY` | 0.16 | 타원 중심 y (왼손이라 왼쪽=y큰 쪽 더 잘 닿음) |
| `ELLIPSE_A` | 0.085 | x 반경 [m] |
| `ELLIPSE_B` | 0.13 | y 반경 [m] |

> 사각형 ∩ 타원 = 도달 가능한 둥근 영역만 샘플링. 이 설정으로 **실제 IK 성공률 100%**, 재학습 후 **eval 성공률 100%**.

# 3. 코드 실행 및 사용법

## 요구 사항

- Python ≥ 3.8, **GPU/CUDA 불필요**(CPU로 충분). conda 권장. *해당 컴퓨터에 GPU 드라이버가 없어 CUDA를 실행할 수 없음
- pip 패키지: `mujoco mink numpy scipy loop-rate-limiters qpsolvers daqp torch`
- 실로봇용(추가): **unitree_sdk2py** + **CycloneDDS 0.10.2** (소스 설치) + G1-EDU + Dex3 손.
- 시뮬 씬: `MINK_DIR/scene_g1_pickplace.xml`, `MINK_DIR/scene_table.xml`.

### 설치 — 한 단계씩 (처음 하는 사람용)
 
각 단계를 **따로따로** 해도 된다. 다만 **순서는 지켜야 한다**: cyclonedds(C 라이브러리)를 먼저 깔아야 그 위에 Unitree SDK가 얹힌다. 단계마다 *무엇을 / 왜 / 됐는지 확인* 순서로 설명한다.
 
> 권장: conda 가상환경 하나 만들어서 그 안에 설치. (전역 파이썬을 안 건드려서 깔끔)
> ```bash
> conda create -n robot python=3.10 -y
> conda activate robot
> ```
 
#### 단계 0 — 빌드 도구 깔기 (apt)
 
**무엇**: 다음 단계에서 cyclonedds를 "소스에서 컴파일"하는데, 그때 필요한 도구들(git, cmake, C 컴파일러 등).
**왜**: cyclonedds는 pip로 바로 안 깔리고 직접 빌드해야 해서, 빌드 도구가 미리 있어야 한다.
 
```bash
sudo apt update
sudo apt install -y git cmake build-essential python3-dev python3-pip
```
 
**확인**: `cmake --version` 과 `git --version` 이 버전을 출력하면 OK.
 
#### 단계 1 — 파이썬 패키지 깔기 (pip)
 
**무엇**: 시뮬·학습·IK에 쓰는 순수 파이썬 패키지들(mujoco, mink, torch 등).
**왜**: 시뮬 돌리고 BC 학습/추론하는 데 필요. **로봇 없이 이 단계까지만 해도** 시뮬/학습은 다 된다.
 
```bash
pip install numpy scipy
pip install mujoco mink
pip install loop-rate-limiters
pip install "qpsolvers[daqp]"   # qpsolvers + daqp 한 번에
pip install torch              # 로봇 구동만 할 거면 생략 가능 (학습/추론에만 필요)
```
 
**확인**:
```bash
python3 -c "import mujoco, mink, numpy, scipy, torch, loop_rate_limiters; print('pip OK')"
```
`pip OK` 가 뜨면 끝. **여기까지면 노트북에서 시뮬·학습은 전부 가능**하다.
(아래 2~4단계는 **실제 로봇을 돌릴 때만** 필요.)
 
#### 단계 2 — CycloneDDS 0.10.2 빌드 (소스)
 
**무엇**: 로봇과 통신(DDS)하는 C 라이브러리. Unitree SDK가 이걸 깔아야 동작한다.
**왜 소스 빌드?**: Unitree SDK가 **딱 0.10.2 버전**을 요구해서, 그 버전을 직접 받아 빌드한다.
 
```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds
mkdir build install
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
```
 
- 첫 줄: 0.10.x 버전 소스를 내려받음
- `mkdir build install`: 빌드 작업 폴더 + 결과물 설치 폴더를 만듦
- `cmake ..`: 빌드 설정(설치 위치를 `../install`로 지정)
- `cmake --build . --target install`: 컴파일 + `~/cyclonedds/install` 에 결과물 설치
**확인**: `ls ~/cyclonedds/install` 했을 때 `lib`, `include` 같은 폴더가 보이면 성공.
 
#### 단계 3 — 환경변수 알려주기 (CYCLONEDDS_HOME)
 
**무엇**: 방금 빌드한 cyclonedds가 **어디 있는지** 컴퓨터(다음 단계 pip)에게 알려주는 변수.
**왜**: 이게 없으면 단계 4에서 "cyclonedds를 못 찾겠다"며 실패한다.
 
```bash
export CYCLONEDDS_HOME=$HOME/cyclonedds/install
```
 
> ⚠️ **흔한 실수**: `export CYCLONEDDS_HOME="~/cyclonedds/install"` 처럼 **큰따옴표 안에 `~`** 를 쓰면 안 된다.
> bash가 `~`를 집 경로로 안 바꿔서 경로가 깨진다. 반드시 **`$HOME`** 을 쓸 것.
 
> 💡 `export` 는 **현재 터미널에서만** 유효해서, 터미널을 새로 열면 다시 풀린다.
> 매번 안 치려면 아래로 영구 등록:
> ```bash
> echo 'export CYCLONEDDS_HOME=$HOME/cyclonedds/install' >> ~/.bashrc
> source ~/.bashrc
> ```
 
**확인**: `echo $CYCLONEDDS_HOME` 가 `/home/<사용자>/cyclonedds/install` 를 출력하면 OK.
 
#### 단계 4 — Unitree SDK 깔기 (unitree_sdk2py)
 
**무엇**: 파이썬에서 G1 로봇을 제어하는 라이브러리. `ik_traj_grip.py` 가 이걸 import 한다.
**왜**: 실제 로봇 구동의 핵심. (이걸 깔 때 cyclonedds 파이썬 바인딩 + numpy + opencv 도 같이 깔린다.)
 
```bash
cd ~/unitree_sdk2_python      # 폴더가 없으면 먼저 git clone (아래 참고 링크)
pip install -e .
```
 
**확인**:
```bash
python3 -c "from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print('sdk OK')"
```
`sdk OK` 가 뜨면 로봇 구동 준비 완료.
 
> **`Could not locate cyclonedds` 에러가 나면** → 단계 3의 `CYCLONEDDS_HOME` 이 안 잡힌 것.
> `echo $CYCLONEDDS_HOME` 로 확인하고, 비어 있으면 단계 3을 다시 한 뒤 단계 4를 재실행.
 
#### 요약
 
| 단계 | 한 줄 | 로봇 없이도? |
|---|---|---|
| 0 | 빌드 도구(apt) | — |
| 1 | pip 패키지 | ✅ 여기까지면 시뮬·학습 가능 |
| 2 | cyclonedds 빌드 | 로봇 구동용 |
| 3 | CYCLONEDDS_HOME 설정 | 로봇 구동용 |
| 4 | Unitree SDK 설치 | 로봇 구동용 |

### 참고 링크
- Unitree SDK2 Python — https://github.com/unitreerobotics/unitree_sdk2_python
- CycloneDDS — https://github.com/eclipse-cyclonedds/cyclonedds
- mink — https://github.com/kevinzakka/mink · MuJoCo — https://github.com/google-deepmind/mujoco
- **G1 처음 켜는 사람용**: 공식 개발자 가이드 https://support.unitree.com/home/en/G1_developer · Quick Start https://support.unitree.com/home/en/G1_developer/quick_start · 보조 가이드(Weston Robot) https://docs.westonrobot.com/tutorial/unitree/g1_dev_guide/

## 파일 구성

| 구분 | 파일 | 역할 | 로봇사용 |
|---|---|---|---|
| 데이터 생성 | `g1_collect_demos.py` | 8-waypoint IK 전문가 궤적 + 랜덤화 | ❌ 시뮬 |
| 학습 | `train_bc.py` | BC 지도학습, `bc_policy.pt`+`bc_stats.npz` 저장 | ❌ |
| 시뮬 평가 | `eval_bc.py` | 무작위 config 성공률 + 렌더 | ❌ 시뮬 |
| 궤적 추출 | `policy_to_csv.py` | 정책 → `arm_traj.csv`(+grasp 열) | ❌ 시뮬 |
| 실로봇 배포 | `ik_traj_grip.py` | CSV → 실로봇 왼팔 + Dex3 손 | ✅ |
| 손 제어 | `dex3_gripper.py` | Dex3 손 클래스 (import용) | ✅ |
| 손 테스트 | `test_grip.py` | 손 단독 열기/닫기 | ✅ |

산출물: `demos/*.npz`(데이터), `bc_policy.pt`+`bc_stats.npz`(학습), `arm_traj.csv`(궤적) -> traj_bx_by_to_gx_gy.csv로 저장되게 수정함.

## 1) 데이터 만드는 방법

```bash
python3 g1_collect_demos.py --n 100        # 100개 수집 (--n 5 --view 로 점검)
```
- **로봇 불필요**(시뮬). 매 episode 박스·목표를 무작위로 놓고 8 waypoint IK → dense 궤적 → `(상태,행동)` 기록.
- 성공한 episode만 `demos/demo_XXXX.npz` 저장.

## 2) 학습하는 방법

```bash
python3 train_bc.py --epochs 200
```
- `demos/`를 정규화하여 BC 정책을 학습 → `bc_policy.pt` + `bc_stats.npz` 저장.
- ⚠️ 데이터를 다시 만들면 **반드시 재학습**(정규화 기준 일치).

## 3) 추론(시뮬 평가)하는 방법

```bash
python3 eval_bc.py --n 50            # 성공률  /  --n 1 --view 로 1개 렌더
```
- 학습에 없던 무작위 박스·목표로 rollout → 성공률(5cm). 실로봇 없이 성능 확인.

## 4) 궤적 CSV 추출

```bash
# 시뮬 미리보기
python3 policy_to_csv.py --bx 0.41 --by 0.15 --tx 0.43 --ty 0.09 --size 0.025 --repeat 1 --view
```
- `--bx --by`(박스) / `--tx --ty`(목표) / `--size`. **학습 범위**(WS_X/WS_Y) 안의 값만 신뢰.
- 실로봇용은 `--view` 빼고 `--repeat 1`로 생성 → `arm_traj.csv` -> 'traj_bx_by_to_gx_gy.csv'.

## 5) 리얼월드(실로봇) 배포

```bash
python3 ik_traj_grip.py $NET arm_traj.csv
```
절차:
1. 로봇 전원 ON → 리모컨 모드: `L2+B`(댐핑) → `L2+UP`(준비) → `R2+A`(기립) → `L2+R2`(SDK/디버그).
2. `ifconfig`로 로봇 연결 인터페이스 확인 → `$NET`에 설정.
3. `policy_to_csv.py ... --repeat 1`로 CSV 생성.
4. 실행 후 **첫 `[Enter]`**=시뮬 프리뷰(팔만), **둘째 `[Enter]`**=실제 로봇(팔+손).
5. 손 세기는 `ik_traj_grip.py` 상단 `GRASP_EXTENT`(물체 있으면 0.6~0.8)로 조절.
6. **물체를 실제로 쥐려면** 손이 내려가는 자리에 물체를 둔다(없으면 공중 여닫힘만 확인).

## 자주 겪는 이슈

- **`Could not locate cyclonedds`** → `CYCLONEDDS_HOME` 미설정(`~` 따옴표 함정, `$HOME` 사용).
- **`No module named 'dex3_gripper'`** → `dex3_gripper.py`가 실행 파일과 같은 폴더에 없음.
- **시작에서 팔이 순간이동** → `ik_traj_grip.py`(내장 스무딩) 사용 + CSV `--repeat 1`.
- **뷰어 안 뜸** → 디스플레이 있는 머신에서 실행(SSH면 X 포워딩).
- **실로봇 SDK 무반응** → 리모컨 모드가 SDK(디버그)까지 안 감 → 5)-1 순서 재확인.
- **범위 밖 좌표 실패** → `WS_X=[0.38,0.45]`, `WS_Y=[0.07,0.23]` 안의 값인지 확인.

## WHC(Whole Body Control)
- 팔과 손 관절이 정확해도 다리, 몸통 등의 관절값이 달라지면(Null space problem과 유사) 손이 정확한 위치로 이동하기 힘듦.
- 때문에 관절값을 정해줄 때 팔 이외의 다른 관절값도 **ik_traj_grip.py**에서 지정하게 수정.
- **허리까지는 컨트롤이 가능하나 다리의 경우 로봇의 자체적 중심잡는 시스템 때문에 수정 시에 위험부담이 커 조정하지 않음.
