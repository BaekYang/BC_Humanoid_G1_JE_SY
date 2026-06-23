# G1 Behavior Cloning — Left-Arm Pick & Place

Unitree G1 휴머노이드용 **행동 복제(Behavior Cloning, BC)** 파이프라인.
로봇팔이 임의 위치의 박스를 집어 임의의 목표 지점에 놓도록 전문가 행동을 학습하고, MuJoCo 시뮬레이션에서 검증한 뒤 실로봇으로 배포하는 프로젝트다.

**왼팔 기반 pick-and-place**를 시뮬레이션에서 학습하고, 학습된 정책의 손 궤적을 실물 로봇 팔 동작으로 재현한다.
비전 없이 **상태(state) 기반** 관측을 사용하고, 전신 역기구학(IK)은 [`mink`](https://github.com/kevinzakka/mink)로 푼다.

---

## 1. 현재 상태 (Status)

| 단계 | 상태 |
|---|---|
| 시뮬 씬 구성 | 완료 |
| 스크립트 전문가(expert) 데모 수집 | 완료 (~150 에피소드) |
| BC 학습 | 완료 (loss → 0) |
| 시뮬 평가 | **50/50 = 100%** (블록이 타겟 5cm 이내) |
| 실물용 궤적 추출 (`arm_traj.csv`) | 완료 |
| 실물 구동 (팔 동작만) | 진행 중 — 시뮬 프리뷰 OK, 실물은 제어모드 핸드오프 확인 필요 |

> **실물 범위 = 팔 동작 재현만.** 물체도, 손가락 grasp도 포함하지 않는다. (이유는 §7 참고.)

---

## 2. 핵심 개념

- **expert가 스크립트(결정적)다.** 블록·타겟 위치가 정해지면 8개 웨이포인트가 고정 공식(블록 위 +12cm, 블록 위 +2cm 등)으로 계산된다. 같은 입력이면 항상 같은 궤적 → 무작위성이 없다.
- **action이 obs의 거의 결정적 함수다.** 정책의 출력(단계별 목표 손위치)이 (블록초기·타겟·단계)로부터 단순 덧셈 공식으로 결정된다. 그래서 신경망이 이 기하 규칙을 거의 완벽히 외워 **loss가 0**, **평가 100%**가 나온다.
- **함의:** 파이프라인이 올바르게 작동함은 증명됐지만, 어려운 일반화를 푼 것은 아니다(분포 내 단순 기하 암기). 일반화/난이도를 높이려면 §8 참고.

---

## 3. 파일 구성

### 이 프로젝트에서 새로 만든 것

| 파일 | 역할 |
|---|---|
| `g1_collect_demos.py` | 스크립트 전문가로 데모 배치 수집 → `demos/demo_XXXX.npz` 저장 |
| `train_bc.py` | BC 학습 → `bc_policy.pt`, `bc_stats.npz` 저장 |
| `eval_bc.py` | 학습된 정책 폐루프 평가(성공률) — **⚠️ 업로드 목록에 누락, §6 참고** |
| `policy_to_csv.py` | 정책을 sim에서 롤아웃해 왼손 궤적을 `arm_traj.csv`로 추출 |
| `bc_policy.pt` | 학습된 정책 가중치 |
| `bc_stats.npz` | 정규화 통계 (`obs_mean/std`, `act_mean/std`, `hidden`) |
| `arm_traj.csv` | 실물 재생용 궤적 (한 가지 블록/타겟 설정의 결과물) |

### 원래 머신에 있던 실물 제어용 (참고/재사용 by 수민)

| 파일 | 역할 |
|---|---|
| `ik_traj.py` | CSV 궤적을 ARM_SDK로 재생 (시뮬 프리뷰 → 실물 구동). **실물 구동의 메인 도구** |
| `ik_pose.py` | 단일 목표 자세를 IK로 실물 제어 |
| `pose_capture.py` | 자세 캡처 도구 |
| `traj_capture.py` | 궤적 캡처 도구 |
| `test1.csv` | 양손 궤적 샘플 CSV (1212프레임 × 63열) |

### 이 저장소에 없지만 필요한 의존 파일 - 제작 예정

- `scene_g1_pickplace.xml` — BC용 시뮬 씬(블록·테이블 포함). 위치: `/home/computer/mink/examples/unitree_g1/`
- `scene_table.xml` — `ik_traj.py`가 쓰는 씬(블록 없음, `nq=50`). 같은 mink 예제 폴더.
- `demos/` — 수집된 데모 `.npz` 폴더(학습 입력).
- `eval_bc.py` — 평가 스크립트.

---

## 4. 파이프라인

```
scene_g1_pickplace.xml          (시뮬 씬)
        │
        ▼  g1_collect_demos.py   (스크립트 전문가, 8 웨이포인트, 운동학적 grasp)
   demos/demo_XXXX.npz
        │
        ▼  train_bc.py
   bc_policy.pt + bc_stats.npz
        │
        ├─▶ eval_bc.py           (폐루프 평가 → 성공률)
        │
        ▼  policy_to_csv.py      (한 설정 롤아웃 → 왼손 궤적 추출)
   arm_traj.csv
        │
        ▼  ik_traj.py $NET arm_traj.csv   (시뮬 프리뷰 → 실물 팔 구동)
   실물 로봇
```

---

## 5. 모델 사양 (재현용)

**관측(obs) = 13차원 + 웨이포인트 인덱스**
```
[ hand_pos(3), block_pos(3), target_pos(3), block_init(3), block_size(1) ]  +  waypoint_idx → Embedding(8, 16)
```
- `block_init`이 obs에 있는 게 핵심: grasp 후엔 `block_pos == hand`라, lift/place 목표를 풀려면 블록 초기위치가 필요.

**행동(action) = 4차원**
```
[ 다음 단계 목표 손위치(3, 절대좌표), grasp(1) ]
```

**네트워크:** Residual MLP (ResBlock ×3, hidden 256), HuberLoss, AdamW, cosine schedule. 정규화 통계는 `bc_stats.npz`.

**전문가 8 웨이포인트:** approach → descend → grasp → lift → move → place → release → retreat.
`grasp`는 시뮬에서 **운동학적 부착**(블록 qpos := 손 자세)으로 구현 — 실물 미전이(§7).

**작업 공간(왼손, 테이블 위):** `x ∈ (0.38, 0.45)`, `y ∈ (0.07, 0.23)`. `APPROACH_H=0.12`, `GRASP_OFF=0.02`. 휴식 손 위치 ≈ `[0.103, 0.213, 0.585]`. 테이블 윗면 `z=0.835`(실물 83.5cm와 일치).

---

## 6. 실행 방법

작업 디렉터리: `~/unitree_sdk2_python/example/my/IK` · conda 환경: `robot` · 학습/평가는 **CPU**로 동작(이 머신 GPU 드라이버 없음, 모델이 작아 충분).

```bash
conda activate robot
cd ~/unitree_sdk2_python/example/my/IK

# 1) 데모 수집 (--view로 시각 확인 가능)
python3 g1_collect_demos.py

# 2) 학습 → bc_policy.pt, bc_stats.npz
python3 train_bc.py

# 3) 평가 (성공률)
python3 eval_bc.py

# 4) 실물용 궤적 추출
#    --repeat : 실물 재생속도(클수록 느림/안전). --slowmo : 뷰어 보기속도만(CSV 무관)
python3 policy_to_csv.py --repeat 8            # arm_traj.csv 생성 (~4.9초 재생)
python3 policy_to_csv.py --view --slowmo 8     # 천천히 보면서 확인
#    블록/타겟 바꾸기: --bx --by --tx --ty --size

# 5) 실물 구동 (프리뷰 → 실물)
python3 ik_traj.py $NET arm_traj.csv
#    1st Enter → 시뮬 프리뷰 / 2nd Enter → 실물
```

> `eval_bc.py`가 저장소에 없으면 4)·5)로 바로 갈 수 있다. 단, 평가 재현이 필요하면 복구해야 한다.

---

## 7. 로봇 특이사항 & 실물 구동

키는법, 정보 : 여기 유튜브 링크좀, 그리고 완전 처음 보는 사람이 로봇을 켜고, 실행할수 있게 순서대로 좀 해주라.

### 로봇 정보
- **Unitree G1-EDU** (몸통 29 모터: 다리 12 + 허리 3 + 양팔 14. Dex3-1 손 7DOF×2는 별도, 풀 43DOF).
- 개발 컴퓨터(Jetson Orin NX) IP `192.168.123.164`, 로그인 `unitree` / `123`.
- 통신 네트워크 인터페이스: `$NET = enx588694f6c49a`.

### 관절 인덱스 (공식 매뉴얼 확인)
| 인덱스 | 부위 |
|---|---|
| 0–5 | 왼다리 |
| 6–11 | 오른다리 |
| 12–14 | 허리 (yaw/roll/pitch) |
| **15–21** | **왼팔** (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw) |
| 22–28 | 오른팔 |

### `ik_traj.py`가 움직이는 범위
- **다리(0–11):** 명령 안 함 → 로봇 자체 밸런스 컨트롤러가 서있게 유지.
- **허리+양팔(12–28):** ARM_SDK로 명령. 왼팔=주 동작, 오른팔=고정 유지(CSV에서 오른손 목표 고정), 허리는 전신 IK로 살짝 따라 움직일 수 있음.
- 시작 시 ARM_SDK 가중치를 0→1로 램프해 부드럽게 시작. CSV 첫 점은 휴식 자세로 prepend되어 시작 점프 방지.

### `arm_traj.csv` 형식 (= `ik_traj.py` 입력)
```
[ timestamp, LH_xyz(3), LH_rpy(3, 라디안), RH_xyz(3), RH_rpy(3, 라디안) ]
```

### 실물 제어 모드 (실물이 안 움직일 때 핵심)
| 모드 | 의미 / 진입 |
|---|---|
| Zero Torque | 모터 무동작·무저항(흐물) — 부팅 직후 |
| Damping | 무동작이나 저항 있음 / `L2 + B` (= 비상정지 상태) |
| Ready | 준비 자세 / `L2 + UP` |
| Motion | 리모컨 동작 제어 / `R2 + A` |
| Debug | SDK 개발용, 내장 모션 정지 / `L2 + R2` |

- 로봇이 **댐핑/제로토크 상태로 멈춰 있으면** SDK 위치 명령을 능동적으로 따르지 않는다 → 시뮬은 되는데 실물이 안 움직이는 전형적 원인.
- ARM_SDK 명령을 먹이려면 로봇을 **제어 가능한 상태**로 올려야 함: 리모컨으로 세운 뒤**(`L2+B`→`L2+UP`→`R2+A`)**

### 전원 / 안전
- 전원 버튼은 **배터리팩**에 있음: 짧게 1회 → 2초 이상 길게 눌러 ON.
- **비상정지: `L2 + B`** (댐핑 모드로 천천히 주저앉음).
- 약 35kg의 강력한 로봇 → 서스펜션에 매단 상태, 비상정지 손 닿는 곳, 주변 비우고 운영자와 함께 구동.

---

## 8. 한계 & 다음 단계

**한계**
- **grasp 미전이:** 시뮬의 잡기는 운동학적 부착(가짜)이라 실물에 안 넘어감. 현재 실물 코드(ARM_SDK)는 손가락(Dex3)을 제어하지 않음 → 실물은 "팔 동작 시늉"만. -> 세부화해서 조정 예정
- **단순 기하 암기:** §2 참고. 분포 내 과제라 100%가 곧 강한 일반화는 아님.
- **CPU 전용:** 이 머신엔 NVIDIA 드라이버 없음(작은 모델이라 무방).

**가능한 다음 단계**
1. 일반화 테스트: 작업공간을 넓혀 학습/평가, 분포 밖 성능 확인.
2. 난이도 ↑: `waypoint_idx`를 obs에서 제거 → 상태만으로 단계 추론(더 어려운 BC).
3. 실물 grasp: Dex3 손가락 제어 추가(별도 SDK) + 실제 물체.
4. 비전 기반 관측으로 확장.

---

## 9. 의존성

- Python (conda 환경 `robot`)
- `mujoco`, `mink` (kevinzakka/mink, `humanoid_g1` 예제)
- `torch`, `numpy`, `scipy`
- `loop_rate_limiters`
- `unitree_sdk2py` (실물 구동: `ik_traj.py`, `ik_pose.py`)
