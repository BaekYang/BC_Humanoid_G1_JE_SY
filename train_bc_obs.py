"""
train_bc_obs.py — 장애물(기둥) 회피 BC 학습.

train_bc.py 에 기둥 정보를 추가한 버전.
입력(state, 16): 손위치(3) + 블록현재위치(3) + 타겟위치(3) + 블록초기위치(3) + 블록크기(1)
                + 기둥위치(3) + waypoint_idx(단계 임베딩)
출력(action, 4): 현재 단계의 '목표 손위치'(절대 3) + grasp(1)

* 기둥위치를 obs에 넣는 이유: 정책이 "기둥이 여기 있네 → 우회 경유점으로 가야지"를 배우려면
  기둥 위치를 관측해야 함. (안 넣으면 매번 바뀌는 기둥을 모르니 못 피함)
* 웨이포인트 9개: approach, descend, grasp, lift, [detour], move, place, release, retreat
  → detour(기둥 우회 경유점)가 lift와 move 사이에 추가됨. 그 좌표는 npz의 detour_xpos 사용.

실행:
    python train_bc_obs.py --epochs 200
"""

import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DEMO_DIR = "demos_obs"
N_WAYPOINTS = 9            # 8 → 9 (detour 추가)
APPROACH_H = 0.12          # 수집기와 동일해야 함
GRASP_OFF = 0.02
OBS_DIM = 16               # 13 → 16 (기둥 위치 3D 추가)
ACT_DIM = 4


def above(p): return p + np.array([0, 0, APPROACH_H])
def on(p):    return p + np.array([0, 0, GRASP_OFF])


def load_dataset():
    files = sorted(glob.glob(f"{DEMO_DIR}/*.npz"))
    if not files:
        raise SystemExit(f"{DEMO_DIR}/ 에 npz가 없어. 먼저 g1_collect_demos_obs.py 로 데모를 모아줘.")
    O, W, A = [], [], []
    for f in files:
        d = np.load(f)
        T = d["hand_pos"].shape[0]
        binit = np.asarray(d["block_init_xpos"], dtype=np.float32)   # (3,)
        tgt = np.asarray(d["target_xpos"], dtype=np.float32)         # (3,)
        obs_pos = np.asarray(d["obstacle_xpos"], dtype=np.float32)   # (3,) 기둥 위치
        detour = np.asarray(d["detour_xpos"], dtype=np.float32)      # (3,) 우회 경유점
        bs = np.asarray(d["block_size"], dtype=np.float32).reshape(T, 1)
        hand = np.asarray(d["hand_pos"], dtype=np.float32)
        bpos = np.asarray(d["block_pos"], dtype=np.float32)
        tpos = np.asarray(d["target_pos"], dtype=np.float32)
        wp = np.asarray(d["waypoint_idx"]).reshape(T).astype(np.int64)

        # obs: 기존 13 + 기둥위치 3 = 16
        obs = np.concatenate([hand, bpos, tpos, np.tile(binit, (T, 1)), bs,
                              np.tile(obs_pos, (T, 1))], axis=1).astype(np.float32)

        # 단계별 목표 손위치 (수집 9개 웨이포인트와 동일 순서)
        # approach, descend, grasp, lift, detour, move, place, release, retreat
        goals = np.stack([above(binit), on(binit), on(binit), above(binit),
                          detour,
                          above(tgt), on(tgt), on(tgt), above(tgt)]).astype(np.float32)  # (9,3)
        wp_goal = goals[wp]                                          # (T,3)
        grasp = np.asarray(d["act_grasp"], dtype=np.float32).reshape(T, 1)
        act = np.concatenate([wp_goal, grasp], axis=1).astype(np.float32)

        O.append(obs); W.append(wp); A.append(act)
    return np.concatenate(O), np.concatenate(W), np.concatenate(A), len(files)


class ResBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
    def forward(self, x): return x + self.net(x)


class Policy(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256, n_wp=N_WAYPOINTS, emb=16):
        super().__init__()
        self.emb = nn.Embedding(n_wp, emb)
        self.inp = nn.Sequential(nn.Linear(obs_dim + emb, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.blocks = nn.Sequential(ResBlock(hidden), ResBlock(hidden), ResBlock(hidden))
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, act_dim))
    def forward(self, obs, wp):
        return self.out(self.blocks(self.inp(torch.cat([obs, self.emb(wp)], dim=-1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs, wp, act, n_demos = load_dataset()
    print(f"데모 {n_demos}개, 총 {len(obs)} 스텝, device={device}  (obs={OBS_DIM}D, waypoints={N_WAYPOINTS})")

    obs_mean, obs_std = obs.mean(0), obs.std(0) + 1e-6
    act_mean, act_std = act.mean(0), act.std(0) + 1e-6
    obs_n = (obs - obs_mean) / obs_std
    act_n = (act - act_mean) / act_std

    ds = TensorDataset(torch.from_numpy(obs_n), torch.from_numpy(wp), torch.from_numpy(act_n))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    model = Policy(hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.HuberLoss(delta=1.0)

    model.train()
    for ep in range(args.epochs):
        tot = 0.0
        for ob, w, ac in dl:
            ob, w, ac = ob.to(device), w.to(device), ac.to(device)
            loss = lossf(model(ob, w), ac)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(ob)
        sched.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}/{args.epochs}  loss {tot/len(ds):.5f}")

    torch.save(model.state_dict(), "bc_policy_obs.pt")
    np.savez("bc_stats_obs.npz", obs_mean=obs_mean, obs_std=obs_std,
             act_mean=act_mean, act_std=act_std, hidden=np.int64(args.hidden))
    print("[V] 저장: bc_policy_obs.pt, bc_stats_obs.npz")


if __name__ == "__main__":
    main()