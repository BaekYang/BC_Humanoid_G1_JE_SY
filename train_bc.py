"""
train_bc.py (v2) — 행동복제(BC) 학습.

입력(state, 13): 손위치(3) + 블록현재위치(3) + 타겟위치(3) + 블록초기위치(3) + 블록크기(1)
                + waypoint_idx (단계 임베딩)
출력(action, 4): 현재 단계의 '목표 손위치'(절대좌표 3) + grasp(1)
   → 정책이 "이 단계에선 손을 여기로 보내라"를 직접 출력 (eval에서 그대로 IK 목표로 사용)

* 블록초기위치를 obs에 넣는 이유: 잡은 뒤엔 블록=손이 돼서 현재 블록위치만으론
  'lift/복귀' 목표(블록 위)를 알 수 없기 때문.

실행:
    python3 train_bc.py --epochs 200
"""

import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DEMO_DIR = "demos"
N_WAYPOINTS = 8
APPROACH_H = 0.12          # 수집기와 동일해야 함
GRASP_OFF = 0.02
OBS_DIM = 13
ACT_DIM = 4


def above(p): return p + np.array([0, 0, APPROACH_H])
def on(p):    return p + np.array([0, 0, GRASP_OFF])


def load_dataset():
    files = sorted(glob.glob(f"{DEMO_DIR}/*.npz"))
    if not files:
        raise SystemExit("demos/ 에 npz가 없어. 먼저 g1_collect_demos.py 로 데모를 모아줘.")
    O, W, A = [], [], []
    for f in files:
        d = np.load(f)
        T = d["hand_pos"].shape[0]
        binit = np.asarray(d["block_init_xpos"], dtype=np.float32)   # (3,)
        tgt = np.asarray(d["target_xpos"], dtype=np.float32)         # (3,)
        bs = np.asarray(d["block_size"], dtype=np.float32).reshape(T, 1)
        hand = np.asarray(d["hand_pos"], dtype=np.float32)
        bpos = np.asarray(d["block_pos"], dtype=np.float32)
        tpos = np.asarray(d["target_pos"], dtype=np.float32)
        wp = np.asarray(d["waypoint_idx"]).reshape(T).astype(np.int64)

        obs = np.concatenate([hand, bpos, tpos, np.tile(binit, (T, 1)), bs], axis=1).astype(np.float32)

        # 단계별 목표 손위치 재구성 (수집 때 웨이포인트와 동일 공식)
        goals = np.stack([above(binit), on(binit), on(binit), above(binit),
                          above(tgt), on(tgt), on(tgt), above(tgt)]).astype(np.float32)  # (8,3)
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
    print(f"데모 {n_demos}개, 총 {len(obs)} 스텝, device={device}")

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

    torch.save(model.state_dict(), "bc_policy.pt")
    np.savez("bc_stats.npz", obs_mean=obs_mean, obs_std=obs_std,
             act_mean=act_mean, act_std=act_std, hidden=np.int64(args.hidden))
    print("[V] 저장: bc_policy.pt, bc_stats.npz")


if __name__ == "__main__":
    main()
