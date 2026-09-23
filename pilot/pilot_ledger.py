"""
Forgetting Ledger — pilot experiments.
Trajectory-integrated, complete attribution of catastrophic forgetting to
(parameters) x (new-task training samples) x (old-task classes).

Phase A: train on task A.  Phase B: plain SGD on task B, and at each step t
    dL_A^c(t) ~= 0.5*(g_A^c(th_t)+g_A^c(th_{t+1})) . (th_{t+1}-th_t)      (trapezoid)
Since th_{t+1}-th_t = -lr/|B| sum_b g_b(th_t), the step contribution splits
exactly across parameters (elementwise) and across batch samples (dot products).
"""
import gzip, json, math, os, sys, time, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.func import functional_call, vmap, grad

torch.set_num_threads(2)
DATA = "/home/claude/data"
DEV = "cpu"


def load_idx(path):
    with gzip.open(path, "rb") as f:
        buf = f.read()
    magic = int.from_bytes(buf[2:4], "big")
    nd = buf[3]
    dims = [int.from_bytes(buf[4 + 4 * i:8 + 4 * i], "big") for i in range(nd)]
    return np.frombuffer(buf, dtype=np.uint8, offset=4 + 4 * nd).reshape(dims)


def load(name):
    d = f"{DATA}/{name}"
    Xtr = load_idx(f"{d}/train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255.
    ytr = load_idx(f"{d}/train-labels-idx1-ubyte.gz").astype(np.int64)
    Xte = load_idx(f"{d}/t10k-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255.
    yte = load_idx(f"{d}/t10k-labels-idx1-ubyte.gz").astype(np.int64)
    return Xtr, ytr, Xte, yte


def make_benchmark(bench, seed, nA=20000, nB=10000):
    """returns dict with A/B train/test tensors, probe set for A, n classes."""
    rng = np.random.RandomState(seed)
    if bench == "pmnist":
        Xtr, ytr, Xte, yte = load("mnist")
        perm = rng.permutation(784)
        idx = rng.permutation(len(Xtr))
        iA, iB, iP = idx[:nA], idx[nA:nA + nB], idx[nA + nB:nA + nB + 1000]
        A = (Xtr[iA], ytr[iA]); B = (Xtr[iB][:, perm], ytr[iB])
        P = (Xtr[iP], ytr[iP])
        At = (Xte, yte); Bt = (Xte[:, perm], yte)
        ncls = 10
    elif bench == "sfmnist":  # split Fashion-MNIST, domain-incremental (shared 5-way head)
        Xtr, ytr, Xte, yte = load("fashion")
        mA, mB = ytr < 5, ytr >= 5
        XA, yA = Xtr[mA], ytr[mA]; XB, yB = Xtr[mB], ytr[mB] - 5
        ia = rng.permutation(len(XA)); ib = rng.permutation(len(XB))
        A = (XA[ia[:nA]], yA[ia[:nA]]); P = (XA[ia[nA:nA + 1000]], yA[ia[nA:nA + 1000]])
        B = (XB[ib[:nB]], yB[ib[:nB]])
        At = (Xte[yte < 5], yte[yte < 5]); Bt = (Xte[yte >= 5], yte[yte >= 5] - 5)
        ncls = 5
    else:
        raise ValueError(bench)
    T = lambda p: (torch.tensor(p[0]), torch.tensor(p[1]))
    return dict(A=T(A), B=T(B), P=T(P), At=T(At), Bt=T(Bt), ncls=ncls)


class MLP(nn.Module):
    def __init__(self, ncls, h=200):
        super().__init__()
        self.fc1 = nn.Linear(784, h); self.fc2 = nn.Linear(h, h); self.fc3 = nn.Linear(h, ncls)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


def acc(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def class_acc(model, X, y, ncls):
    with torch.no_grad():
        p = model(X).argmax(1)
    return np.array([(p[y == c] == c).float().mean().item() for c in range(ncls)])


def train_A(model, A, seed, epochs=5, lr=1e-3, bs=128):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X, y = A
    for ep in range(epochs):
        idx = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), bs):
            b = idx[i:i + bs]
            opt.zero_grad(); F.cross_entropy(model(X[b]), y[b]).backward(); opt.step()
    return model


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat(model, v):
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel(); p.copy_(v[i:i + n].view_as(p)); i += n


def layer_slices(model):
    out, i = {}, 0
    for n, p in model.named_parameters():
        out[n] = (i, i + p.numel()); i += p.numel()
    return out


def perclass_grads(model, P, ncls):
    """returns [ncls, Pdim] gradient of mean probe loss per old class, and losses"""
    X, y = P
    G, L = [], []
    for c in range(ncls):
        m = y == c
        model.zero_grad()
        loss = F.cross_entropy(model(X[m]), y[m])
        loss.backward()
        G.append(torch.cat([p.grad.reshape(-1) for p in model.parameters()]))
        L.append(loss.item())
    return torch.stack(G), np.array(L)


def per_sample_grads(model, xb, yb):
    params = {k: v.detach() for k, v in model.named_parameters()}

    def lossf(prm, x, y):
        out = functional_call(model, prm, (x.unsqueeze(0),))
        return F.cross_entropy(out, y.unsqueeze(0))
    gs = vmap(grad(lossf), in_dims=(None, 0, 0))(params, xb, yb)
    return torch.cat([gs[k].reshape(len(xb), -1) for k, _ in model.named_parameters()], 1)


def train_B(model, B, seed, epochs=3, lr=0.05, bs=50, keep=None, P=None, ncls=None, track=False):
    """Plain SGD on B. If track: returns ledger (param x class, sample x class)."""
    X, y = B
    N = len(X)
    g = torch.Generator().manual_seed(seed + 999)
    orders = [torch.randperm(N, generator=g) for _ in range(epochs)]  # fixed order for counterfactuals
    keepmask = torch.ones(N, dtype=torch.bool) if keep is None else keep
    led = None
    if track:
        Pdim = flat_params(model).numel()
        Cpar = torch.zeros(Pdim, ncls, dtype=torch.float64)
        Cdat = torch.zeros(N, ncls, dtype=torch.float64)
        Lpar = torch.zeros(Pdim, dtype=torch.float64)   # B-learning credit per parameter
        ckpt_grads = []
        GA, L0 = perclass_grads(model, P, ncls)
        early = None; steps = 0
        total_steps = sum(math.ceil(N / bs) for _ in orders)
    ckpts = []
    for ep in range(epochs):
        if track: ckpts.append(flat_params(model).clone())
        idx = orders[ep][keepmask[orders[ep]]]
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            if track:
                gb = per_sample_grads(model, X[b], y[b])             # [nb, Pdim]
                delta = -lr * gb.mean(0)
                set_flat(model, flat_params(model) + delta)
                GA1, _ = perclass_grads(model, P, ncls)
                gbar = 0.5 * (GA + GA1)                               # [C, Pdim]
                Cpar += (gbar * delta.unsqueeze(0)).T.double()
                Lpar += (-gb.mean(0) * delta).double()
                Cdat[b] += (-lr / len(b) * gb @ gbar.T).double()
                GA = GA1; steps += 1
                if early is None and steps >= 0.2 * total_steps:
                    early = Cdat.clone()
            else:
                model.zero_grad()
                F.cross_entropy(model(X[b]), y[b]).backward()
                with torch.no_grad():
                    for p in model.parameters():
                        p -= lr * p.grad
    if track:
        _, L1 = perclass_grads(model, P, ncls)
        led = dict(ckpts=ckpts, Lpar=Lpar, Cpar=Cpar, Cdat=Cdat, Cdat_early=early, dL_true=L1 - L0, L0=L0, L1=L1)
    return model, led


def train_B_online(model, B, seed, P, ncls, mode, q=0.2, epochs=3, lr=0.05, bs=50, er_k=10):
    """Online use of the ledger signal. Probe set P doubles as a small memory buffer for all modes."""
    X, y = B; N = len(X)
    g = torch.Generator().manual_seed(seed + 999)
    orders = [torch.randperm(N, generator=g) for _ in range(epochs)]
    gr = torch.Generator().manual_seed(seed + 5)
    XP, yP = P
    for ep in range(epochs):
        idx = orders[ep]
        for i in range(0, N, bs):
            b = idx[i:i + bs]
            gb = per_sample_grads(model, X[b], y[b])
            if mode in ("gate", "er_gate", "agem"):
                GA, _ = perclass_grads(model, P, ncls); gref = GA.mean(0)
            if mode in ("gate", "er_gate"):
                contrib = -(gb @ gref)                       # >0 => this sample increases old loss
                keep = torch.argsort(contrib)[: int(round((1 - q) * len(b)))]
                gb = gb[keep]
            elif mode == "rand_gate":
                keep = torch.randperm(len(b), generator=gr)[: int(round((1 - q) * len(b)))]
                gb = gb[keep]
            gstep = gb.mean(0)
            if mode in ("er", "er_gate"):
                j = torch.randint(len(XP), (er_k,), generator=gr)
                model.zero_grad(); F.cross_entropy(model(XP[j]), yP[j]).backward()
                gm = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
                gstep = (len(gb) * gstep + er_k * gm) / (len(gb) + er_k)
            if mode == "agem":
                d = gstep @ gref
                if d < 0: gstep = gstep - d / (gref @ gref) * gref
            set_flat(model, flat_params(model) - lr * gstep)
    return model


def fisher_diag(model, A, n=2000):
    X, y = A
    gs = []
    for i in range(0, n, 100):
        gs.append(per_sample_grads(model, X[i:i + 100], y[i:i + 100]) ** 2)
    return torch.cat(gs).mean(0)


def run(bench, seed, out):
    t0 = time.time()
    torch.manual_seed(seed)
    D = make_benchmark(bench, seed)
    ncls = D["ncls"]
    model = MLP(ncls)
    train_A(model, D["A"], seed)
    thA = flat_params(model).clone()
    accA0 = acc(model, *D["At"])
    Fi = fisher_diag(model, D["A"])
    GA_start, _ = perclass_grads(model, D["P"], ncls)
    # Phase B tracked
    model, led = train_B(model, D["B"], seed, P=D["P"], ncls=ncls, track=True)
    thB = flat_params(model).clone()
    accA1, accB1 = acc(model, *D["At"]), acc(model, *D["Bt"])
    GA_end, _ = perclass_grads(model, D["P"], ncls)
    dth = thB - thA
    res = dict(bench=bench, seed=seed, accA0=accA0, accA1=accA1, accB1=accB1)

    # ---- E1 completeness
    pred = led["Cpar"].sum(0).numpy(); pred_d = led["Cdat"].sum(0).numpy(); true = led["dL_true"]
    res["E1"] = dict(true_dL=true.tolist(), pred_param=pred.tolist(), pred_data=pred_d.tolist(),
                     rel_err_total=float(abs(pred.sum() - true.sum()) / abs(true.sum())),
                     rel_err_perclass_max=float(np.max(np.abs(pred - true) / np.abs(true).clip(1e-3))),
                     first_order_start=float((GA_start.sum(0) @ dth).item()),
                     first_order_end=float((GA_end.sum(0) @ dth).item()))
    # layer shares
    sl = layer_slices(model); cp = led["Cpar"].sum(1)
    res["layer_share"] = {k: float(cp[a:b].sum() / cp.sum()) for k, (a, b) in sl.items()}

    # ---- E2 parameter rollback faithfulness
    scores = {
        "ledger": cp.float(),
        "ledger_net": (cp - (cp.sum() / led["Lpar"].sum()) * led["Lpar"]).float(),
        "abs_delta": dth.abs(),
        "fisher_delta2": Fi * dth ** 2,
        "taylor_end": (GA_end.sum(0) * dth),          # one-shot linearisation at th_B
        "random": torch.rand(dth.numel(), generator=torch.Generator().manual_seed(seed)),
    }
    E2 = {}
    fracs = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
    for name, s in scores.items():
        order = torch.argsort(s, descending=True)
        rows = []
        for f in fracs:
            k = int(f * dth.numel()); th = thB.clone(); sel = order[:k]
            th[sel] = thA[sel]; set_flat(model, th)
            rows.append(dict(frac=f, accA=acc(model, *D["At"]), accB=acc(model, *D["Bt"])))
        E2[name] = rows
    # layer rollback (causal) vs ledger layer share
    lay = {}
    for k_, (a, b) in sl.items():
        th = thB.clone(); th[a:b] = thA[a:b]; set_flat(model, th)
        lay[k_] = acc(model, *D["At"]) - accA1
    res["E2"] = E2; res["layer_rollback_gain"] = lay
    set_flat(model, thB)

    # ---- E3 data removal counterfactuals (retrain B from th_A without selected samples)
    XB, yB = D["B"]; N = len(XB)
    tot = led["Cdat"].sum(1).float()
    set_flat(model, thA)
    with torch.no_grad():
        lossB_A = F.cross_entropy(model(XB), yB, reduction="none")
    gA0 = GA_start.sum(0)
    static = torch.cat([-(per_sample_grads(model, XB[i:i + 500], yB[i:i + 500]) @ gA0) for i in range(0, N, 500)])
    early = led["Cdat_early"].sum(1).float()
    tcpC = torch.zeros(N, ncls)
    for ck in led["ckpts"]:
        set_flat(model, ck); gk, _ = perclass_grads(model, D["P"], ncls)
        tcpC += torch.cat([-(per_sample_grads(model, XB[i:i + 500], yB[i:i + 500]) @ gk.T) for i in range(0, N, 500)])
    tcp = tcpC.sum(1)
    set_flat(model, thA)
    sel_scores = {"ledger_harmful": tot, "ledger_early20": early, "static_tracin": static, "tracin_cp3": tcp,
                  "high_loss": lossB_A, "random": torch.rand(N, generator=torch.Generator().manual_seed(seed + 7)),
                  "ledger_helpful": -tot}
    E3 = {}
    for name, s in sel_scores.items():
        E3[name] = []
        for f in [0.05, 0.10, 0.20]:
            k = int(f * N); keep = torch.ones(N, dtype=torch.bool)
            keep[torch.argsort(s, descending=True)[:k]] = False
            m = MLP(ncls); set_flat(m, thA)
            train_B(m, D["B"], seed, keep=keep)
            E3[name].append(dict(frac=f, accA=acc(m, *D["At"]), accB=acc(m, *D["Bt"])))
    res["E3"] = E3
    res["rank_corr_early_full"] = float(np.corrcoef(np.argsort(np.argsort(early.numpy())),
                                                     np.argsort(np.argsort(tot.numpy())))[0, 1])

    # ---- E4 class-targeted removal: protect the most-forgotten old class
    Cd = led["Cdat"].float()
    cstar = int(np.argmax(true))
    m_cls = class_acc  # alias
    base = MLP(ncls); set_flat(base, thB)
    ca_base = class_acc(base, *D["At"], ncls)
    E4 = {}
    k = int(0.10 * N)
    for name, s in {"target_class": Cd[:, cstar], "tracin_cp3_class": tcpC[:, cstar], "total": tot,
                    "random": torch.rand(N, generator=torch.Generator().manual_seed(seed + 11))}.items():
        keep = torch.ones(N, dtype=torch.bool); keep[torch.argsort(s, descending=True)[:k]] = False
        m = MLP(ncls); set_flat(m, thA); train_B(m, D["B"], seed, keep=keep)
        ca = class_acc(m, *D["At"], ncls)
        E4[name] = dict(target_gain=float(ca[cstar] - ca_base[cstar]),
                        other_gain=float(np.delete(ca - ca_base, cstar).mean()), accB=acc(m, *D["Bt"]))
    res["interference_BxA"] = [[float(Cd[yB == cb][:, ca].sum()) for ca in range(ncls)] for cb in range(ncls)]
    res["E4"] = dict(cstar=cstar, runs=E4)
    E5 = {}
    for mode in ["none", "rand_gate", "gate", "agem", "er", "er_gate"]:
        m = MLP(ncls); set_flat(m, thA)
        train_B_online(m, D["B"], seed, D["P"], ncls, mode)
        E5[mode] = dict(accA=acc(m, *D["At"]), accB=acc(m, *D["Bt"]))
    res["E5"] = E5
    res["time_s"] = time.time() - t0
    json.dump(res, open(out, "w"), indent=1)
    print(bench, seed, f"{res['time_s']:.0f}s", "accA", accA0, "->", accA1, "accB", accB1,
          "compl err", res["E1"]["rel_err_total"], flush=True)


if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    os.makedirs("/home/claude/pilot/results", exist_ok=True)
    run(bench, seed, f"/home/claude/pilot/results/{bench}_{seed}.json")
