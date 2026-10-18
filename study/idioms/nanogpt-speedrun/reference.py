# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Keller Jordan and the modded-nanogpt contributors
# SPDX-FileCopyrightText: 2026 The Viv authors
"""A faithful, minimal float64 PyTorch version of the parts of modded-nanogpt's
train_gpt.py (KellerJordan/modded-nanogpt @ bc3a0c2) that the Viv port covers,
at a tiny size. It is the reference the Viv port is checked against.

What is kept, with the function in train_gpt.py it mirrors:
  - tied embed / lm_head, untied at 2/3 of training   (GPT.__init__, NorMuonAndAdam.step,
                                                        copy_lm_state_to_embed)
  - smear gate                                         (GPT.forward, "smear token embed")
  - x0 = norm(x), x0 injection gates from the pre MUDD gate (forward_mudd_gate)
  - attention: sa_lambdas-scaled QKV, QK RMS-norm, half-truncated RoPE (Yarn.reset),
    partial key offset on the long-window layer, sliding windows, value embeddings
    with the 2*sigmoid(x[:6] ++ ve[:6]) gate, o_w stored transposed
                                                       (CausalSelfAttention.forward,
                                                        _qk_norm_rope_pad_forward_kernel)
  - resid_lambdas / post_lambdas residual scaling
  - ReLU^2 MLP with c_fc / c_proj                      (FusedLinearReLUSquareFunction)
  - final norm, softcap 23*sigmoid((z+5)/7.5), summed cross-entropy
  - NorMuon: Nesterov momentum, Polar Express, variance reduction, cautious weight
    decay, shape lr multiplier, 2x lr on c_proj        (polar_express, _normuon_update, ...)
  - Adam with bias correction and cautious weight decay, stepped on odd steps only
    with gradients accumulating over the even step     (_adam_update, step_optimizers)
  - lr schedule (stages + cooldown) and Muon momentum warmup/cooldown, scaled to STEPS

Left out (see NOTES.md): MUDD residual mixing, bigram hash embedding, XSA, DC attention,
paired heads, attention gates, the layer-6 skip, fp8/bf16 and mantissa tracking, MTP and
prefix losses, YaRN window extension, the canonical-token mask, distributed comms.

Run:  python3 reference.py         (writes ref.viv beside it and prints the numbers)
"""
import math
import torch

torch.set_default_dtype(torch.float64)

# ---- sizes -------------------------------------------------------------------
L = 2            # layers
H = 2            # heads
HD = 16          # head_dim
QK = 8           # qk_dim (the file: 96 < head_dim 128)
D = H * HD       # model_dim = num_heads * head_dim
M = 4 * D        # mlp hidden
G = 16           # mudd gate hidden (the file: 64)
T = 16           # sequence length
B = 4            # sequences per step
STEPS = 30
WS_SHORT, WS_LONG = 8, 16          # window sizes in tokens
WINDOWS = [WS_SHORT, WS_LONG]      # the last layer is a long-window layer, like layer 10
KEY_OFFSET = [False, True]         # "apply partial key offset to long windows"
VE_LAYERS = [False, True]          # [None, ve0, ...]: layer 0 has no value embedding
EPS = 1.1920928955078125e-7        # float32 eps, the constant in the QK-norm kernel
ROT = QK // 2
ATTN_SCALE = 0.1 * math.sqrt(HD / QK)

TEXT = ("it was the best of times, it was the worst of times, it was the age of wisdom, "
        "it was the age of foolishness, it was the epoch of belief, it was the epoch of "
        "incredulity, it was the season of light, it was the season of darkness, it was "
        "the spring of hope, it was the winter of despair, we had everything before us, "
        "we had nothing before us, we were all going direct to heaven, we were all going "
        "direct the other way. in short, the period was so far like the present period, "
        "that some of its noisiest authorities insisted on its being received, for good "
        "or for evil, in the superlative degree of comparison only.")
VOCAB = sorted(set(TEXT))
V = 32                     # next_multiple_of_n(len(VOCAB), n=32): the file pads 50257 to 50304
TOKENS = [VOCAB.index(c) for c in TEXT]
N = len(TOKENS)


def batch(step):
    """Sequence b of step s starts at ((s*B + b) * 37) mod (N - T - 1)."""
    xs, ys = [], []
    for b in range(B):
        s0 = ((step * B + b) * 37) % (N - T - 1)
        xs.append(TOKENS[s0:s0 + T])
        ys.append(TOKENS[s0 + 1:s0 + T + 1])
    return torch.tensor(xs), torch.tensor(ys)


# ---- rotary factors (Yarn.reset, not paired) ----------------------------------
ang = (1 / 1024) ** torch.linspace(0, 1, steps=ROT // 2)
ang = ang.repeat_interleave(2)
ang = torch.cat([ang, ang.new_zeros(QK - ROT)])
tpos = torch.arange(T, dtype=torch.float64)
theta_rope = torch.outer(tpos, ang)
F1 = theta_rope.cos()
F2 = theta_rope.sin()
F2[..., 1::2] *= -1


def norm(x):
    # F.rms_norm(x, (x.size(-1),), eps=EPS), written out: torch 2.2 predates it
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS)


def qk_norm_rope(qk, key_offset):
    """qk: (T, 2H, QK). Mirrors _qk_norm_rope_pad_forward_kernel (not paired)."""
    n = qk / torch.sqrt((qk * qk).mean(-1, keepdim=True) + EPS)
    flip = n.view(T, 2 * H, QK // 2, 2).flip(-1).reshape(T, 2 * H, QK)
    y = F1[:, None, :] * n + F2[:, None, :] * flip
    if key_offset:
        y = y.clone()
        # key heads, token > 0, lanes >= rotary_dim: the previous token's normalized lanes
        y[1:, H:, ROT:] = n[:-1, H:, ROT:]
    return y[:, :H], y[:, H:]


def gelu(x):
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))


def attn_mask(window):
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    return (j <= i) & (j >= i - window)


MASKS = [attn_mask(w) for w in WINDOWS]


def forward(P, idx, tgt):
    """Summed softcapped cross-entropy of one sequence. P: dict of tensors."""
    x = P["embed"][idx]                                   # (T, D)
    sg = torch.sigmoid(x[1:, :12] @ P["smear_gate"].T)    # (T-1, 1)
    smear = P["scalars"][2 * L] * sg
    x = torch.cat([x[:1], x[1:] + smear * x[:-1]])
    x = x0 = norm(x)
    gate = (gelu(x0 @ P["gate_w1"].T) @ P["gate_w2"].T + P["gate_b2"]) * P["gate_scale"]  # (T, L)
    ve = P["value_embeds"][idx]                           # (T, D)
    for l in range(L):
        sa = P["scalars"][2 * l: 2 * l + 2]
        xn = norm(x)
        qk = (xn @ (sa[0] * P[f"qk{l}"]).T).view(T, 2 * H, QK)
        v = (xn @ (sa[0] * P[f"v{l}"]).T).view(T, H, HD)
        q, k = qk_norm_rope(qk, KEY_OFFSET[l])
        if VE_LAYERS[l]:
            gate_in = torch.cat([xn[:, :6], ve[:, :6]], dim=-1)
            veg = 2 * torch.sigmoid(gate_in @ P["ve_gate"].T)      # (T, H)
            v = v + veg[:, :, None] * ve.view(T, H, HD)
        s = torch.einsum("ihd,jhd->hij", q, k) * ATTN_SCALE
        s = s.masked_fill(~MASKS[l], float("-inf"))
        p = torch.softmax(s, dim=-1)
        y = torch.einsum("hij,jhd->ihd", p, v).reshape(T, D)
        out = y @ (sa[1] * P[f"o{l}"])
        x = P["resid"][l, 0] * x + P["post"][l, 0] * out + x0 * gate[:, l:l + 1]
        h = torch.relu(norm(x) @ P[f"fc{l}"].T) ** 2
        x = P["resid"][l, 1] * x + P["post"][l, 1] * (h @ P[f"proj{l}"])
    z = norm(x) @ P["lm_head"]
    logits = 23 * torch.sigmoid((z + 5) / 7.5)
    return torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")


def total_loss(P, step):
    xs, ys = batch(step)
    return sum(forward(P, xs[b], ys[b]) for b in range(B))


# ---- parameters ----------------------------------------------------------------
# name -> (optim, beta1, beta2, lr_mul, wd_mul), from TrainingManager.param_table
TABLE = {
    "lm_head":      ("adam", 0.5, 0.95, 1.0, 150.0),
    "embed":        ("adam", 0.5, 0.95, 1.0, 150.0),
    "value_embeds": ("adam", 0.75, 0.95, 75.0, 5.0),
    "ve_gate":      ("adam", 0.9, 0.99, 1.0, 1.0),
    "smear_gate":   ("adam", 0.9, 0.99, 0.01, 0.0),
    "scalars":      ("adam", 0.9, 0.99, 5.0, 0.0),
    "resid":        ("adam", 0.9, 0.95, 5.0, 0.0),
    "post":         ("adam", 0.9, 0.95, 1.0, 0.0),
    "gate_w1":      ("adam", 0.9, 0.99, 0.1, 1.0),
    "gate_w2":      ("adam", 0.9, 0.99, 0.1, 1.0),
    "gate_b2":      ("adam", 0.9, 0.99, 0.1, 0.0),
    "gate_scale":   ("adam", 0.9, 0.99, 0.1, 0.0),
}
for l in range(L):
    for nm in ("qk", "v", "o", "fc", "proj"):
        TABLE[f"{nm}{l}"] = ("normuon", 0.0, 0.0, 1.0, 1.0)
ORDER = list(TABLE)
ADAM_LR, ADAM_EPS, ADAM_WD = 0.008, 1e-10, 0.005
MUON_LR, MUON_BETA2, MUON_WD = 0.023, 0.9, 1.2


def init_params(gen):
    """The file's initializations, at this size."""
    bound = 3 ** 0.5 * 0.5 * D ** -0.5
    P = {}
    P["lm_head"] = torch.randn(D, V, generator=gen) * 0.005
    P["embed"] = P["lm_head"].T.clone()
    P["value_embeds"] = 0.01 * torch.randn(V, D, generator=gen)
    P["ve_gate"] = torch.zeros(H, 12)
    P["smear_gate"] = torch.zeros(1, 12)
    P["scalars"] = torch.tensor([0.5, 1.0] * L + [0.0])
    P["resid"] = torch.full((L, 2), 1.1 ** 0.5)
    P["post"] = torch.ones(L, 2)
    b1 = 1 / math.sqrt(D)                  # kaiming_uniform_(a=sqrt(5)) bound
    P["gate_w1"] = (torch.rand(G, D, generator=gen) * 2 - 1) * b1
    P["gate_w2"] = torch.zeros(L, G)
    P["gate_b2"] = torch.zeros(L)          # x0 gate bias starts at 0
    P["gate_scale"] = torch.tensor(0.1)
    for l in range(L):
        P[f"qk{l}"] = (torch.rand(2 * H * QK, D, generator=gen) * 2 - 1) * bound
        P[f"v{l}"] = (torch.rand(D, D, generator=gen) * 2 - 1) * bound
        P[f"o{l}"] = (torch.rand(D, D, generator=gen) * 2 - 1) * bound
        P[f"fc{l}"] = (torch.rand(M, D, generator=gen) * 2 - 1) * bound
        P[f"proj{l}"] = torch.zeros(M, D)
    return P


def perturbed_params(gen):
    """Every parameter nonzero, so every backward path carries a gradient."""
    P = init_params(gen)
    for k in P:
        P[k] = P[k] + 0.3 * torch.randn(P[k].shape, generator=gen) * (P[k].abs().mean() + 0.05)
    return P


# ---- the optimizer (NorMuonAndAdam, one device) --------------------------------
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def polar_express(g, buf, momentum):
    buf.lerp_(g, 1 - momentum)
    g = g.lerp(buf, momentum)
    X = g / (g.norm() * (1 + 2e-2) + 1e-6)
    tall = X.size(-2) > X.size(-1)
    for a, b, c in polar_express_coeffs:
        if tall:
            A = X.T @ X
            X = a * X + X @ (b * A + c * A @ A)
        else:
            A = X @ X.T
            X = a * X + (b * A + c * A @ A) @ X
    return X


def normuon_variance_reduction(v, second, beta2, red_dim):
    v_mean = v.square().mean(dim=red_dim, keepdim=True)
    red_dim_size = v.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    second.lerp_(v_mean, 1 - beta2)
    step_size = second.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    return v * (step_size * (v_norm / v_norm_new.clamp_min(1e-10)))


def muon_groups(name, p):
    """(row0, rows, per-matrix lr mul) for each matrix Polar Express sees."""
    if name.startswith("qk"):
        return [(0, 2 * QK, 1.0), (2 * QK, 2 * QK, 1.0)]  # one group per head pair of Q, of K
    if name.startswith("proj"):
        return [(0, p.size(0), 2.0)]                        # "c_proj gets 2x LR"
    return [(0, p.size(0), 1.0)]


class Optimizer:
    def __init__(self, P):
        self.state = {}
        for name, p in P.items():
            if TABLE[name][0] == "adam":
                self.state[name] = dict(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
            else:
                st = []
                for r0, rows, _ in muon_groups(name, p):
                    cols = p.size(1)
                    shp = (rows, 1) if rows >= cols else (1, cols)
                    st.append(dict(buf=torch.zeros(rows, cols), second=torch.zeros(shp)))
                self.state[name] = st
        self.split = False

    def adam(self, name, p, g, lr_sched):
        _, b1, b2, lr_mul, wd_mul = TABLE[name]
        lr = ADAM_LR * lr_sched * lr_mul
        s = self.state[name]
        s["step"] += 1
        t = s["step"]
        bias1, bias2 = 1 - b1 ** t, 1 - b2 ** t
        step_size = lr * (bias2 ** 0.5 / bias1)
        eff_wd = lr * lr * ADAM_WD * wd_mul
        s["m"].mul_(b1).add_(g, alpha=1 - b1)
        s["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
        update = s["m"].div(s["v"].sqrt().add(ADAM_EPS)).mul(step_size)
        mask = (update * p) > 0
        update.addcmul_(p, mask, value=eff_wd)
        p.add_(update, alpha=-1.0)

    def normuon(self, name, p, g, lr_sched, momentum):
        rows_all, cols = p.shape
        shape_mult = max(1.0, rows_all / cols) ** 0.5 if not name.startswith("qk") else 1.0
        for (r0, rows, mat_mul), s in zip(muon_groups(name, p), self.state[name]):
            gg = g[r0:r0 + rows].clone()
            v = polar_express(gg, s["buf"], momentum)
            red_dim = -1 if rows >= cols else -2
            v = normuon_variance_reduction(v, s["second"], MUON_BETA2, red_dim)
            lr = MUON_LR * lr_sched * shape_mult * mat_mul
            wd = MUON_WD * MUON_LR * lr_sched          # wd_mul * weight_decay * p_cfg.lr
            pp = p[r0:r0 + rows]
            mask = (v * pp) >= 0
            pp.copy_(pp - pp * mask * wd * lr - v * lr)

    def step(self, P, grads, step):
        lr_sched = get_lr(step)
        mom = get_muon_momentum(step)
        do_adam = step % 2 == 1
        if do_adam and not self.split:
            grads["lm_head"] = grads["lm_head"] + grads["embed"].T
        for name in ORDER:
            kind = TABLE[name][0]
            if kind == "adam" and not do_adam:
                continue
            if name == "embed" and not self.split:
                continue
            if kind == "adam":
                self.adam(name, P[name], grads[name], lr_sched)
            else:
                self.normuon(name, P[name], grads[name], lr_sched, mom)
        if do_adam and not self.split:
            P["embed"].copy_(P["lm_head"].T)
        # clear the gradients the step used; Adam's accumulate over the even step
        for name in ORDER:
            if TABLE[name][0] == "adam" and not do_adam:
                continue
            grads[name] = torch.zeros_like(grads[name])
        if step == SPLIT_STEP:
            se, sl = self.state["embed"], self.state["lm_head"]
            se["step"] = sl["step"]
            se["m"] = sl["m"].T.clone()
            se["v"] = sl["v"].T.clone()
            self.split = True


# ---- schedules (TrainingSchedule.get_lr, get_muon_momentum), scaled to STEPS ---
STAGE_ENDS = [STEPS // 3, 2 * STEPS // 3, STEPS]
STAGE_LR = [1.0, 1.52, 1.73]
COOLDOWN_FRAC = 0.60
SPLIT_STEP = STAGE_ENDS[1] | 1
MOM_WARMUP = 7       # 300 of 1290 steps
MOM_COOLDOWN = 2     # 50 of 1290 steps


def get_lr(step):
    stage = next(i for i, e in enumerate(STAGE_ENDS) if step < e) if step < STEPS else 2
    lr = STAGE_LR[stage]
    cd_start = int(STEPS * (1 - COOLDOWN_FRAC))
    if step >= cd_start:
        t = min(1.0, (step - cd_start) / (STEPS - cd_start))
        lr = lr * (1 - t) + 0.15 * t
    return lr


def get_muon_momentum(step, momentum_min=0.85, momentum_max=0.95):
    cd_start = STEPS - MOM_COOLDOWN
    if step < MOM_WARMUP:
        return momentum_min + step / MOM_WARMUP * (momentum_max - momentum_min)
    if step > cd_start:
        return momentum_max - (step - cd_start) / MOM_COOLDOWN * (momentum_max - momentum_min)
    return momentum_max


def grads_of(P, step):
    Q = {k: v.clone().requires_grad_(True) for k, v in P.items()}
    loss = total_loss(Q, step)
    loss.backward()
    return loss.item(), {k: Q[k].grad.detach().clone() for k in P}


def train(P):
    opt = Optimizer(P)
    acc = {k: torch.zeros_like(v) for k, v in P.items()}
    losses = []
    for step in range(STEPS):
        loss, g = grads_of(P, step)
        losses.append(loss / (B * T))
        for k in acc:
            acc[k] = acc[k] + g[k]
        opt.step(P, acc, step)
    losses.append(total_loss(P, STEPS).item() / (B * T))
    return losses


# ---- export ------------------------------------------------------------------
def viv_array(t):
    t = t.detach()
    if t.dim() == 0:
        return repr(float(t))
    if t.dim() == 1:
        return "[" + ", ".join(repr(float(x)) for x in t) + "]"
    rows = [viv_array(r) for r in t]
    return "[" + ",\n    ".join(rows) + "]"


def viv_matrix(t):
    """Always rank 2, so the port can param_set it."""
    t = t.detach()
    if t.dim() == 0:
        t = t.reshape(1, 1)
    elif t.dim() == 1:
        t = t.reshape(1, -1)
    return viv_array(t)


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(1234)
    P_init = init_params(gen)
    P_pert = perturbed_params(torch.Generator().manual_seed(99))

    # 1. forward loss and every gradient at the perturbed point
    loss_p, g_p = grads_of(P_pert, 0)
    # 2. one optimizer step (an odd step, so Adam runs too) from the perturbed point
    P_step = {k: v.clone() for k, v in P_pert.items()}
    opt = Optimizer(P_step)
    opt.step(P_step, dict(g_p), 1)
    # 3. training from the file's initialization
    P_train = {k: v.clone() for k, v in P_init.items()}
    loss_init = total_loss(P_init, 0).item()
    losses = train(P_train)

    print(f"V={V} N={N} D={D} H={H} HD={HD} QK={QK} L={L} T={T} B={B} STEPS={STEPS}")
    print(f"params={sum(v.numel() for v in P_init.values())}")
    print(f"perturbed loss (sum over {B}x{T} tokens) = {loss_p:.15e}")
    print(f"init loss per token = {loss_init / (B*T):.15e}  (log V = {math.log(V):.6f})")
    for k in ORDER:
        print(f"  |grad {k:13s}| = {g_p[k].norm().item():.15e}")
    for k in ORDER:
        print(f"  |step {k:13s}| = {(P_step[k] - P_pert[k]).norm().item():.15e}")
    for s, l in enumerate(losses):
        print(f"step {s:2d} loss/token {l:.12f}")

    with open(__file__.replace("reference.py", "ref.viv"), "w") as f:
        f.write("###\nGenerated by reference.py; do not edit. The corpus, the two weight sets and\n"
                "the numbers PyTorch computed from them.\n###\n\n")
        f.write(f"REF_V = {V}\n")
        f.write(f"REF_TOKENS = {viv_array(torch.tensor(TOKENS, dtype=torch.float64))}\n\n")
        for tag, P in (("init", P_init), ("pert", P_pert)):
            f.write(f"#| The {'file' if tag == 'init' else 'perturbed'} initial weights, by tensor, in the order of ORDER.\n")
            f.write(f"def ref_{tag}(k) {{\n")
            for i, name in enumerate(ORDER):
                f.write(f"    if k == {i} {{\n        return {viv_matrix(P[name])}\n    }}\n")
            f.write("    return zeros(1, 1)\n}\n\n")
        f.write(f"REF_PERT_LOSS = {loss_p!r}\n")
        f.write(f"REF_INIT_LOSS = {loss_init / (B * T)!r}\n")
        f.write(f"REF_GRAD_NORMS = {viv_array(torch.tensor([g_p[k].norm().item() for k in ORDER]))}\n")
        f.write(f"REF_STEP_NORMS = {viv_array(torch.tensor([(P_step[k] - P_pert[k]).norm().item() for k in ORDER]))}\n")
        f.write(f"REF_LOSSES = {viv_array(torch.tensor(losses))}\n")
        # a few full tensors, entry by entry
        f.write(f"REF_GRAD_QK1 = {viv_matrix(g_p['qk1'])}\n")
        f.write(f"REF_GRAD_EMBED = {viv_matrix(g_p['embed'])}\n")
        f.write(f"REF_STEP_FC0 = {viv_matrix(P_step['fc0'])}\n")
        f.write(f"REF_STEP_LM = {viv_matrix(P_step['lm_head'])}\n")
