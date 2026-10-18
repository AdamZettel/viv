# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The reference run for ppo.viv: CleanRL's own `Agent` and `layer_init`
(imported from the clone at viv-wt/src/cleanrl-ppo, not copied), gymnasium's own
CartPoleEnv.step, torch autograd, torch.optim.Adam and clip_grad_norm_, and the
training loop of cleanrl/ppo.py lines 185-297 at a small size.

Every random draw comes from Viv's generator (Park-Miller, 16807 mod 2^31-1,
seeded by srand), reimplemented here, so both programs see the same weights,
the same resets, the same actions and the same minibatches:

  - weights: torch's orthogonal_ algorithm (QR of a Gaussian matrix, signs fixed
    by diag R) with the Gaussian drawn by Box-Muller from the shared stream;
    gains sqrt(2), sqrt(2), 1 (critic) and sqrt(2), sqrt(2), 0.01 (actor), zero biases
  - reset: uniform(-0.05, 0.05) per state entry (gymnasium: np_random.uniform)
  - action: inverse CDF of the policy at one uniform draw (ppo.py: probs.sample())
  - minibatches: Fisher-Yates (ppo.py: np.random.shuffle)

Two more departures, both so the numbers can agree to rounding:
  - float64 throughout, and the policy sees the float64 state; gymnasium hands
    back a float32 copy of it
  - the vector env is written out here (same-step autoreset, a 500-step time
    limit and the episode return, as gymnasium 0.29's SyncVectorEnv,
    TimeLimit and RecordEpisodeStatistics do) around gymnasium's CartPoleEnv

    python3 reference.py            # the run, in the format ppo.viv prints
    python3 reference.py --digits   # plus the 17-digit values the tests use
"""
import math
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
CLONE = os.path.join(HERE, "..", "..", "..", "..", "src", "cleanrl-ppo", "cleanrl")
sys.modules.setdefault("tyro", types.ModuleType("tyro"))
sys.modules.setdefault("torch.utils.tensorboard", types.SimpleNamespace(SummaryWriter=None))
sys.path.insert(0, CLONE)
from ppo import Agent  # noqa: E402  (CleanRL's own network)
from gymnasium.envs.classic_control.cartpole import CartPoleEnv  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
DIGITS = "--digits" in sys.argv

# ---- the sizes (the same constants sit at the top of ppo.viv) ----
SEED = 1
NUM_ENVS = 4
NUM_STEPS = 32
NUM_ITERATIONS = 40
LEARNING_RATE = 2.5e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
NUM_MINIBATCHES = 4
UPDATE_EPOCHS = 4
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
MAX_EPISODE_STEPS = 500
BATCH_SIZE = NUM_ENVS * NUM_STEPS
MINIBATCH_SIZE = BATCH_SIZE // NUM_MINIBATCHES


class ParkMiller:
    """Viv's srand/rand, as the C runtime writes them."""

    def __init__(self):
        self.state = 12345.0

    def srand(self, s):
        m = 2147483646.0
        v = abs(s)
        v = v - m * math.trunc(v / m)
        self.state = math.trunc(v) + 1.0

    def rand(self):
        p = 16807.0 * self.state
        self.state = p - 2147483647.0 * math.trunc(p / 2147483647.0)
        return self.state / 2147483647.0

    def randn(self):
        u1 = self.rand()
        u2 = self.rand()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * 3.141592653589793 * u2)


rng = ParkMiller()


def g(x):
    """C's %g, which is what Viv's `show` prints (and never a negative zero)."""
    return "%g" % (x + 0.0)


def show(label, x):
    print(f"{label}: {g(x)}")


# ---- torch.nn.init.orthogonal_, with the Gaussian from the shared stream ----
def orthogonal(rows, cols, gain):
    flat = torch.tensor([[rng.randn() for _ in range(cols)] for _ in range(rows)])
    if rows < cols:
        flat = flat.t()
    q, r = torch.linalg.qr(flat)
    q = q * torch.sign(torch.diag(r, 0))
    if rows < cols:
        q = q.t()
    return gain * q


def init_agent(agent):
    layers = [agent.critic[0], agent.critic[2], agent.critic[4], agent.actor[0], agent.actor[2], agent.actor[4]]
    gains = [math.sqrt(2), math.sqrt(2), 1.0, math.sqrt(2), math.sqrt(2), 0.01]
    with torch.no_grad():
        for layer, gain in zip(layers, gains):
            out_f, in_f = layer.weight.shape
            layer.weight.copy_(orthogonal(out_f, in_f, gain))
            layer.bias.zero_()


# ---- gymnasium 0.29 SyncVectorEnv + TimeLimit + RecordEpisodeStatistics ----
class VecCartPole:
    def __init__(self, n):
        self.envs = [CartPoleEnv() for _ in range(n)]
        self.elapsed = [0] * n
        self.ep_return = [0.0] * n
        self.ep_length = [0] * n
        self.single_observation_space = types.SimpleNamespace(shape=(4,))
        self.single_action_space = types.SimpleNamespace(n=2, shape=())

    def reset_one(self, i):
        env = self.envs[i]
        env.state = np.array([-0.05 + 0.1 * rng.rand() for _ in range(4)])
        env.steps_beyond_terminated = None
        self.elapsed[i] = 0
        self.ep_return[i] = 0.0
        self.ep_length[i] = 0
        return list(env.state)

    def reset(self):
        return np.array([self.reset_one(i) for i in range(len(self.envs))])

    def step(self, actions):
        obs, rewards, dones, finished = [], [], [], []
        for i, env in enumerate(self.envs):
            _, reward, terminated, _, _ = env.step(int(actions[i]))
            self.elapsed[i] += 1
            truncated = self.elapsed[i] >= MAX_EPISODE_STEPS
            self.ep_return[i] += reward
            self.ep_length[i] += 1
            state = list(env.state)
            if terminated or truncated:
                finished.append(self.ep_return[i])
                state = self.reset_one(i)
            obs.append(state)
            rewards.append(reward)
            dones.append(1.0 if (terminated or truncated) else 0.0)
        return np.array(obs), np.array(rewards), np.array(dones), finished


# ---- GAE, ppo.py lines 219-231 as a function of the rollout ----
def gae(rewards, values, dones, next_value, next_done, num_steps):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
    return advantages


def gae_on_a_fixed_trajectory():
    """Six steps of three envs, with an episode ending mid-rollout in two of them."""
    rng.srand(7)
    T, N = 6, 3
    rewards = torch.ones(T, N)
    values = torch.tensor([[10.0 * rng.rand() for _ in range(N)] for _ in range(T)])
    dones = torch.zeros(T, N)
    dones[2, 0] = 1.0
    dones[4, 1] = 1.0
    dones[5, 1] = 1.0
    next_value = torch.tensor([[10.0 * rng.rand() for _ in range(N)]])
    next_done = torch.tensor([0.0, 1.0, 0.0])
    adv = gae(rewards, values, dones, next_value, next_done, T)
    print("narrate: GAE on a fixed trajectory")
    for t in range(T):
        print("adv: [" + ", ".join(g(a) for a in adv[t].tolist()) + "]")
    if DIGITS:
        print("gae digits", [f"{a:.17g}" for a in adv.reshape(-1).tolist()])


def main():
    gae_on_a_fixed_trajectory()
    rng.srand(SEED)
    envs = VecCartPole(NUM_ENVS)
    agent = Agent(envs)
    init_agent(agent)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    obs = torch.zeros((NUM_STEPS, NUM_ENVS, 4))
    actions = torch.zeros((NUM_STEPS, NUM_ENVS))
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS))
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS))
    dones = torch.zeros((NUM_STEPS, NUM_ENVS))
    values = torch.zeros((NUM_STEPS, NUM_ENVS))

    global_step = 0
    next_obs = torch.tensor(envs.reset())
    next_done = torch.zeros(NUM_ENVS)
    for iteration in range(1, NUM_ITERATIONS + 1):
        frac = 1.0 - (iteration - 1.0) / NUM_ITERATIONS
        lrnow = frac * LEARNING_RATE
        optimizer.param_groups[0]["lr"] = lrnow
        finished = []
        for step in range(0, NUM_STEPS):
            global_step += NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                probs = torch.softmax(agent.actor(next_obs), dim=1)
                action = torch.tensor([0.0 if rng.rand() < probs[i, 0].item() else 1.0 for i in range(NUM_ENVS)])
                action, logprob, _, value = agent.get_action_and_value(next_obs, action)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            o, r, d, fin = envs.step(action.tolist())
            finished += fin
            rewards[step] = torch.tensor(r)
            next_obs, next_done = torch.tensor(o), torch.tensor(d)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = gae(rewards, values, dones, next_value, next_done, NUM_STEPS)
            returns = advantages + values

        b_obs = obs.reshape((-1, 4))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(BATCH_SIZE)
        clipfracs = []
        for epoch in range(UPDATE_EPOCHS):
            for i in range(BATCH_SIZE - 1, 0, -1):  # np.random.shuffle(b_inds)
                j = int(rng.rand() * (i + 1))
                b_inds[i], b_inds[j] = b_inds[j], b_inds[i]
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -CLIP_COEF, CLIP_COEF)
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        y_pred, y_true = b_values.numpy(), b_returns.numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        show("iteration", iteration)
        show("global_step", global_step)
        show("episodes", len(finished))
        show("mean_return", sum(finished) / len(finished) if finished else 0.0)
        show("value_loss", v_loss.item())
        show("policy_loss", pg_loss.item())
        show("entropy", entropy_loss.item())
        show("old_approx_kl", old_approx_kl.item())
        show("approx_kl", approx_kl.item())
        show("clipfrac", float(np.mean(clipfracs)))
        show("explained_variance", explained_var)
        if DIGITS and iteration in (1, 2, NUM_ITERATIONS):
            print("digits", iteration, [f"{x:.17g}" for x in (v_loss.item(), pg_loss.item(), entropy_loss.item(), approx_kl.item(), explained_var)])
    if DIGITS:
        w = agent.actor[4].weight.detach()
        print("final actor out weight", [f"{x:.17g}" for x in w[:, :3].reshape(-1).tolist()])


if __name__ == "__main__":
    main()
