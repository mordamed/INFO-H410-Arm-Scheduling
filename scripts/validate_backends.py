import time
from arm_scheduler.core.generator import generate_block
from arm_scheduler.solvers.csp import CSPScheduler
from arm_scheduler.solvers.mdp import MDPScheduler, DEVICE, _TORCH

print(f"PyTorch: {_TORCH} | Device: {DEVICE}")
print()

csp = CSPScheduler(k=3)
for n in [10, 30, 50]:
    block = generate_block(n=n, seed=42)
    t0 = time.perf_counter()
    sched, cycles, stats = csp.schedule(block)
    elapsed = time.perf_counter() - t0
    print(f"CSP n={n:2d}: {cycles} cycles, {stats['n_nops']} NOPs, "
          f"{elapsed:.3f}s, optimal={stats['optimal']} [{stats['backend']}]")

print()
print("MDP DQN test n=10 (500 episodes)...")
mdp = MDPScheduler(k=3, n_episodes=500)
block = generate_block(n=10, seed=42)
t0 = time.perf_counter()
mdp.train(block)
train_t = time.perf_counter() - t0
sched, cycles, stats = mdp.schedule(block)
print(f"MDP n=10: {cycles} cycles, train={train_t:.1f}s, "
      f"infer={stats['wall_time']:.3f}s  [{stats['backend']}]")
