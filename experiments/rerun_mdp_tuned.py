
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from arm_scheduler.core.generator import generate_block
from arm_scheduler.core.instruction import build_dependency_graph, validate_schedule
from arm_scheduler.solvers.bayesian import compute_total_expected_leakage
from arm_scheduler.solvers.mdp import MDPScheduler

CSV_HEADER = [
    "method", "n_instructions", "seed", "violation_penalty",
    "total_cycles", "n_nops", "n_violations", "expected_leakage",
    "wall_time", "train_time", "valid", "backend",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MDP reward-shaping ablation (violation_penalty rescaling)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--k", type=int, default=3,
                   help="Security distance (same as main benchmark).")
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 30, 50],
                   help="Block sizes to ablate over. Default mirrors the main "
                        "benchmark; expect ~9 h GPU end-to-end at penalty=-100.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                   help="Same seeds as main benchmark for direct comparison.")
    p.add_argument("--episodes", type=int, default=5_000,
                   help="DQN training episodes per (n, seed).")
    p.add_argument("--penalty", type=float, nargs="+", default=[-100.0],
                   help="violation_penalty value(s) to sweep. Default main "
                        "benchmark used -10.")
    p.add_argument("--stochastic", action="store_true",
                   help="Enable stochastic latency during training.")
    p.add_argument("--output", type=str,
                   default="experiments/results/benchmark_results_mdp_tuned.csv",
                   help="Destination CSV.")
    return p.parse_args()


def run_one(
    n: int, seed: int, k: int, episodes: int,
    violation_penalty: float, stochastic: bool,
) -> dict:
    instructions = generate_block(n=n, seed=seed)
    run_id = f"tuned_{int(violation_penalty)}_n{n}_s{seed}"

    solver = MDPScheduler(
        k=k,
        n_episodes=episodes,
        stochastic=stochastic,
        violation_penalty=violation_penalty,
    )

    # --- Training ---
    t_train0 = time.perf_counter()
    solver.train(instructions, verbose=True, run_id=run_id, n=n, seed=seed)
    train_time = time.perf_counter() - t_train0

    # --- Inference (greedy rollout) ---
    t_inf0 = time.perf_counter()
    schedule, total_cycles, stats = solver.schedule(instructions)
    wall_time = time.perf_counter() - t_inf0

    n_violations = stats.get("n_violations", -1)
    n_nops = stats.get("n_nops", sum(1 for _, i in schedule if i is None))
    backend = stats.get("backend", solver.backend)

    expected_leakage = compute_total_expected_leakage(schedule)
    predecessors = build_dependency_graph(instructions)
    ok, _errors = validate_schedule(schedule, instructions, predecessors, k)
    valid = bool(ok)

    return {
        "method": "mdp_tuned",
        "n_instructions": n,
        "seed": seed,
        "violation_penalty": violation_penalty,
        "total_cycles": total_cycles,
        "n_nops": n_nops,
        "n_violations": n_violations,
        "expected_leakage": expected_leakage,
        "wall_time": wall_time,
        "train_time": train_time,
        "valid": bool(valid),
        "backend": backend,
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("  MDP rerun with rescaled violation_penalty (ablation)")
    print("=" * 66)
    print(f"  k         : {args.k}")
    print(f"  sizes     : {args.sizes}")
    print(f"  seeds     : {args.seeds}")
    print(f"  episodes  : {args.episodes}")
    print(f"  penalties : {args.penalty}")
    print(f"  output    : {out_path}")
    print("=" * 66 + "\n")

    write_header = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()

        total_runs = len(args.penalty) * len(args.sizes) * len(args.seeds)
        i = 0
        for penalty in args.penalty:
            for n in args.sizes:
                for seed in args.seeds:
                    i += 1
                    print(f"[{i}/{total_runs}] n={n} seed={seed} "
                          f"penalty={penalty}")
                    row = run_one(
                        n=n, seed=seed, k=args.k,
                        episodes=args.episodes,
                        violation_penalty=penalty,
                        stochastic=args.stochastic,
                    )
                    writer.writerow(row)
                    f.flush()
                    print(f"  -> cycles={row['total_cycles']} "
                          f"nops={row['n_nops']} "
                          f"violations={row['n_violations']} "
                          f"E[L]={row['expected_leakage']:.2f} "
                          f"valid={row['valid']} "
                          f"train={row['train_time']:.1f}s")

    print(f"\nDone. Results appended to {out_path}")


if __name__ == "__main__":
    main()
