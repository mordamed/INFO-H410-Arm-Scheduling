
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from arm_scheduler.solvers.mdp import MDPScheduler
from arm_scheduler.evaluation.benchmark import run_benchmark, print_summary_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generalist DQN: train on corpus, evaluate on benchmark seeds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--k", type=int, default=3,
                   help="Security distance.")
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 30, 50],
                   help="Block sizes in the training corpus AND at evaluation.")
    p.add_argument("--train-seeds", type=int, nargs="+", default=list(range(100, 130)),
                   help="Seeds for corpus generation. Must not include test seeds.")
    p.add_argument("--test-seeds", type=int, nargs="+", default=[42, 43, 44],
                   help="Benchmark seeds (same as standard benchmark).")
    p.add_argument("--episodes", type=int, default=20_000,
                   help="Total training episodes across all corpus blocks.")
    p.add_argument("--penalty", type=float, default=-100.0,
                   help="Violation penalty. Divided by n in shaped_quality mode — "
                        "use ≤ -100 so the normalized penalty remains significant.")
    p.add_argument("--methods", nargs="+", default=["bayesian", "csp", "mdp"],
                   choices=["bayesian", "csp", "mdp"],
                   help="Solvers to include in the benchmark phase.")
    p.add_argument("--checkpoint", type=str,
                   default="experiments/checkpoints/generalist.pt",
                   help="Path to save/resume the generalist model.")
    p.add_argument("--output-dir", type=str,
                   default="experiments/results",
                   help="Directory for benchmark CSV and summary.")
    p.add_argument("--output-csv", type=str,
                   default="experiments/results/benchmark_generalist.csv",
                   help="CSV file for generalist benchmark results.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    #Vérifie si les seeds de train et de test se chevauchent pour séparer les datas de train et de test 
    overlap = set(args.train_seeds) & set(args.test_seeds)
    if overlap:
        print(f"[ERROR] Training seeds overlap with test seeds: {overlap}")
        print("Remove the conflicting seeds from --train-seeds to avoid data leakage.")
        sys.exit(1)

    print("=" * 66)
    print("  Generalist DQN — corpus training + benchmark evaluation")
    print("=" * 66)
    print(f"  Reward mode    : shaped_quality (normalized by n)")
    print(f"  k              : {args.k}")
    print(f"  Sizes          : {args.sizes}")
    print(f"  Train seeds    : {args.train_seeds[:5]}{'...' if len(args.train_seeds) > 5 else ''} ({len(args.train_seeds)} total)")
    print(f"  Test seeds     : {args.test_seeds}")
    print(f"  Episodes       : {args.episodes}")
    print(f"  Penalty        : {args.penalty}  (per-violation, normalized by n)")
    print(f"  Checkpoint     : {args.checkpoint}")
    print("=" * 66 + "\n")

    
    # Phase 1: Train generalist
    
    print("── Phase 1: Generalist Training ──")

    solver = MDPScheduler(
        k=args.k,
        violation_penalty=args.penalty,
        reward_mode="shaped_quality",
    )

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    rewards = solver.train_generalist(
        sizes=args.sizes,
        train_seeds=args.train_seeds,
        n_episodes=args.episodes,
        verbose=True,
        checkpoint_path=args.checkpoint,
    )
    train_elapsed = time.perf_counter() - t0

    print(f"\nTraining complete in {train_elapsed:.1f}s  "
          f"({train_elapsed / max(args.episodes, 1) * 1000:.1f} ms/episode)")

    solver.save(args.checkpoint)
    print(f"Model saved → {args.checkpoint}")

    
    # Phase 2: Benchmark on test seeds (inference only)
   
    print("\n── Phase 2: Benchmark (inference only, no retraining) ──")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = run_benchmark(
        sizes=args.sizes,
        seeds=args.test_seeds,
        k=args.k,
        methods=args.methods,
        output_dir=args.output_dir,
        verbose=True,
        resume=False,
        pretrained_mdp=solver if "mdp" in args.methods else None,
        output_csv=str(out_path),
    )

    print(f"Results → {out_path}")

    print("\n── Results Summary ──")
    print_summary_table(df)

    print(f"\nDone. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
