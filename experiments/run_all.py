
from __future__ import annotations

import argparse
import sys
import os
import time

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from arm_scheduler.core.generator import generate_block, describe_block
from arm_scheduler.evaluation.benchmark import run_benchmark, print_summary_table
from arm_scheduler.evaluation.visualizer import generate_all_figures, plot_gantt
from arm_scheduler.solvers.bayesian import BayesianScheduler
from arm_scheduler.solvers.csp import CSPScheduler
from arm_scheduler.solvers.mdp import MDPScheduler



# Argument parsing and CLI interface

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ARM32 Instruction Scheduler — INFO-H410 Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # security distance
    parser.add_argument(
        "--k", type=int, default=3,
        help="Security distance (cycles) between different-share instructions.",
    )
    #sizes of instruction blocks 
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[10, 30, 50],
        help="Block sizes (number of instructions) to benchmark.",
    )
    # seeds to generate random insturction blocks
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44],
        help="Random seeds (one run per seed × size × method).",
    )
    # Q-Learning training episodes
    parser.add_argument(
        "--episodes", type=int, default=5_000,
        help="Q-Learning training episodes (MDP approach).",
    )
    #stochastic latency
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Enable stochastic latency during MDP training.",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["bayesian", "csp", "mdp"],
        choices=["bayesian", "csp", "mdp"],
        help="Solvers to include.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/results",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke test: skip large blocks, reduce episodes.",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume benchmark from existing CSV results instead of restarting.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip figure generation (useful in headless CI environments).",
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Simultaneously write output to this file (Tee behavior).",
    )
    return parser.parse_args()



# Tee Logger (Terminal + File)

class TeeLogger:
    def __init__(self, filename: str, stream):
        self.stream = stream
        self.file = open(filename, "a", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()



# Main

def main() -> None:
    args = parse_args()

    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        # Mirror stdout and stderr
        sys.stdout = TeeLogger(args.log, sys.stdout)
        sys.stderr = TeeLogger(args.log, sys.stderr)
        print(f"--- Logging started to {args.log} ---")

#make fast smoke test mode 
    if args.quick:
        args.sizes = [10]
        args.seeds = [42]
        args.episodes = 300
        print("[quick mode] sizes=10, seeds=[42], episodes=300")

    print("\n" + "=" * 60)
    print("  ARM32 Instruction Scheduler — INFO-H410 Benchmark")
    print("=" * 60)
    print(f"  k (security distance) : {args.k}")
    print(f"  Block sizes           : {args.sizes}")
    print(f"  Seeds                 : {args.seeds}")
    print(f"  MDP episodes          : {args.episodes}")
    print(f"  Stochastic latency    : {args.stochastic}")
    print(f"  Solvers               : {args.methods}")
    print(f"  Output directory      : {args.output_dir}")
    print("=" * 60 + "\n")

    # Step 1: Show an example block
    
    print("── Example Block (n=10, seed=42) ──")
    example_instructions = generate_block(n=10, seed=42)
    describe_block(example_instructions)

    # Step 2: Run benchmark
  
    print("── Running Benchmark ──")
    t_start = time.perf_counter()

    df = run_benchmark(
        sizes=args.sizes,
        seeds=args.seeds,
        k=args.k,
        methods=args.methods,
        mdp_episodes=args.episodes,
        mdp_stochastic=args.stochastic,
        output_dir=args.output_dir,
        verbose=True,
        resume=args.resume,
    )

    t_bench = time.perf_counter() - t_start
    print(f"\nTotal benchmark time: {t_bench:.1f}s\n")

    
    # Step 3: Print summary table
    
    print("── Results Summary ──")
    print_summary_table(df)

    
    # Step 4: Generate example schedules for Gantt plots
   
    example_schedules = []

    # Train MDP once to get learning curve
    rewards = None
    if "mdp" in args.methods:
        print("── Generating MDP Learning Curve (n=10, seed=42) ──")
        mdp = MDPScheduler(k=args.k, n_episodes=args.episodes, stochastic=args.stochastic)
        rewards = mdp.train(example_instructions, verbose=True)

    # CSP Gantt
    if "csp" in args.methods:
        solver = CSPScheduler(k=args.k)
        sched, _, _ = solver.schedule(example_instructions)
        example_schedules.append(
            (sched, example_instructions, "CSP Schedule (n=10, seed=42)", "fig5_gantt_csp.png")
        )

    # MDP Gantt
    if "mdp" in args.methods and rewards is not None:
        sched, _, _ = mdp.schedule(example_instructions)
        example_schedules.append(
            (sched, example_instructions, "MDP (Q-Learning) Schedule (n=10, seed=42)", "fig5_gantt_mdp.png")
        )

  
    # Step 5: Generate figures
    if not args.no_plots:
        print("── Generating Figures ──")

        # Main comparison figures
        generate_all_figures(
            df=df,
            rewards=rewards,
            output_dir=args.output_dir,
            verbose=True,
        )

        # Individual Gantt diagrams
        for sched, instrs, title, fname in example_schedules:
            path = plot_gantt(
                sched, instrs, title=title,
                output_dir=args.output_dir, filename=fname,
            )
            print(f"  Saved: {path}")

        # (1) Bayesian Gantt Chart
        if "bayesian" in args.methods:
            try:
                print("Generating Bayesian Gantt Chart...")
                block_example = generate_block(n=min(10, args.sizes[0]), seed=args.seeds[0])
                solver = BayesianScheduler()
                out, total, _ = solver.schedule(block_example)
                plot_gantt(out, block_example, 
                           title=f"Bayesian Schedule (n={len(block_example)}, tau=0.15)",
                           output_dir=args.output_dir, 
                           filename="fig5_gantt_bayesian.png")
            except Exception as e:
                print(f"  Warning: Could not generate Bayesian Gantt plot: {e}")

    print("\n── Done ──")
    print(f"All results are in: {os.path.abspath(args.output_dir)}\n")


if __name__ == "__main__":
    main()
