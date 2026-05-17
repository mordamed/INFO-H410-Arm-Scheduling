
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from ..core.generator import generate_block
from ..core.instruction import validate_schedule
from ..core.pipeline import PipelineState
from ..solvers.bayesian import BayesianScheduler, compute_total_expected_leakage
from ..solvers.csp import CSPScheduler
from ..solvers.mdp import MDPScheduler



# Result dataclass

@dataclass
class BenchmarkResult:
    method: str
    n_instructions: int
    seed: int
    total_cycles: int
    n_nops: int
    n_violations: int
    expected_leakage: float
    wall_time: float
    train_time: float
    optimal: bool
    valid: bool
    backend: str


# Single-run function 

def _run_once(
    method: str,
    n: int,
    seed: int,
    k: int,
    mdp_episodes: int,
    mdp_stochastic: bool,
    verbose: bool = False,
    pretrained_mdp: Optional["MDPScheduler"] = None,
) -> BenchmarkResult:
    instructions = generate_block(n=n, seed=seed)
    predecessors = PipelineState(instructions, k).predecessors
    train_time = 0.0
    backend = method
    optimal = False

    try:
        if method == "bayesian":
            solver = BayesianScheduler()
            schedule, total_cycles, stats = solver.schedule(instructions)
            optimal = stats.get("optimal", False)
            backend = stats.get("method", "bayesian")

        elif method == "csp":
            solver = CSPScheduler(k=k)
            schedule, total_cycles, stats = solver.schedule(instructions)
            optimal = stats.get("optimal", False)
            backend = stats.get("backend", "csp")

        elif method == "mdp":
            if pretrained_mdp is not None: #avoid retraining for each seed/size combo if we already have trained a model
                
                train_time = 0.0
                t_infer = time.perf_counter()
                schedule, total_cycles, stats = pretrained_mdp._agent.schedule_greedy(instructions)
                stats["wall_time"] = time.perf_counter() - t_infer
                backend = stats.get("backend", "mdp_generalist")
            else:
                solver = MDPScheduler(k=k, n_episodes=mdp_episodes, stochastic=mdp_stochastic)
                t_train = time.perf_counter()
                solver.train(instructions, verbose=verbose, run_id=f"n{n}_s{seed}", n=n, seed=seed)
                train_time = time.perf_counter() - t_train

                t_infer = time.perf_counter()
                schedule, total_cycles, stats = solver.schedule(instructions)
                stats["wall_time"] = time.perf_counter() - t_infer
                backend = stats.get("backend", "mdp")
        else:
            raise ValueError(f"Unknown method: {method}")

        wall_time = stats.get("wall_time", 0.0)
        n_nops = stats.get("n_nops", sum(1 for _, i in schedule if i is None))

        valid, errors = validate_schedule(schedule, instructions, predecessors, k)
        n_violations = sum(1 for e in errors if "Security" in e)
        
        # Calculate expected leakage for ALL approaches
        expected_leakage = compute_total_expected_leakage(schedule)

    except Exception as exc:
        return BenchmarkResult(
            method=method, n_instructions=n, seed=seed,
            total_cycles=-1, n_nops=-1, n_violations=-1,
            expected_leakage=-1.0,
            wall_time=-1.0, train_time=-1.0,
            optimal=False, valid=False, backend=f"ERROR: {exc}",
        )

    return BenchmarkResult(
        method=method, n_instructions=n, seed=seed,
        total_cycles=total_cycles, n_nops=n_nops,
        n_violations=n_violations, expected_leakage=expected_leakage, 
        wall_time=wall_time,
        train_time=train_time, optimal=optimal,
        valid=valid, backend=backend,
    )


# Main benchmark runner

def run_benchmark(
    sizes: List[int] = [10, 30, 50],
    seeds: List[int] = [42, 43, 44],
    k: int = 3,
    methods: List[str] = ["bayesian", "csp", "mdp"],
    mdp_episodes: int = 5_000,
    mdp_stochastic: bool = False,
    output_dir: str = "experiments/results",
    verbose: bool = True,
    n_jobs: int = 1,
    resume: bool = True,
    pretrained_mdp: Optional["MDPScheduler"] = None,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_csv) if output_csv else Path(output_dir) / "benchmark_results.csv"

    existing_combos = set()
    if resume and csv_path.exists():
        try:
            old_df = pd.read_csv(csv_path)
            for _, row in old_df.iterrows():
                existing_combos.add((row["method"], int(row["n_instructions"]), int(row["seed"])))
            if verbose and existing_combos:
                print(f"[Resume] Found {len(existing_combos)} completed jobs in {csv_path.name}.")
        except Exception as e:
            if verbose:
                print(f"[Resume] Error reading {csv_path.name}: {e}. Starting fresh.")
    elif not resume and csv_path.exists():
        csv_path.unlink()

    def append_result(res: BenchmarkResult):
        df_res = pd.DataFrame([asdict(res)])
        if not csv_path.exists():
            df_res.to_csv(csv_path, index=False)
        else:
            df_res.to_csv(csv_path, mode='a', header=False, index=False)

    # Build job list — MDP last (so we can batch train per size)
    non_mdp = [(m, n, s) for m in methods if m != "mdp"
               for n in sizes for s in seeds if (m, n, s) not in existing_combos]
    mdp_jobs = [(m, n, s) for m in methods if m == "mdp"
                for n in sizes for s in seeds if (m, n, s) not in existing_combos]
    combos = non_mdp + mdp_jobs


    # Determine parallelism
    import multiprocessing as mp
    import os
    if n_jobs == -1:
        n_jobs = max(1, os.cpu_count() - 1)
    # MDP with DQN cannot be parallelised (PyTorch CUDA context is not fork-safe)
    # So we split the work
    can_parallel = [c for c in combos if c[0] != "mdp"]
    must_serial = [c for c in combos if c[0] == "mdp"]

    results: List[BenchmarkResult] = []
    pbar_total = len(combos)

    print(f"\n[Phase 1/2] Standard Solvers (Bayesian, CSP)")
    print(f"Running {len(can_parallel)} parallel jobs on {n_jobs} CPU workers...")
    print("-" * 60)

    with tqdm(total=pbar_total, desc="Total Progress", unit="run", disable=not verbose) as pbar:

        # ----- Parallel A* / CSP -----
        if n_jobs > 1 and can_parallel:
            with mp.Pool(processes=min(n_jobs, len(can_parallel))) as pool:  #fork-safe multiprocessing pool
                futures = [
                    pool.apply_async(
                        _run_once,
                        args=(m, n, s, k, mdp_episodes, mdp_stochastic)
                    )
                    for m, n, s in can_parallel
                ]
                for (m, n, s), fut in zip(can_parallel, futures):
                    pbar.set_description(f"Current: {m} (n={n}, seed={s})")
                    result = fut.get()
                    results.append(result)
                    if verbose:
                        pbar.write(
                            f"  {m:8s}  n={n:2d}  seed={s}  ->  {result.total_cycles} cycles"
                        )
                    append_result(result)
                    pbar.update(1)
        else:
            for m, n, s in can_parallel:
                pbar.set_description(f"Current: {m} (n={n}, seed={s})")
                result = _run_once(m, n, s, k, mdp_episodes, mdp_stochastic)
                results.append(result)
                if verbose:
                    pbar.write(
                        f"  {m:8s}  n={n:2d}  seed={s}  ->  {result.total_cycles} cycles"
                    )
                append_result(result)
                pbar.update(1)

        # ----- Sequential MDP -----
        if must_serial:
            print(f"\n[Phase 2/2] Deep Reinforcement Learning (DQN Training)")
            print(f"Running {len(must_serial)} sequential training sessions on GPU...")
            print("-" * 60)

        for m, n, s in must_serial:
            label = "Evaluating" if pretrained_mdp is not None else "Training"
            pbar.set_description(f"{label}: MDP (n={n}, seed={s})")
            result = _run_once(m, n, s, k, mdp_episodes, mdp_stochastic, verbose=True, pretrained_mdp=pretrained_mdp)
            results.append(result)
            if verbose:
                pbar.write(
                    f"  {m:8s}  n={n:2d}  seed={s}  ->  {result.total_cycles} cycles (train={result.train_time:.1f}s)"
                )
            append_result(result)
            pbar.update(1)

    # Build DataFrame from the combined (old and new) CSV
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=[f.name for f in asdict(BenchmarkResult("", 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, False, False, ""))])
        df.to_csv(csv_path, index=False)


    if verbose:
        print(f"\nResults → {csv_path}")

    summary = _build_summary(df, methods, sizes)
    json_path = Path(output_dir) / "benchmark_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    if verbose:
        print(f"Summary → {json_path}")

    return df



# Summary + table helpers

def _build_summary(df, methods, sizes):
    
    summary = {}
    for method in methods:
        summary[method] = {}
        for n in sizes:
            sub = df[(df["method"] == method) & (df["n_instructions"] == n)]
            if sub.empty:
                continue
            summary[method][str(n)] = {
                "total_cycles_mean": float(sub["total_cycles"].mean()),
                "total_cycles_std":  float(sub["total_cycles"].std()),
                "n_nops_mean":       float(sub["n_nops"].mean()),
                "wall_time_mean":    float(sub["wall_time"].mean()),
                "train_time_mean":   float(sub["train_time"].mean()),
                "n_violations_mean": float(sub["n_violations"].mean()),
                "expected_leak_mean": float(sub["expected_leakage"].mean()),
                "valid_rate":        float(sub["valid"].mean()),
            }
    return summary


def print_summary_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 94)
    print(f"{'Method':8s}  {'n':>4}  {'Cycles (μ±σ)':>18}  {'NOPs':>6}  "
          f"{'Time (s)':>9}  {'Violations':>10}  {'Leakage(E)':>10}  {'Valid':>5}")
    print("=" * 94)
    for method in df["method"].unique():
        for n in sorted(df["n_instructions"].unique().tolist()):
            sub = df[(df["method"] == method) & (df["n_instructions"] == n)]
            if sub.empty:
                continue
            std_str = f"{sub['total_cycles'].std():5.1f}" if len(sub) > 1 else "  n/a"
            print(
                f"{method:8s}  {n:>4}  "
                f"{sub['total_cycles'].mean():6.1f} ± {std_str}  "
                f"{sub['n_nops'].mean():>6.1f}  "
                f"{sub['wall_time'].mean():>9.4f}  "
                f"{sub['n_violations'].mean():>10.2f}  "
                f"{sub['expected_leakage'].mean():>10.2f}  "
                f"{sub['valid'].mean():>5.1%}"
            )
        print("-" * 94)
    print()
