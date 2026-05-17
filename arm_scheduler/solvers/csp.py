
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from ..core.instruction import Instruction, ShareType
from ..core.pipeline import PipelineState

TIME_LIMIT = 15.0   # seconds per solve call (shortened to prevent benchmark stalls)


# Backend detection

try:
    from ortools.sat.python import cp_model as _cp_model
    _ORTOOLS = True
except (ImportError, OSError):
    _ORTOOLS = False

if not _ORTOOLS:
    try:
        from constraint import Problem, AllDifferentConstraint  # type: ignore
        _PYCONSTRAINT = True
    except ImportError:
        _PYCONSTRAINT = False
else:
    _PYCONSTRAINT = False


class CSPScheduler:

    def __init__(self, k: int = 3, time_limit: float = TIME_LIMIT) -> None:
        self.k = k
        self.time_limit = time_limit
        if not _ORTOOLS and not _PYCONSTRAINT:
            raise RuntimeError(
                "No CSP backend found. Install OR-Tools:  pip install ortools"
            )

    
    # Public API
    
    def schedule(
        self,
        instructions: List[Instruction],
    ) -> Tuple[List[Tuple[int, Optional[Instruction]]], int, Dict]:
        t0 = time.perf_counter()
        state = PipelineState(instructions, self.k)
        n = len(instructions)

        if _ORTOOLS:
            schedule_out, makespan, extra = self._solve_ortools(instructions, state, t0)
            backend = "ortools_cpsat"
        else:
            schedule_out, makespan, extra = self._solve_pyconstraint(instructions, state, t0)
            backend = "python_constraint"

        wall_time = time.perf_counter() - t0
        nops = sum(1 for _, i in schedule_out if i is None)

        stats = {
            "method": "csp",
            "backend": backend,
            "total_cycles": makespan,
            "n_nops": nops,
            "wall_time": wall_time,
            "optimal": extra.get("optimal", True),
        }
        return schedule_out, makespan, stats

    
    # OR-Tools CP-SAT backend (fast, recommended)


    def _solve_ortools(
        self,
        instructions: List[Instruction],
        state: PipelineState,
        t0: float,
    ) -> Tuple[List, int, Dict]:
        cp = _cp_model.CpModel()
        n = len(instructions)

        # Warm-start: greedy gives a tight upper bound 
        greedy_sched, greedy_total = self._greedy_fallback(state, instructions)
        # Any optimal solution fits within [0, greedy_total)
        t_max = greedy_total

        # Decision variables 
        t = [cp.new_int_var(0, t_max - 1, f"t_{i}") for i in range(n)]

        # Constraint 1: No slot collision 
        cp.add_all_different(t)

        #  Constraint 2: RAW data hazards 
        for j, preds in state.predecessors.items():
            for i in preds:
                cp.add(t[j] >= t[i] + instructions[i].latency)

        #  Constraint 3: Security distance 
        # Only SHARE_A vs SHARE_B pairs: half-reified for solver efficiency
        share_A = [i for i in instructions if i.share_type == ShareType.SHARE_A]
        share_B = [i for i in instructions if i.share_type == ShareType.SHARE_B]
        for ia in share_A:
            for ib in share_B:
                b = cp.new_bool_var(f"o_{ia.idx}_{ib.idx}")
                cp.add(t[ia.idx] - t[ib.idx] >= self.k).only_enforce_if(b)
                cp.add(t[ib.idx] - t[ia.idx] >= self.k).only_enforce_if(~b)

        #  Objective: minimise makespan 
        makespan_var = cp.new_int_var(0, t_max, "makespan")
        cp.add_max_equality(
            makespan_var,
            [t[i] + instructions[i].latency for i in range(n)]
        )
        cp.minimize(makespan_var)

        #  Hint from greedy (warm start) 
        greedy_placement = {
            instr.idx: cycle
            for cycle, instr in greedy_sched if instr is not None
        }
        for i, instr in enumerate(instructions):
            if instr.idx in greedy_placement:
                cp.add_hint(t[i], greedy_placement[instr.idx])

        #  Solve 
        solver = _cp_model.CpSolver()
        remaining = max(self.time_limit - (time.perf_counter() - t0), 1.0)
        solver.parameters.max_time_in_seconds = min(remaining, self.time_limit)
        # num_workers = 1 to avoid thread oversubscription when run in multiprocessing pool
        solver.parameters.num_workers = 1
        solver.parameters.log_search_progress = False

        status = solver.solve(cp)

        if status in (_cp_model.OPTIMAL, _cp_model.FEASIBLE):
            assignment = {instructions[i].idx: solver.value(t[i]) for i in range(n)}
            real_makespan = int(solver.objective_value)
            schedule_out = self._assignment_to_schedule(
                assignment, instructions, real_makespan
            )
            optimal = status == _cp_model.OPTIMAL
            return schedule_out, real_makespan, {
                "method": "csp",
                "backend": "ortools_cpsat",
                "optimal": optimal,
                "total_cycles": real_makespan,
                "n_nops": sum(1 for _, i in schedule_out if i is None),
                "n_violations": 0,
                "wall_time": time.perf_counter() - t0,
            }
        else:
            # Timeout without feasible solution → return greedy guarantee
            return greedy_sched, greedy_total, {"optimal": False}



    
    # python-constraint backend (slow fallback, small n only)

    def _solve_pyconstraint(
        self,
        instructions: List[Instruction],
        state: PipelineState,
        t0: float,
    ) -> Tuple[List, int, Dict]:
        from constraint import Problem, AllDifferentConstraint  # type: ignore

        n = len(instructions)
        best = None
        t_max = n

        while time.perf_counter() - t0 < self.time_limit:
            solution = self._pyconstraint_for_tmax(instructions, state, t_max)
            if solution is not None:
                makespan = max(
                    solution[instr.idx] + instr.latency for instr in instructions
                )
                best = (solution, makespan)
                break
            t_max += 1
            if t_max > n * 5:
                break

        if best is None:
            sched, total = self._greedy_fallback(state, instructions)
            return sched, total, {"optimal": False}

        solution, makespan = best
        schedule_out = self._assignment_to_schedule(solution, instructions, makespan)
        return schedule_out, makespan, {"optimal": True}

    def _pyconstraint_for_tmax(self, instructions, state, t_max):
        from constraint import Problem, AllDifferentConstraint  # type: ignore
        n = len(instructions)
        problem = Problem()
        domain = list(range(t_max))
        for instr in instructions:
            problem.addVariable(instr.idx, domain)
        problem.addConstraint(AllDifferentConstraint())
        k = self.k
        for j, preds in state.predecessors.items():
            for i in preds:
                lat = instructions[i].latency
                def raw_c(ti, tj, _lat=lat): return tj >= ti + _lat
                problem.addConstraint(raw_c, (instructions[i].idx, instructions[j].idx))
        share_instrs = [i for i in instructions if i.share_type != ShareType.NEUTRAL]
        for a in range(len(share_instrs)):
            ia = share_instrs[a]
            for ib in share_instrs[a + 1:]:
                if ia.share_type != ib.share_type:
                    def sec_c(ta, tb, _k=k): return abs(ta - tb) >= _k
                    problem.addConstraint(sec_c, (ia.idx, ib.idx))
        return problem.getSolution()

    
    # Shared helpers
    
    @staticmethod
    def _assignment_to_schedule(
        assignment: Dict[int, int],
        instructions: List[Instruction],
        makespan: int,
    ) -> List[Tuple[int, Optional[Instruction]]]:
        instr_map = {instr.idx: instr for instr in instructions}
        cycle_to_instr = {cycle: instr_map[idx] for idx, cycle in assignment.items()}
        return [(c, cycle_to_instr.get(c, None)) for c in range(makespan)]

    def _greedy_fallback(
        self,
        state: PipelineState,
        instructions: List[Instruction],
    ) -> Tuple[List, int]:
        n = len(instructions)
        scheduled, finish_times, placement = set(), {}, {}
        sequence, cycle = [], 0
        while len(scheduled) < n:
            ready = state.get_ready_instructions(scheduled, finish_times, cycle)
            valid = [i for i in ready if state.is_security_valid(i, cycle, placement)]
            if valid:
                chosen = max(valid, key=lambda i: state._critical_path[i.idx])
                scheduled.add(chosen.idx)
                finish_times[chosen.idx] = cycle + chosen.latency
                placement[chosen.idx] = cycle
                sequence.append((cycle, chosen))
            else:
                sequence.append((cycle, None))
            cycle += 1
        return sequence, cycle
