
from __future__ import annotations

import time
from typing import Dict, List, Optional, Set, Tuple

from ..core.instruction import Instruction, ShareType
from ..core.pipeline import PipelineState


# Bayesian Conditional Probability Table (CPT)


def get_cpt_prob(delta_t: int) -> float:
    if delta_t == 0:
        return 1.00 # Simultaneous leakage (should not happen in single-issue pipeline)
    if delta_t == 1:
        return 0.95
    if delta_t == 2:
        return 0.50
    if delta_t == 3:
        return 0.10
    return 0.00


def compute_marginal_leakage(
    candidate: Instruction, 
    candidate_cycle: int, 
    placement: Dict[int, int], 
    instructions: List[Instruction]
) -> float:
    if candidate.share_type == ShareType.NEUTRAL:
        return 0.0
        
    marginal_leakage = 0.0
    for placed_idx, placed_cycle in placement.items():
        placed_instr = instructions[placed_idx]
        if placed_instr.share_type == ShareType.NEUTRAL:
            continue
            
        # If shares are different (e.g. A and B), there is an Overlap risk
        if candidate.share_type != placed_instr.share_type:
            delta_t = abs(candidate_cycle - placed_cycle)
            marginal_leakage += get_cpt_prob(delta_t)
            
    return marginal_leakage


def compute_total_expected_leakage(
    schedule: List[Tuple[int, Optional[Instruction]]]
) -> float:
    total_leakage = 0.0
    # Extract only valid instructions placed in time
    placed_instructions = [(cycle, instr) for cycle, instr in schedule if instr is not None]
    
    for i, (cycle_a, instr_a) in enumerate(placed_instructions):
        if instr_a.share_type == ShareType.NEUTRAL:
            continue
        for cycle_b, instr_b in placed_instructions[i+1:]:
            if instr_b.share_type == ShareType.NEUTRAL:
                continue
            if instr_a.share_type != instr_b.share_type:
                total_leakage += get_cpt_prob(abs(cycle_a - cycle_b))
                
    return total_leakage



# Solver Class


class BayesianScheduler:

    def __init__(self, tau: float = 0.15, k: int = 3) -> None:
        # Note: k is kept as a parameter for compatibility with the generic constructor 
        # in the benchmark flow, but the Bayesian model relies on `tau` and the CPT.
        self.tau = tau
        self.k = k

    def schedule(
        self,
        instructions: List[Instruction],
    ) -> Tuple[List[Tuple[int, Optional[Instruction]]], int, Dict]:
        t0 = time.perf_counter()
        
        n = len(instructions)
        # We reuse PipelineState purely for extracting RAW dependencies (get_ready_instructions)
        # We DO NOT use its strict `is_security_valid` method.
        state = PipelineState(instructions, self.k)
        
        scheduled: Set[int] = set()
        finish_times: Dict[int, int] = {}
        placement: Dict[int, int] = {}
        sequence: List[Tuple[int, Optional[Instruction]]] = []
        cycle = 0

        while len(scheduled) < n:
            ready = state.get_ready_instructions(scheduled, finish_times, cycle)
            
            if not ready:
                # RAW hazard forces a NOP
                sequence.append((cycle, None))
                cycle += 1
                continue
                
            # Score all ready instructions based on their risk
            scored_candidates = []
            for instr in ready:
                risk = compute_marginal_leakage(instr, cycle, placement, instructions)
                scored_candidates.append((risk, instr))
            
            # Find the minimum risk available
            min_risk = min(scored_candidates, key=lambda x: x[0])[0]
            
            if min_risk > self.tau:
                # All ready instructions are too dangerous. Inject a NOP.
                # This increments cycle, increasing delta_t for the next loop iteration,
                # which decays the risk probability in the CPT.
                sequence.append((cycle, None))
                cycle += 1
            else:
                # Among those tied for minimum risk, break ties using the critical path heuristic
                best_candidates = [instr for risk, instr in scored_candidates if risk == min_risk]
                chosen = max(best_candidates, key=lambda i: state._critical_path[i.idx])
                
                scheduled.add(chosen.idx)
                finish_times[chosen.idx] = cycle + chosen.latency
                placement[chosen.idx] = cycle
                sequence.append((cycle, chosen))
                cycle += 1

        total_cycles = cycle
        nops = sum(1 for _, i in sequence if i is None)
        wall_time = time.perf_counter() - t0
        
        total_expected_leakage = compute_total_expected_leakage(sequence)

        stats = {
            "method": "bayesian",
            "backend": "bayesian",
            "optimal": False,
            "total_cycles": total_cycles,
            "n_nops": nops,
            "n_violations": -1, # Violations are not strictly defined for Bayesian
            "wall_time": wall_time,
            "expected_leakage": total_expected_leakage,
            "tau_threshold": self.tau
        }
        return sequence, total_cycles, stats
