
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .instruction import Instruction, ShareType, build_dependency_graph


_NOT_PLACED = -1

class PipelineState:

    def __init__(self, instructions: List[Instruction], k: int = 3) -> None:
        self.instructions: List[Instruction] = instructions
        self.k: int = k
        self.n: int = len(instructions)

        # Dependency graph: predecessors[j] = [i, ...] (i must finish before j)
        self.predecessors: Dict[int, List[int]] = build_dependency_graph(instructions)

        # Successors: successors[i] = [j, ...] (j depends on i)
        self.successors: Dict[int, List[int]] = {i: [] for i in range(self.n)}
        for j, preds in self.predecessors.items():
            for i in preds:
                self.successors[i].append(j)

        # Index map for fast lookup: idx → Instruction
        self.idx_map: Dict[int, Instruction] = {instr.idx: instr for instr in instructions}

        # Critical-path lengths (cycles from each node to schedule end)
        # Used as an admissible A* heuristic (never overestimates remaining work)
        self._critical_path: Dict[int, int] = self._compute_critical_paths()

    
    # Critical path (for A* heuristic)
  

    def _compute_critical_paths(self) -> Dict[int, int]:
       
        in_degree = {i: len(self.predecessors[i]) for i in range(self.n)}
        queue = [i for i, d in in_degree.items() if d == 0]
        topo: List[int] = []
        temp = dict(in_degree)
        while queue:
            node = queue.pop(0)
            topo.append(node)
            for succ in self.successors[node]:
                temp[succ] -= 1
                if temp[succ] == 0:
                    queue.append(succ)

        cp: Dict[int, int] = {}
        for idx in reversed(topo):
            instr = self.idx_map[idx]
            if not self.successors[idx]:
                cp[idx] = instr.latency
            else:
                cp[idx] = instr.latency + max(cp[s] for s in self.successors[idx])
        return cp

    def heuristic(self, remaining: FrozenSet[int]) -> int:
        if not remaining:
            return 0
        return max(self._critical_path[idx] for idx in remaining)

    
    # Ready instruction query; used in bayes mdp and csp solvers
 
    def get_ready_instructions(
        self,
        scheduled: Set[int],
        finish_times: Dict[int, int],
        current_cycle: int,
    ) -> List[Instruction]:
        ready: List[Instruction] = []
        for instr in self.instructions:
            if instr.idx in scheduled:
                continue
            preds = self.predecessors[instr.idx]
            if all(
                p in finish_times and finish_times[p] <= current_cycle
                for p in preds
            ):
                ready.append(instr)
        return ready

    
    # Security constraint check
  
    def is_security_valid(
        self,
        instr: Instruction,
        cycle: int,
        placement: Dict[int, int],     # {idx: start_cycle} of already placed instrs
    ) -> bool:
        if instr.share_type == ShareType.NEUTRAL:
            return True  # NEUTRAL instructions never cause violations

        for idx, start in placement.items():
            other = self.idx_map[idx]
            if other.share_type == ShareType.NEUTRAL:
                continue
            if other.share_type != instr.share_type:
                if abs(start - cycle) < self.k:
                    return False
        return True

   
    # Convenience: earliest possible start for each instruction
  
#not used
    def earliest_starts(self, finish_times: Dict[int, int]) -> Dict[int, int]:
        result: Dict[int, int] = {}
        for instr in self.instructions:
            preds = self.predecessors[instr.idx]
            if not preds:
                result[instr.idx] = 0
            else:
                result[instr.idx] = max(
                    finish_times.get(p, 0) for p in preds
                )
        return result

