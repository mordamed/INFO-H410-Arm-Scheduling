
from __future__ import annotations

import random
from typing import List, Tuple

from .instruction import Instruction, ShareType

# ---------------------------------------------------------------------------
# ARM32 opcode table: {name: (latency, has_two_sources)}
# ---------------------------------------------------------------------------
_OPS: dict[str, Tuple[int, bool]] = {
    "LDR": (2, False),   # load: 2-cycle latency, one base-register source
    "STR": (1, True),    # store: 1-cycle, src + base
    "ADD": (1, True),    # Rd = Rn + Rm
    "SUB": (1, True),    # Rd = Rn - Rm
    "EOR": (1, True),    # Rd = Rn ^ Rm  (critical for masking!)
    "AND": (1, True),    # Rd = Rn & Rm
    "MOV": (1, False),   # Rd = Rm
    "MUL": (2, True),    # Rd = Rn * Rm  (2-cycle multiplier)
    "ORR": (1, True),    # Rd = Rn | Rm
    "LSL": (1, True),    # Rd = Rn << Rm
}

_REGISTERS: List[str] = [f"r{i}" for i in range(13)]   # r0–r12


# Public API
"""Generate a reproducible block of *n* ARM32 instructions.

Parameters
----------
n              : Number of instructions in the block.
seed           : Random seed (guarantees reproducibility).
share_a_ratio  : Fraction of instructions tagged SHARE_A.
share_b_ratio  : Fraction of instructions tagged SHARE_B
                    (remainder → NEUTRAL).
dep_probability: Probability that a source register is chosen from the
                    set of already-written registers (creates RAW hazards).

Returns
-------
List[Instruction] with idx 0, 1, …, n-1 in their original order.
"""
def generate_block(
    n: int,
    seed: int = 42,
    share_a_ratio: float = 0.35,
    share_b_ratio: float = 0.35,
    dep_probability: float = 0.45,
) -> List[Instruction]:

    rng = random.Random(seed)
    instructions: List[Instruction] = []

    # Tracks (register, writer_idx) pairs for RAW dependency injection
    written_regs: List[Tuple[str, int]] = []

    ops = list(_OPS.keys())

    for i in range(n):
       
        op = rng.choice(ops)
        latency, two_srcs = _OPS[op]
        dest = rng.choice(_REGISTERS)
        srcs: List[str] = []
        if two_srcs or op in ("LDR",):
            # First source: possibly create a RAW dependency
            if written_regs and rng.random() < dep_probability:
                dep_reg, _ = rng.choice(written_regs)
                srcs.append(dep_reg)
            else:
                srcs.append(rng.choice(_REGISTERS))
            if two_srcs and op != "STR":
                if written_regs and rng.random() < dep_probability * 0.5:
                    dep_reg, _ = rng.choice(written_regs)
                    srcs.append(dep_reg)
                else:
                    srcs.append(rng.choice(_REGISTERS))
        # STR needs value + base
        elif op == "STR":
            srcs.append(rng.choice(_REGISTERS))   # value
            srcs.append(rng.choice(_REGISTERS))   # base address

        roll = rng.random()  #determine share type based on ratios
        if roll < share_a_ratio:
            share = ShareType.SHARE_A
        elif roll < share_a_ratio + share_b_ratio:
            share = ShareType.SHARE_B
        else:
            share = ShareType.NEUTRAL

        instr = Instruction(
            idx=i,
            name=op,
            dest_reg=dest,
            source_regs=tuple(srcs),
            latency=latency,
            share_type=share,
        )
        instructions.append(instr)
        written_regs.append((dest, i))

    return instructions


def describe_block(instructions: List[Instruction]) -> None:
    n = len(instructions)
    counts = {s: 0 for s in ShareType}
    for instr in instructions:
        counts[instr.share_type] += 1

    print(f"\n=== Instruction Block (n={n}) ===")
    for instr in instructions:
        print(f"  {instr}")
    print(
        f"\nShare distribution: "
        f"A={counts[ShareType.SHARE_A]} ({counts[ShareType.SHARE_A]/n:.0%}), "
        f"B={counts[ShareType.SHARE_B]} ({counts[ShareType.SHARE_B]/n:.0%}), "
        f"N={counts[ShareType.NEUTRAL]} ({counts[ShareType.NEUTRAL]/n:.0%})"
    )
    print()
