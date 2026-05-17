
import pytest
from arm_scheduler.core.instruction import (
    Instruction, ShareType, build_dependency_graph, validate_schedule
)
from arm_scheduler.core.pipeline import PipelineState
from arm_scheduler.core.generator import generate_block


# ---------------------------------------------------------------------------
# Instruction model
# ---------------------------------------------------------------------------

def make_instr(idx, name, dest, srcs, latency, share):
    return Instruction(idx=idx, name=name, dest_reg=dest,
                       source_regs=tuple(srcs), latency=latency, share_type=share)


def test_instruction_reads_reg():
    i = make_instr(0, "ADD", "r0", ["r1", "r2"], 1, ShareType.SHARE_A)
    assert i.reads_reg("r1")
    assert i.reads_reg("r2")
    assert not i.reads_reg("r0")


def test_instruction_writes_reg():
    i = make_instr(0, "ADD", "r0", ["r1", "r2"], 1, ShareType.NEUTRAL)
    assert i.writes_reg("r0")
    assert not i.writes_reg("r1")


# ---------------------------------------------------------------------------
# Dependency graph
# ---------------------------------------------------------------------------

def test_raw_dependency_detected():
    i1 = make_instr(0, "MOV", "r0", [], 1, ShareType.NEUTRAL)
    i2 = make_instr(1, "ADD", "r1", ["r0", "r2"], 1, ShareType.NEUTRAL)
    deps = build_dependency_graph([i1, i2])
    assert 0 in deps[1], "i2 should depend on i1"


def test_no_spurious_dependency():
    i1 = make_instr(0, "ADD", "r0", ["r1", "r2"], 1, ShareType.NEUTRAL)
    i2 = make_instr(1, "ADD", "r3", ["r4", "r5"], 1, ShareType.NEUTRAL)
    deps = build_dependency_graph([i1, i2])
    assert deps[1] == []


def test_only_latest_writer():
    i0 = make_instr(0, "MOV", "r0", [], 1, ShareType.NEUTRAL)
    i1 = make_instr(1, "MOV", "r0", [], 1, ShareType.NEUTRAL)  # overwrites r0
    i2 = make_instr(2, "ADD", "r3", ["r0"], 1, ShareType.NEUTRAL)
    deps = build_dependency_graph([i0, i1, i2])
    # i2 should depend on i1 (latest writer of r0), not i0
    assert 1 in deps[2]
    assert 0 not in deps[2]


# ---------------------------------------------------------------------------
# PipelineState
# ---------------------------------------------------------------------------

def test_critical_path_single():
    i0 = make_instr(0, "LDR", "r0", [], 2, ShareType.NEUTRAL)
    ps = PipelineState([i0], k=3)
    assert ps._critical_path[0] == 2


def test_critical_path_chain():
    i0 = make_instr(0, "MOV", "r0", [], 1, ShareType.NEUTRAL)
    i1 = make_instr(1, "ADD", "r1", ["r0"], 1, ShareType.NEUTRAL)
    i2 = make_instr(2, "ADD", "r2", ["r1"], 1, ShareType.NEUTRAL)
    ps = PipelineState([i0, i1, i2], k=3)
    assert ps._critical_path[0] == 3
    assert ps._critical_path[1] == 2
    assert ps._critical_path[2] == 1


def test_heuristic_is_safe():
    instrs = generate_block(n=10, seed=42)
    ps = PipelineState(instrs, k=3)
    h = ps.heuristic(frozenset(i.idx for i in instrs))
    assert h <= 20, f"Heuristic {h} suspiciously large"


# ---------------------------------------------------------------------------
# Security validity
# ---------------------------------------------------------------------------

def test_security_valid_pass():
    i0 = make_instr(0, "EOR", "r0", ["r1"], 1, ShareType.SHARE_A)
    i1 = make_instr(1, "EOR", "r2", ["r3"], 1, ShareType.SHARE_B)
    ps = PipelineState([i0, i1], k=3)
    placement = {0: 0}   # i0 placed at cycle 0
    # i1 at cycle 3: distance = 3 = k → valid
    assert ps.is_security_valid(i1, 3, placement)


def test_security_valid_fail():
    i0 = make_instr(0, "EOR", "r0", ["r1"], 1, ShareType.SHARE_A)
    i1 = make_instr(1, "EOR", "r2", ["r3"], 1, ShareType.SHARE_B)
    ps = PipelineState([i0, i1], k=3)
    placement = {0: 0}   # i0 placed at cycle 0
    # i1 at cycle 2: distance = 2 < 3 → invalid
    assert not ps.is_security_valid(i1, 2, placement)


def test_neutral_never_violates():
    i0 = make_instr(0, "MOV", "r0", [], 1, ShareType.SHARE_A)
    i1 = make_instr(1, "MOV", "r2", [], 1, ShareType.NEUTRAL)
    ps = PipelineState([i0, i1], k=3)
    placement = {0: 0}
    assert ps.is_security_valid(i1, 0, placement)   # same cycle, but NEUTRAL


# ---------------------------------------------------------------------------
# Static validation
# ---------------------------------------------------------------------------

def test_validate_correct_schedule():
    i0 = make_instr(0, "MOV", "r0", [], 1, ShareType.SHARE_A)
    i1 = make_instr(1, "MOV", "r2", [], 1, ShareType.SHARE_B)
    instrs = [i0, i1]
    preds = build_dependency_graph(instrs)
    # Valid schedule: i0 at cycle 0, i1 at cycle 3
    schedule = [(0, i0), (1, None), (2, None), (3, i1)]
    valid, errors = validate_schedule(schedule, instrs, preds, k=3)
    assert valid, errors


def test_validate_security_violation():
    i0 = make_instr(0, "MOV", "r0", [], 1, ShareType.SHARE_A)
    i1 = make_instr(1, "MOV", "r2", [], 1, ShareType.SHARE_B)
    instrs = [i0, i1]
    preds = build_dependency_graph(instrs)
    # Invalid: i0 at cycle 0, i1 at cycle 1 (distance 1 < k=3)
    schedule = [(0, i0), (1, i1)]
    valid, errors = validate_schedule(schedule, instrs, preds, k=3)
    assert not valid
    assert any("Security" in e for e in errors)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def test_generator_reproducible():
    b1 = generate_block(n=20, seed=99)
    b2 = generate_block(n=20, seed=99)
    assert [i.name for i in b1] == [i.name for i in b2]


def test_generator_indices():
    block = generate_block(n=15, seed=1)
    for i, instr in enumerate(block):
        assert instr.idx == i


def test_generator_share_distribution():
    block = generate_block(n=50, seed=42)
    types = {instr.share_type for instr in block}
    assert ShareType.SHARE_A in types
    assert ShareType.SHARE_B in types
    assert ShareType.NEUTRAL in types
