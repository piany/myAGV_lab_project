"""
myagv_lab/phase3_delivery/pddl_planner/pddl_solver.py
======================================================
Module 2 of 4 in the Phase 3 pipeline.

Parses PDDL strings with pyperplan, grounds all action schemas,
and searches for a plan using BFS (default) or A*.

Pipeline position
-----------------
  (domain_str, problem_str)  ──►  [THIS MODULE]  ──►  list[str]
                                                        grounded actions
"""

from __future__ import annotations

import os
import tempfile
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("pddl_solver")


# ═══════════════════════════════════════════════════════════════════════════════
#  PLAN STEP  (structured representation of one grounded action)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlanStep:
    """One step in the plan returned by pyperplan."""
    index:    int          # 1-based
    name:     str          # action name, e.g. "navigate"
    args:     list[str]    # grounded arguments, e.g. ["agv1", "home", "loading_area"]
    raw:      str          # original string from pyperplan, e.g. "(navigate agv1 home loading_area)"

    @classmethod
    def from_raw(cls, index: int, raw: str) -> "PlanStep":
        inner  = raw.strip().strip("()")
        tokens = inner.split()
        name   = tokens[0]
        args   = tokens[1:]
        return cls(index=index, name=name, args=args, raw=raw)

    def __str__(self) -> str:
        return f"Step {self.index:2d}: {self.raw}"


# ═══════════════════════════════════════════════════════════════════════════════
#  SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def solve_pddl(
    domain_pddl:  str,
    problem_pddl: str,
    use_astar:    bool = False,
) -> list[PlanStep]:
    """
    Parse domain + problem PDDL strings and return a plan.

    Args
    ----
    domain_pddl  : PDDL domain as a string.
    problem_pddl : PDDL problem as a string.
    use_astar    : If True, use A* with lm-cut heuristic (faster for
                   larger problems); if False, use breadth-first search.

    Returns
    -------
    list[PlanStep]
        Ordered list of grounded plan steps.

    Raises
    ------
    RuntimeError
        If pyperplan finds no plan (unsolvable problem).
    ValueError
        If the PDDL strings are syntactically invalid.
    """
    from pyperplan.pddl.parser import Parser
    from pyperplan import grounding
    from pyperplan.search import breadth_first_search, astar_search
    from pyperplan.heuristics.blind import BlindHeuristic

    # pyperplan requires files on disk
    domain_file  = None
    problem_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pddl", delete=False, prefix="domain_"
        ) as df:
            df.write(domain_pddl)
            domain_file = df.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pddl", delete=False, prefix="problem_"
        ) as pf:
            pf.write(problem_pddl)
            problem_file = pf.name

        # ── Parse ──────────────────────────────────────────────────────────────
        log.info("[Solver] Parsing PDDL …")
        try:
            parser  = Parser(domain_file, problem_file)
            domain  = parser.parse_domain()
            problem = parser.parse_problem(domain)
        except Exception as e:
            raise ValueError(f"PDDL parse error: {e}") from e

        # ── Ground ─────────────────────────────────────────────────────────────
        log.info("[Solver] Grounding action schemas …")
        task = grounding.ground(problem)

        log.info(f"[Solver] Grounded operators: {len(task.operators)}")
        log.info(f"[Solver] Initial facts:       {len(task.initial_state)}")

        # ── Search ─────────────────────────────────────────────────────────────
        if use_astar:
            log.info("[Solver] Searching with A* (blind heuristic) …")
            heuristic = BlindHeuristic(task)
            solution  = astar_search(task, heuristic)
        else:
            log.info("[Solver] Searching with BFS …")
            solution  = breadth_first_search(task)

        if solution is None:
            raise RuntimeError(
                "pyperplan found no plan.\n"
                "Check that your PDDL goal is reachable from the initial state.\n"
                "Hint: validate your problem.pddl at editor.planning.domains"
            )

        # ── Build plan ─────────────────────────────────────────────────────────
        raw_names = [str(op.name) for op in solution]
        plan      = [PlanStep.from_raw(i + 1, r) for i, r in enumerate(raw_names)]

        log.info(f"[Solver] Plan found — {len(plan)} action(s):")
        for step in plan:
            log.info(f"  {step}")

        return plan

    finally:
        for path in (domain_file, problem_file):
            if path and os.path.exists(path):
                os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════════
#  PLAN DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def print_plan(plan: list[PlanStep], title: str = "Generated Plan") -> None:
    """Pretty-print a plan to stdout."""
    width = 60
    print()
    print("┌" + "─" * width + "┐")
    print(f"│  {title:<{width - 3}}│")
    print("├" + "─" * width + "┤")
    for step in plan:
        action_str = f"  {step.index:2d}. [{step.name}]  {' '.join(step.args)}"
        print(f"│{action_str:<{width}}│")
    print("└" + "─" * width + "┘")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, argparse
    from pathlib import Path
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from myagv_lab.phase3_delivery.pddl_planner.llm_translator import fallback_pddl

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="deliver_A",
                        choices=["deliver_A", "deliver_AB", "recharge_then_deliver"])
    parser.add_argument("--astar", action="store_true")
    args = parser.parse_args()

    domain, problem = fallback_pddl(args.scenario)
    plan = solve_pddl(domain, problem, use_astar=args.astar)
    print_plan(plan, f"Plan for scenario: {args.scenario}")
