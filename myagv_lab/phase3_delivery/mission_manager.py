"""
myagv_lab/phase3_delivery/mission_manager.py
=============================================
Phase 3 — LLM-PDDL Mission Manager

Top-level node that wires all four pipeline modules:

  Natural Language
       │
       ▼  [llm_translator]
  PDDL domain + problem strings
       │
       ▼  [pddl_solver]
  list[PlanStep]
       │
       ▼  [primitive_executor]
  Physical robot motion (Nav2 + cobot)

Sim mode  : Full pipeline, simulated hardware.
Real mode : Full pipeline, real ROS2 hardware.
No-LLM    : Skip the API call, use a hard-coded fallback PDDL problem.

Usage
-----
  # Simulation with LLM translation
  python3 -m myagv_lab.phase3_delivery.mission_manager --sim \\
    --task "Deliver package_A to the delivery area and return home."

  # Simulation without API key (fallback PDDL)
  python3 -m myagv_lab.phase3_delivery.mission_manager --sim --fallback

  # Simulation with live two-panel visualisation
  python3 -m myagv_lab.phase3_delivery.mission_manager --sim --fallback --viz

  # Real robot
  python3 -m myagv_lab.phase3_delivery.mission_manager --real \\
    --task "Pick up package_A and deliver it."
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from myagv_lab.sim_layer import USE_SIM
from myagv_lab.phase3_delivery.pddl_planner.llm_translator  import (
    natural_language_to_pddl, fallback_pddl,
)
from myagv_lab.phase3_delivery.pddl_planner.pddl_solver     import (
    solve_pddl, print_plan,
)
from myagv_lab.phase3_delivery.pddl_planner.primitive_executor import (
    PrimitiveExecutor, StepResult,
)

log = logging.getLogger("mission_manager")


# ═══════════════════════════════════════════════════════════════════════════════
#  STATUS TRACKER  (collects all status events for a post-run summary)
# ═══════════════════════════════════════════════════════════════════════════════

class StatusTracker:
    def __init__(self):
        self.events: list[tuple[float, str]] = []
        self._start = time.monotonic()

    def record(self, status: str) -> None:
        elapsed = time.monotonic() - self._start
        self.events.append((elapsed, status))
        log.info(f"[STATUS]  {status}  (t={elapsed:.1f}s)")

    def print_summary(self) -> None:
        print()
        print("─" * 60)
        print("  Mission Timeline")
        print("─" * 60)
        for t, s in self.events:
            print(f"  t={t:6.1f}s  →  {s}")
        print("─" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
#  MISSION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class MissionManager:
    """
    Orchestrates the full NL → PDDL → Plan → Execution pipeline.

    Parameters
    ----------
    use_llm     : bool — call the Claude API; False uses hard-coded PDDL
    fallback_scenario : str — which hard-coded scenario to use if use_llm=False
    use_astar   : bool — use A* search in pyperplan (vs. BFS)
    """

    def __init__(
        self,
        use_llm:           bool = True,
        fallback_scenario: str  = "deliver_A",
        use_astar:         bool = False,
    ):
        self._use_llm  = use_llm
        self._fallback = fallback_scenario
        self._astar    = use_astar
        self._tracker  = StatusTracker()

        # Executor is built here so it initialises ROS2 / sim early
        self._executor = PrimitiveExecutor(
            on_status=self._tracker.record,
            on_step=lambda step: log.info(
                f"\n{'='*50}\n  Executing: {step.raw}\n{'='*50}"
            ),
        )

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(self, task_description: str) -> bool:
        """
        Run the full pipeline for the given task description.

        Returns True if all steps succeeded, False otherwise.
        """
        self._banner()
        self._tracker.record("PIPELINE_START")

        # ── Stage 1: NL → PDDL ───────────────────────────────────────────────
        log.info("")
        log.info("━" * 50)
        log.info("  Stage 1/3 — Natural Language → PDDL")
        log.info("━" * 50)
        log.info(f"  Task: {task_description!r}")
        log.info("")

        self._tracker.record("LLM_TRANSLATING")

        try:
            if self._use_llm:
                domain_pddl, problem_pddl = natural_language_to_pddl(task_description)
            else:
                domain_pddl, problem_pddl = fallback_pddl(self._fallback)
        except (EnvironmentError, RuntimeError) as e:
            log.error(f"[Stage 1] FAILED: {e}")
            self._tracker.record("LLM_FAILED")
            return False

        self._tracker.record("PDDL_GENERATED")
        log.info("[Stage 1] PDDL generated successfully.")

        # ── Stage 2: PDDL → Plan ─────────────────────────────────────────────
        log.info("")
        log.info("━" * 50)
        log.info("  Stage 2/3 — PDDL → Plan (pyperplan)")
        log.info("━" * 50)

        self._tracker.record("PLANNING")

        try:
            plan = solve_pddl(domain_pddl, problem_pddl, use_astar=self._astar)
        except (ValueError, RuntimeError) as e:
            log.error(f"[Stage 2] Planning FAILED: {e}")
            self._tracker.record("PLANNING_FAILED")
            return False

        print_plan(plan, title=f"Plan for: {task_description[:50]}")
        self._tracker.record(f"PLAN_READY({len(plan)}_steps)")

        # ── Stage 3: Execution ────────────────────────────────────────────────
        log.info("")
        log.info("━" * 50)
        log.info("  Stage 3/3 — Executing Plan")
        log.info("━" * 50)

        results = self._executor.execute_plan(plan)

        # ── Summary ───────────────────────────────────────────────────────────
        success = all(r.success for r in results)
        failed  = [r for r in results if not r.success]

        self._tracker.print_summary()

        print()
        if success:
            print("  ✓  Mission COMPLETE — all steps succeeded.")
        else:
            print(f"  ✗  Mission FAILED — {len(failed)} step(s) failed:")
            for r in failed:
                print(f"       {r}")
        print()

        return success

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _banner(self) -> None:
        mode = "SIM" if USE_SIM else "REAL"
        llm  = "LLM → PDDL" if self._use_llm else f"Fallback ({self._fallback})"
        planner = "A*" if self._astar else "BFS"
        print()
        print("┌" + "═" * 58 + "┐")
        print("│  myAGV Lab — Phase 3: LLM-PDDL Mission Manager" + " " * 10 + "│")
        print("│" + "─" * 58 + "│")
        print(f"│  Mode    : {mode:<47}│")
        print(f"│  Planner : {planner:<47}│")
        print(f"│  NL→PDDL : {llm:<47}│")
        print("└" + "═" * 58 + "┘")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Phase 3 — LLM-PDDL Mission Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sim with LLM
  python3 -m myagv_lab.phase3_delivery.mission_manager --sim \\
    --task "Deliver package_A to delivery_area and return home."

  # Sim without API key
  python3 -m myagv_lab.phase3_delivery.mission_manager --sim --fallback

  # Sim with live visualisation
  python3 -m myagv_lab.phase3_delivery.mission_manager --sim --fallback --viz

  # Sim with two packages + visualisation
  python3 -m myagv_lab.phase3_delivery.mission_manager --sim \\
    --fallback --scenario deliver_AB --viz

  # Sim with recharge + delivery + visualisation
  python3 -m myagv_lab.phase3_delivery.mission_manager --sim \\
    --fallback --scenario recharge_then_deliver --viz

  # Real robot (no viz)
  python3 -m myagv_lab.phase3_delivery.mission_manager --real \\
    --task "Deliver package_A to delivery_area."
""",
    )

    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument("--sim",  action="store_true", default=True,
                          help="Simulation mode (default)")
    mode_grp.add_argument("--real", action="store_true",
                          help="Real robot mode (ROS2 must be sourced)")

    parser.add_argument("--task", type=str, default=None,
                        help="Natural language task description")
    parser.add_argument("--fallback", action="store_true",
                        help="Skip LLM; use a hard-coded PDDL problem")
    parser.add_argument("--scenario", default="deliver_A",
                        choices=["deliver_A", "deliver_AB", "recharge_then_deliver"],
                        help="Which fallback scenario to use")
    parser.add_argument("--astar", action="store_true",
                        help="Use A* instead of BFS in pyperplan")
    parser.add_argument("--viz", action="store_true",
                        help="Show live two-panel matplotlib visualisation (sim only)")

    args = parser.parse_args()

    # Determine task description
    if not args.fallback:
        if args.task is None:
            args.task = input(
                "Enter task description:\n> "
            ).strip()
        if not args.task:
            parser.error("Provide --task or --fallback")

    use_sim = not args.real

    # Override env var so sim_layer picks up the right mode
    import os
    os.environ["MYAGV_USE_SIM"] = "1" if use_sim else "0"

    task_desc = args.task or f"[fallback scenario: {args.scenario}]"

    if args.viz and use_sim:
        from myagv_lab.phase3_delivery.mission_viz import MissionVisualizer
        viz     = MissionVisualizer()
        success = viz.run(
            task_description=task_desc,
            use_llm=not args.fallback,
            fallback_scenario=args.scenario,
            use_astar=args.astar,
        )
        print("Close the window to exit.")
        viz.show()
    else:
        if args.viz and not use_sim:
            log.warning("--viz is only supported in sim mode; running without visualisation.")
        manager = MissionManager(
            use_llm=not args.fallback,
            fallback_scenario=args.scenario,
            use_astar=args.astar,
        )
        success = manager.run(task_desc)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
