"""
myagv_lab/phase3_delivery/pddl_planner/primitive_executor.py
============================================================
Module 3 of 4 in the Phase 3 pipeline.

The grounding layer: maps each abstract PDDL action string
to the correct robot primitive call.

Sim mode  : Calls SimRobot.navigate_to(), SimCobot.load()/unload()
Real mode : Calls NavigationManager (Nav2) + ROS2 cobot topics

Dispatch table
--------------
  PDDL action         →  primitive
  ─────────────────────────────────────────────────────────────
  navigate            →  nav_manager.navigate(to_loc)
  load-package        →  cobot.load(package, agv)
  deliver-package     →  cobot.unload(package, agv)
  recharge            →  sim: sleep; real: ros2 service call

Pipeline position
-----------------
  list[PlanStep]  ──►  [THIS MODULE]  ──►  physical robot motion
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from myagv_lab.sim_layer import (
    USE_SIM, SimRobot, SimCobot, Pose2D, NavResult,
    get_robot, get_cobot, get_map,
)
from myagv_lab.phase2_nav.nav_node import NavigationManager, WAYPOINTS
from myagv_lab.phase3_delivery.pddl_planner.pddl_solver import PlanStep

log = logging.getLogger("primitive_executor")

RECHARGE_DURATION = 4.0   # seconds (sim)


# ═══════════════════════════════════════════════════════════════════════════════
#  EXECUTION RESULT
# ═══════════════════════════════════════════════════════════════════════════════

class StepResult:
    def __init__(self, step: PlanStep, success: bool, message: str = ""):
        self.step    = step
        self.success = success
        self.message = message

    def __str__(self) -> str:
        tag = "✓" if self.success else "✗"
        return f"{tag} {self.step}  [{self.message}]"


# ═══════════════════════════════════════════════════════════════════════════════
#  PRIMITIVE EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

class PrimitiveExecutor:
    """
    Executes a pyperplan plan by dispatching each PlanStep to the
    matching robot primitive.

    Parameters
    ----------
    robot       : SimRobot instance (sim) or None (real)
    cobot       : SimCobot instance (sim) or None (real)
    on_status   : callback(str) called on every status change
    on_step     : callback(PlanStep) called before each step executes
    """

    def __init__(
        self,
        robot:     Optional[SimRobot] = None,
        cobot:     Optional[SimCobot] = None,
        on_status: Callable[[str], None] = None,
        on_step:   Callable[[PlanStep], None] = None,
    ):
        self._on_status = on_status or (lambda s: None)
        self._on_step   = on_step   or (lambda s: None)

        # ── Build sub-systems ──────────────────────────────────────────────────
        if USE_SIM:
            self._robot = robot or get_robot(WAYPOINTS.get("home", Pose2D()))
            self._cobot = cobot or get_cobot(on_status=self._on_status)
            self._nav   = NavigationManager(robot=self._robot,
                                            on_status=self._on_status)
        else:
            # Real mode: NavigationManager handles ROS2 internally
            self._robot = None
            self._cobot = None        # real cobot driven via ROS2 topics
            self._nav   = NavigationManager(on_status=self._on_status)
            self._init_real_cobot()

    # ── Real-robot cobot setup (ROS2 publisher) ───────────────────────────────

    def _init_real_cobot(self) -> None:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init()
        self._cobot_node = Node("cobot_executor")
        self._cobot_pub  = self._cobot_node.create_publisher(
            String, "/cobot_command", 10
        )
        self._cobot_status: str = "IDLE"
        self._cobot_node.create_subscription(
            String, "/cobot_status",
            lambda msg: setattr(self, "_cobot_status", msg.data),
            10,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def execute_plan(self, plan: list[PlanStep]) -> list[StepResult]:
        """
        Execute each step in the plan sequentially.

        Returns a list of StepResult objects (one per step).
        Stops early if any step fails.
        """
        results: list[StepResult] = []
        self._on_status("EXECUTING_PLAN")
        log.info(f"[Executor] Starting plan with {len(plan)} step(s)")

        for step in plan:
            self._on_step(step)
            log.info(f"[Executor] ── {step} ──")
            try:
                result = self._dispatch(step)
            except Exception as e:
                result = StepResult(step, False, str(e))
                log.error(f"[Executor] Exception in step {step.index}: {e}")

            results.append(result)
            log.info(f"[Executor] {result}")

            if not result.success:
                log.error("[Executor] Aborting plan due to step failure.")
                self._on_status("PLAN_ABORTED")
                return results

        self._on_status("MISSION_COMPLETE")
        log.info("[Executor] ✓  Mission complete.")
        return results

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def _dispatch(self, step: PlanStep) -> StepResult:
        """Route one PlanStep to the correct primitive method."""
        name = step.name
        args = step.args

        if name == "navigate":
            # args: robot, from_loc, to_loc
            if len(args) < 3:
                return StepResult(step, False, "navigate needs 3 args")
            _robot, from_loc, to_loc = args[0], args[1], args[2]
            return self._exec_navigate(step, from_loc, to_loc)

        elif name == "load-package":
            # args: robot, arm, package, location
            if len(args) < 4:
                return StepResult(step, False, "load-package needs 4 args")
            _robot, _arm, package, location = args
            return self._exec_load(step, package, location)

        elif name == "deliver-package":
            # args: robot, package, location
            if len(args) < 3:
                return StepResult(step, False, "deliver-package needs 3 args")
            _robot, package, location = args
            return self._exec_deliver(step, package, location)

        elif name == "recharge":
            # args: robot, location
            if len(args) < 2:
                return StepResult(step, False, "recharge needs 2 args")
            _robot, location = args
            return self._exec_recharge(step, location)

        else:
            msg = f"Unknown action {name!r} — no primitive registered"
            log.warning(f"[Executor] {msg}")
            return StepResult(step, False, msg)

    # ── Primitives ────────────────────────────────────────────────────────────

    def _exec_navigate(self, step: PlanStep,
                       from_loc: str, to_loc: str) -> StepResult:
        self._on_status(f"NAVIGATING:{to_loc}")
        nav_result: NavResult = self._nav.navigate(to_loc)
        if nav_result.success:
            return StepResult(step, True,
                              f"Arrived at {to_loc}")
        return StepResult(step, False,
                          f"Navigation failed: {nav_result.message}")

    def _exec_load(self, step: PlanStep,
                   package: str, location: str) -> StepResult:
        self._on_status(f"LOADING:{package}")

        if USE_SIM:
            ok = self._cobot.load(package, self._robot)
            return StepResult(step, ok,
                              "Loaded" if ok else "Load failed")
        else:
            return self._real_cobot_cmd("LOAD", "LOAD_COMPLETE", timeout=30.0,
                                        step=step, tag=package)

    def _exec_deliver(self, step: PlanStep,
                      package: str, location: str) -> StepResult:
        self._on_status(f"DELIVERING:{package}@{location}")

        if USE_SIM:
            ok = self._cobot.unload(package, self._robot)
            return StepResult(step, ok,
                              "Delivered" if ok else "Unload failed")
        else:
            return self._real_cobot_cmd("UNLOAD", "UNLOAD_COMPLETE", timeout=20.0,
                                        step=step, tag=package)

    def _exec_recharge(self, step: PlanStep, location: str) -> StepResult:
        self._on_status(f"RECHARGING@{location}")
        log.info(f"[Executor] Recharging at {location} …")

        if USE_SIM:
            # Simulate charging time
            for i in range(int(RECHARGE_DURATION / 0.5)):
                time.sleep(0.5)
                pct = int((i + 1) / (RECHARGE_DURATION / 0.5) * 100)
                log.info(f"[Executor]   Charging … {pct}%")
        else:
            # Real: call a ROS2 service or just wait
            time.sleep(RECHARGE_DURATION)

        self._on_status("CHARGED")
        return StepResult(step, True, f"Recharged at {location}")

    # ── Real cobot helper ─────────────────────────────────────────────────────

    def _real_cobot_cmd(self, command: str, expected_status: str,
                        timeout: float, step: PlanStep, tag: str) -> StepResult:
        """Publish a cobot command and spin until the expected status arrives."""
        import rclpy
        from std_msgs.msg import String

        self._cobot_status = "IDLE"
        msg = String(); msg.data = command
        self._cobot_pub.publish(msg)
        log.info(f"[Executor] Sent /cobot_command: {command}")

        deadline = time.monotonic() + timeout
        while self._cobot_status != expected_status:
            if time.monotonic() > deadline:
                return StepResult(step, False,
                                  f"Cobot timeout waiting for {expected_status}")
            rclpy.spin_once(self._cobot_node, timeout_sec=0.5)

        return StepResult(step, True, f"{tag}: {expected_status}")
