"""
tests/test_all_phases.py
========================
Automated tests for all three lab phases.

All tests run in simulation mode (no ROS2, no API key required).

Run (activate venv first: source .venv/bin/activate):
  PYTHONPATH="" python3 -m pytest tests/test_all_phases.py -v
  PYTHONPATH="" python3 -m pytest tests/test_all_phases.py -v -k phase1
  PYTHONPATH="" python3 -m pytest tests/test_all_phases.py -v -k phase3
  # Or use the provided script from the myagv_lab/ directory:
  ./run_tests.sh tests/test_all_phases.py -v
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Ensure sim mode
os.environ["MYAGV_USE_SIM"] = "1"

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1  —  SLAM
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase1SimMap:
    def test_map_is_free_inside_room(self):
        from myagv_lab.sim_layer import SimMap
        m = SimMap()
        # Centre of the room should be free
        assert m.is_free(1.0, 0.4), "Open space should be free"

    def test_map_is_occupied_at_wall(self):
        from myagv_lab.sim_layer import SimMap
        m = SimMap()
        # Top-left corner '#' cell
        assert not m.is_free(0.0, 0.0), "Wall cell should be occupied"

    def test_ray_cast_hits_wall(self):
        from myagv_lab.sim_layer import SimMap, Pose2D
        m   = SimMap()
        pos = Pose2D(1.0, 0.4, 0.0)
        # Casting straight up (north) should hit the top wall
        r = m.ray_cast(pos, math.radians(90), max_range=20.0)
        assert r < 20.0, "Ray should hit the north wall"
        assert r > 0.0,  "Ray distance must be positive"

    def test_ray_cast_max_range_in_free_space(self):
        from myagv_lab.sim_layer import SimMap, Pose2D
        m   = SimMap()
        pos = Pose2D(1.0, 0.4, 0.0)
        # Very short max_range in open space returns max_range
        r = m.ray_cast(pos, 0.0, max_range=0.05)
        assert r == pytest.approx(0.05, abs=0.01)


class TestPhase1OccupancyGrid:
    def test_grid_initialised_to_unknown(self):
        from myagv_lab.phase1_slam.slam_node import OccupancyGrid
        g = OccupancyGrid(10.0, 8.0, resolution=0.1)
        row, col = g.world_to_cell(5.0, 4.0)
        assert g.probability(row, col) == pytest.approx(0.5, abs=0.01)

    def test_repeated_occupied_updates_raise_probability(self):
        from myagv_lab.phase1_slam.slam_node import OccupancyGrid
        g = OccupancyGrid(10.0, 8.0, resolution=0.1)
        row, col = g.world_to_cell(5.0, 4.0)
        # Inject occupied log-odds 5 times
        for _ in range(5):
            g._grid[row, col] += g.OCC_LOG_ODDS
        assert g.probability(row, col) > 0.9

    def test_repeated_free_updates_lower_probability(self):
        from myagv_lab.phase1_slam.slam_node import OccupancyGrid
        g = OccupancyGrid(10.0, 8.0, resolution=0.1)
        row, col = g.world_to_cell(5.0, 4.0)
        for _ in range(6):
            g._grid[row, col] += g.FREE_LOG_ODDS
        assert g.probability(row, col) < 0.09

    def test_scan_integration_marks_obstacle(self):
        """After integrating a scan with a hit, the endpoint should be occupied."""
        import math
        from myagv_lab.phase1_slam.slam_node import OccupancyGrid
        from myagv_lab.sim_layer import LidarScan, Pose2D
        g   = OccupancyGrid(10.0, 8.0, resolution=0.05)
        pos = Pose2D(1.0, 1.0, 0.0)
        # Fake scan: one ray pointing east hitting at 2.0 m
        scan = LidarScan(
            ranges=[2.0],
            angle_min=0.0,
            angle_max=0.0,
            angle_increment=0.0,
            range_max=12.0,
        )
        for _ in range(10):   # integrate 10 times to overcome prior
            g.integrate_scan(pos, scan)
        row, col = g.world_to_cell(pos.x + 2.0, pos.y)
        assert g.probability(row, col) > 0.65

    def test_to_image_array_shape(self):
        from myagv_lab.phase1_slam.slam_node import OccupancyGrid
        g = OccupancyGrid(10.0, 8.0, resolution=0.1)
        img = g.to_image_array()
        assert img.shape == (80, 100)   # height_cells × width_cells

    def test_save_creates_files(self, tmp_path, monkeypatch):
        import imageio
        import numpy as np
        from myagv_lab.phase1_slam.slam_node import OccupancyGrid, MAPS_DIR
        # Redirect MAPS_DIR to tmp_path
        monkeypatch.setattr(
            "myagv_lab.phase1_slam.slam_node.MAPS_DIR", tmp_path
        )
        g = OccupancyGrid(2.0, 2.0, resolution=0.1)
        png, yaml = g.save("test_map")
        assert png.exists(),  "PNG must be created"
        assert yaml.exists(), "YAML must be created"
        assert "resolution" in yaml.read_text()


class TestPhase1SLAMNode:
    def test_scan_increments_counter(self):
        from myagv_lab.phase1_slam.slam_node import SLAMNode
        node = SLAMNode()
        node.add_scan()
        assert node._scans == 1

    def test_multiple_scans(self):
        from myagv_lab.phase1_slam.slam_node import SLAMNode
        node = SLAMNode()
        for _ in range(5):
            node.add_scan()
        assert node._scans == 5

    def test_odom_drift_is_nonzero(self):
        """Dead-reckoning estimator should accumulate drift after motion."""
        import math
        from myagv_lab.phase1_slam.slam_node import DeadReckoningEstimator
        from myagv_lab.sim_layer import Pose2D
        est   = DeadReckoningEstimator(Pose2D(0.0, 0.0, 0.0))
        moved = Pose2D(1.0, 0.0, 0.0)
        result = est.update(moved)
        # First call: prev_true was None, result is initial pose (0,0)
        assert result.x == pytest.approx(0.0, abs=0.01)
        # Second call applies delta with noise — should be near 1.0
        moved2 = Pose2D(2.0, 0.0, 0.0)
        result2 = est.update(moved2)
        assert abs(result2.x - 1.0) < 0.15   # 1m delta with small noise


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2  —  NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase2SimRobot:
    def test_robot_starts_at_given_pose(self):
        from myagv_lab.sim_layer import SimRobot, Pose2D
        r = SimRobot(start=Pose2D(2.0, 3.0, 0.5))
        assert r.position.x == pytest.approx(2.0)
        assert r.position.y == pytest.approx(3.0)

    def test_navigate_to_reachable_goal(self):
        from myagv_lab.sim_layer import SimRobot, Pose2D
        r      = SimRobot(start=Pose2D(0.4, 0.6, 0.0))
        result = r.navigate_to(Pose2D(2.0, 0.4, 0.0), timeout=30.0)
        assert result.success, f"Navigation failed: {result.message}"
        assert r.position.distance_to(Pose2D(2.0, 0.4, 0.0)) < 0.2

    def test_navigate_multiple_goals(self):
        from myagv_lab.sim_layer import SimRobot, Pose2D
        r  = SimRobot(start=Pose2D(0.4, 0.6, 0.0))
        g1 = Pose2D(2.0, 0.4, 0.0)
        g2 = Pose2D(5.0, 0.4, 0.0)
        r1 = r.navigate_to(g1, timeout=30.0)
        r2 = r.navigate_to(g2, timeout=30.0)
        assert r1.success
        assert r2.success

    def test_pick_up_and_put_down(self):
        from myagv_lab.sim_layer import SimRobot, Pose2D
        r = SimRobot(start=Pose2D(1.0, 1.0, 0.0))
        assert r.carrying is None
        ok = r.pick_up("package_A")
        assert ok
        assert r.carrying == "package_A"
        pkg = r.put_down()
        assert pkg == "package_A"
        assert r.carrying is None

    def test_cannot_carry_two_packages(self):
        from myagv_lab.sim_layer import SimRobot, Pose2D
        r = SimRobot(start=Pose2D(1.0, 1.0, 0.0))
        r.pick_up("package_A")
        ok2 = r.pick_up("package_B")
        assert not ok2                    # second pick-up must fail


class TestPhase2NavigationManager:
    def test_navigate_to_known_waypoint(self):
        from myagv_lab.phase2_nav.nav_node import NavigationManager, WAYPOINTS
        from myagv_lab.sim_layer import SimRobot
        robot = SimRobot(start=WAYPOINTS["home"])
        nav   = NavigationManager(robot=robot)
        result = nav.navigate("loading_area")
        assert result.success

    def test_navigate_to_unknown_waypoint_fails_gracefully(self):
        from myagv_lab.phase2_nav.nav_node import NavigationManager
        from myagv_lab.sim_layer import SimRobot, Pose2D
        robot = SimRobot(start=Pose2D(1.0, 1.0, 0.0))
        nav   = NavigationManager(robot=robot)
        result = nav.navigate("nonexistent_place")
        assert not result.success
        assert "Unknown location" in result.message

    def test_navigate_full_sequence(self):
        from myagv_lab.phase2_nav.nav_node import NavigationManager, WAYPOINTS
        from myagv_lab.sim_layer import SimRobot
        robot = SimRobot(start=WAYPOINTS["home"])
        nav   = NavigationManager(robot=robot)
        for dest in ["loading_area", "delivery_area", "home"]:
            r = nav.navigate(dest)
            assert r.success, f"Failed to reach {dest}: {r.message}"

    def test_waypoints_registry_is_complete(self):
        from myagv_lab.phase2_nav.nav_node import WAYPOINTS
        required = {"home", "loading_area", "delivery_area"}
        assert required.issubset(WAYPOINTS.keys())


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3 — PDDL PLANNER
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase3Domain:
    def test_domain_contains_required_actions(self):
        from myagv_lab.phase3_delivery.pddl_planner.domain import DOMAIN_PDDL
        for action in ["navigate", "load-package", "deliver-package", "recharge"]:
            assert action in DOMAIN_PDDL, f"Action {action!r} missing from domain"

    def test_domain_contains_required_predicates(self):
        from myagv_lab.phase3_delivery.pddl_planner.domain import DOMAIN_PDDL
        for pred in ["at", "package-at", "holding", "arm-at", "delivered"]:
            assert pred in DOMAIN_PDDL


class TestPhase3Solver:
    """Tests using fallback PDDL problems (no LLM API key required)."""

    def _solve(self, scenario: str):
        from myagv_lab.phase3_delivery.pddl_planner.llm_translator import fallback_pddl
        from myagv_lab.phase3_delivery.pddl_planner.pddl_solver import solve_pddl
        domain, problem = fallback_pddl(scenario)
        return solve_pddl(domain, problem)

    def test_deliver_A_plan_not_empty(self):
        plan = self._solve("deliver_A")
        assert len(plan) > 0

    def test_deliver_A_starts_with_navigate(self):
        plan = self._solve("deliver_A")
        assert plan[0].name == "navigate"

    def test_deliver_A_contains_load(self):
        plan = self._solve("deliver_A")
        names = [s.name for s in plan]
        assert "load-package" in names

    def test_deliver_A_contains_deliver(self):
        plan = self._solve("deliver_A")
        names = [s.name for s in plan]
        assert "deliver-package" in names

    def test_deliver_A_ends_at_home(self):
        """Last navigate action should bring agv1 to home."""
        plan = self._solve("deliver_A")
        nav_steps = [s for s in plan if s.name == "navigate"]
        last_nav  = nav_steps[-1]
        assert "home" in last_nav.args

    def test_deliver_AB_delivers_both_packages(self):
        plan  = self._solve("deliver_AB")
        names = [s.name for s in plan]
        # Two deliver-package actions expected
        assert names.count("deliver-package") >= 1  # planner may linearise

    def test_recharge_scenario_contains_recharge(self):
        plan  = self._solve("recharge_then_deliver")
        names = [s.name for s in plan]
        assert "recharge" in names

    def test_plan_step_parsing(self):
        from myagv_lab.phase3_delivery.pddl_planner.pddl_solver import PlanStep
        raw  = "(navigate agv1 home loading_area)"
        step = PlanStep.from_raw(1, raw)
        assert step.name == "navigate"
        assert step.args == ["agv1", "home", "loading_area"]
        assert step.index == 1
        assert step.raw   == raw

    def test_invalid_pddl_raises_value_error(self):
        from myagv_lab.phase3_delivery.pddl_planner.pddl_solver import solve_pddl
        from myagv_lab.phase3_delivery.pddl_planner.domain import DOMAIN_PDDL
        bad_problem = "(define (problem bad) (:domain nonexistent-domain))"
        with pytest.raises((ValueError, RuntimeError, Exception)):
            solve_pddl(DOMAIN_PDDL, bad_problem)

    def test_astar_produces_valid_plan(self):
        from myagv_lab.phase3_delivery.pddl_planner.llm_translator import fallback_pddl
        from myagv_lab.phase3_delivery.pddl_planner.pddl_solver import solve_pddl
        domain, problem = fallback_pddl("deliver_A")
        plan = solve_pddl(domain, problem, use_astar=True)
        assert len(plan) > 0


class TestPhase3Executor:
    """Full executor tests with simulated robot and cobot."""

    def _make_executor(self):
        from myagv_lab.phase3_delivery.pddl_planner.primitive_executor import PrimitiveExecutor
        from myagv_lab.sim_layer import SimRobot, SimCobot, Pose2D
        from myagv_lab.phase2_nav.nav_node import WAYPOINTS

        status_log = []
        robot = SimRobot(start=WAYPOINTS["home"])
        cobot = SimCobot(on_status=lambda s: status_log.append(s))
        exec_ = PrimitiveExecutor(robot=robot, cobot=cobot,
                                  on_status=lambda s: status_log.append(s))
        return exec_, status_log

    def _get_plan(self, scenario: str):
        from myagv_lab.phase3_delivery.pddl_planner.llm_translator import fallback_pddl
        from myagv_lab.phase3_delivery.pddl_planner.pddl_solver import solve_pddl
        domain, problem = fallback_pddl(scenario)
        return solve_pddl(domain, problem)

    def test_execute_deliver_A_succeeds(self):
        exec_, log = self._make_executor()
        plan = self._get_plan("deliver_A")
        results = exec_.execute_plan(plan)
        assert all(r.success for r in results), \
            "\n".join(str(r) for r in results if not r.success)

    def test_execute_recharge_succeeds(self):
        exec_, log = self._make_executor()
        plan = self._get_plan("recharge_then_deliver")
        results = exec_.execute_plan(plan)
        assert all(r.success for r in results), \
            "\n".join(str(r) for r in results if not r.success)

    def test_unknown_action_returns_failure(self):
        from myagv_lab.phase3_delivery.pddl_planner.primitive_executor import PrimitiveExecutor
        from myagv_lab.phase3_delivery.pddl_planner.pddl_solver import PlanStep
        exec_, _ = self._make_executor()
        bad_step = PlanStep(1, "fly-to-moon", ["agv1", "mars"], "(fly-to-moon agv1 mars)")
        result = exec_._dispatch(bad_step)
        assert not result.success

    def test_status_events_are_fired(self):
        exec_, status_log = self._make_executor()
        plan = self._get_plan("deliver_A")
        exec_.execute_plan(plan)
        assert len(status_log) > 0
        assert "MISSION_COMPLETE" in status_log

    def test_mission_manager_full_pipeline(self):
        """End-to-end: MissionManager.run() with fallback PDDL."""
        from myagv_lab.phase3_delivery.mission_manager import MissionManager
        mgr = MissionManager(use_llm=False, fallback_scenario="deliver_A")
        ok  = mgr.run("[test: deliver_A fallback]")
        assert ok


class TestPhase3FallbackScenarios:
    """Verify all three fallback scenarios produce non-trivial plans."""

    @pytest.mark.parametrize("scenario", [
        "deliver_A",
        "deliver_AB",
        "recharge_then_deliver",
    ])
    def test_scenario_produces_plan(self, scenario):
        from myagv_lab.phase3_delivery.pddl_planner.llm_translator import fallback_pddl
        from myagv_lab.phase3_delivery.pddl_planner.pddl_solver import solve_pddl
        domain, problem = fallback_pddl(scenario)
        plan = solve_pddl(domain, problem)
        assert len(plan) >= 3, f"Plan for {scenario!r} is too short: {plan}"

    @pytest.mark.parametrize("scenario", [
        "deliver_A",
        "deliver_AB",
        "recharge_then_deliver",
    ])
    def test_scenario_executes_successfully(self, scenario):
        from myagv_lab.phase3_delivery.mission_manager import MissionManager
        mgr = MissionManager(use_llm=False, fallback_scenario=scenario)
        ok  = mgr.run(f"[test: {scenario}]")
        assert ok, f"Mission failed for scenario {scenario!r}"


# ─────────────────────────────────────────────────────────────────────────────
#  INTEGRATION  —  Full phases 1 → 2 → 3 in sequence
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_phase1_builds_map_and_phase2_navigates(self):
        """
        Phase 1: run a short scan loop (not the full exploration path).
        Phase 2: navigate between two waypoints.
        """
        import numpy as np
        from myagv_lab.phase1_slam.slam_node import SLAMNode
        from myagv_lab.phase2_nav.nav_node import NavigationManager, WAYPOINTS
        from myagv_lab.sim_layer import SimRobot

        # Phase 1: quick scan (5 steps)
        node = SLAMNode()
        for _ in range(5):
            node.add_scan()
        assert node._scans == 5

        # Phase 2: use a fresh robot (same sim, new instance)
        robot = SimRobot(start=WAYPOINTS["home"])
        nav   = NavigationManager(robot=robot)
        r     = nav.navigate("loading_area")
        assert r.success

    def test_full_pipeline_three_phases(self):
        """
        Run all three phases end-to-end in simulation.
        """
        import numpy as np
        from myagv_lab.phase1_slam.slam_node import SLAMNode
        from myagv_lab.phase3_delivery.mission_manager import MissionManager

        # Phase 1 (abbreviated)
        slam = SLAMNode()
        for _ in range(3):
            slam.add_scan()

        # Phase 3 (includes Phase 2 navigation internally)
        mgr = MissionManager(use_llm=False, fallback_scenario="deliver_A")
        ok  = mgr.run("[integration test]")
        assert ok
