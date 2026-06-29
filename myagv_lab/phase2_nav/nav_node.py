"""
myagv_lab/phase2_nav/nav_node.py
=================================
Phase 2 — Navigation

Sim mode  : Uses SimRobot.navigate_to() with the SimMap loaded from
            the map saved in Phase 1.  Visualises the path in ASCII
            and logs arrival times.

Real mode : Wraps the Nav2 navigate_to_pose ActionClient.  The same
            NavigationManager API is used in both modes so Phase 3
            can call navigate() without knowing which mode is active.

Student learning goals
----------------------
* See how AMCL localises the robot on the saved map.
* Understand global vs. local costmaps and their parameters.
* Write Python code that sends goals to Nav2 and waits for results.

Usage
-----
  # Simulation — interactive waypoint selection
  python3 -m myagv_lab.phase2_nav.nav_node --sim

  # Real robot
  python3 -m myagv_lab.phase2_nav.nav_node --real
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from myagv_lab.sim_layer import (
    USE_SIM, SimRobot, SimMap, Pose2D, NavResult, get_robot, get_map, _CLOCK,
)

log = logging.getLogger("phase2_nav")


# ═══════════════════════════════════════════════════════════════════════════════
#  WAYPOINT REGISTRY  (single source of truth for both sim and real)
# ═══════════════════════════════════════════════════════════════════════════════

# Real-robot coordinates are in the saved map frame (metres).
# Update these after running Phase 1 and reading coordinates from RViz2.
WAYPOINTS: dict[str, Pose2D] = {
    "home":            Pose2D(0.4, 0.6,  0.0),
    "loading_area":    Pose2D(0.8, 0.4,  0.0),
    "delivery_area":   Pose2D(2.0, 0.4,  math.radians(90)),
    "storage_area":    Pose2D(5.0, 0.4,  math.radians(180)),
    "charger_station": Pose2D(7.0, 0.4,  math.radians(270)),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  ASCII MAP VISUALISER  (sim only)
# ═══════════════════════════════════════════════════════════════════════════════

class AsciiVisualiser:
    """
    Renders the SimMap + robot pose + planned path as ASCII art
    in the terminal — gives students an intuitive view of navigation.
    """

    CELL = 0.5     # metres per ASCII character
    SYMBOLS = {
        "obstacle": "▓",
        "free":     "·",
        "robot":    "R",
        "goal":     "G",
        "start":    "S",
        "path":     "○",
    }

    def __init__(self, sim_map: SimMap):
        self._map = sim_map
        self.cols = int(sim_map.width  * sim_map.resolution / self.CELL)
        self.rows = int(sim_map.height * sim_map.resolution / self.CELL)

    def render(self, robot_pose: Pose2D,
               goal: Optional[Pose2D] = None,
               path: list[Pose2D] = None,
               start: Optional[Pose2D] = None) -> str:
        # Build base grid
        grid = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                wx = c * self.CELL
                wy = r * self.CELL
                row.append("·" if self._map.is_free(wx, wy) else "▓")
            grid.append(row)

        def world_to_rc(pose: Pose2D) -> tuple[int, int]:
            c = int(pose.x / self.CELL)
            r = int(pose.y / self.CELL)
            return max(0, min(r, self.rows - 1)), max(0, min(c, self.cols - 1))

        # Draw path
        if path:
            for wp in path:
                pr, pc = world_to_rc(wp)
                if grid[pr][pc] == "·":
                    grid[pr][pc] = "○"

        # Draw start
        if start:
            sr, sc = world_to_rc(start)
            grid[sr][sc] = "S"

        # Draw goal
        if goal:
            gr, gc = world_to_rc(goal)
            grid[gr][gc] = "G"

        # Draw robot (on top)
        rr, rc = world_to_rc(robot_pose)
        direction_chars = {
            "E": "►", "W": "◄", "N": "▲", "S": "▼",
        }
        # Heading to cardinal
        deg = math.degrees(robot_pose.yaw) % 360
        if   deg < 45 or deg >= 315: ch = "►"
        elif deg < 135:              ch = "▲"
        elif deg < 225:              ch = "◄"
        else:                        ch = "▼"
        grid[rr][rc] = ch

        # Build string
        lines = ["┌" + "─" * self.cols + "┐"]
        for row in reversed(grid):     # flip Y so north is up
            lines.append("│" + "".join(row) + "│")
        lines.append("└" + "─" * self.cols + "┘")

        # Waypoint legend
        lines.append("  Waypoints:")
        for name, wp in WAYPOINTS.items():
            wr, wc = world_to_rc(wp)
            lines.append(f"    {name:20s}  ({wp.x:.1f}, {wp.y:.1f})")

        return "\n".join(lines)

    def print(self, *args, **kwargs):
        print(self.render(*args, **kwargs))


# ═══════════════════════════════════════════════════════════════════════════════
#  NAVIGATION MANAGER  (unified sim / real interface)
# ═══════════════════════════════════════════════════════════════════════════════

class NavigationManager:
    """
    Single API for navigation, usable in both simulation and real modes.

    Sim  : delegates to SimRobot.navigate_to()
    Real : delegates to Nav2 navigate_to_pose ActionClient

    Methods
    -------
    navigate(location: str) -> NavResult
        Navigate to a named waypoint.

    navigate_to_pose(goal: Pose2D) -> NavResult
        Navigate to an arbitrary pose.

    current_pose() -> Pose2D
        Return the robot's current estimated pose.
    """

    def __init__(self, robot: SimRobot = None,
                 on_status: Callable[[str], None] = None):
        self._sim  = USE_SIM
        self._robot: Optional[SimRobot] = robot
        self._on_status = on_status or (lambda s: None)

        if self._sim and self._robot is None:
            self._robot = get_robot()

        if not self._sim:
            self._init_real()

    # ── Real-robot setup ──────────────────────────────────────────────────────

    def _init_real(self) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.action import ActionClient
        from nav2_msgs.action import NavigateToPose

        if not rclpy.ok():
            rclpy.init()
        self._node   = Node("nav_manager")
        self._client = ActionClient(self._node, NavigateToPose, "navigate_to_pose")
        log.info("[NavManager] Waiting for Nav2 action server …")
        self._client.wait_for_server()
        log.info("[NavManager] Nav2 ready.")

    # ── Core API ──────────────────────────────────────────────────────────────

    def navigate(self, location: str) -> NavResult:
        """Navigate to a named waypoint."""
        if location not in WAYPOINTS:
            return NavResult(False,
                             f"Unknown location {location!r}. "
                             f"Known: {list(WAYPOINTS.keys())}")
        goal = WAYPOINTS[location]
        log.info(f"[NavManager] navigate({location!r})  →  {goal}")
        self._on_status(f"NAVIGATING_TO_{location.upper()}")
        result = self.navigate_to_pose(goal)
        if result.success:
            self._on_status(f"ARRIVED_AT_{location.upper()}")
        else:
            self._on_status(f"FAILED_{location.upper()}")
        return result

    def navigate_to_pose(self, goal: Pose2D) -> NavResult:
        """Navigate to an arbitrary pose."""
        if self._sim:
            return self._robot.navigate_to(goal)
        else:
            return self._nav_real(goal)

    def current_pose(self) -> Pose2D:
        if self._sim:
            return self._robot.position
        else:
            # In real mode, read from /odom (simplified)
            return Pose2D()   # students: replace with TF listener

    # ── Real navigation ───────────────────────────────────────────────────────

    def _nav_real(self, goal: Pose2D) -> NavResult:
        import rclpy
        from nav2_msgs.action import NavigateToPose
        from geometry_msgs.msg import PoseStamped

        goal_msg = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp    = self._node.get_clock().now().to_msg()
        pose.pose.position.x = goal.x
        pose.pose.position.y = goal.y
        yaw = goal.yaw
        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)
        goal_msg.pose = pose

        future  = self._client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self._node, future)
        handle  = future.result()

        if not handle.accepted:
            return NavResult(False, "Goal rejected by Nav2")

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, result_future)
        return NavResult(True, "Arrived", goal)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        if not self._sim:
            self._node.destroy_node()
            import rclpy
            rclpy.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE DEMO  (Phase 2 student exercise)
# ═══════════════════════════════════════════════════════════════════════════════

def run_sim_demo() -> None:
    """
    Interactive CLI demo: student picks waypoints, robot navigates,
    ASCII map updates live.
    """
    log.info("=" * 60)
    log.info("Phase 2 — Navigation  (Simulation Mode)")
    log.info("=" * 60)

    robot  = get_robot(WAYPOINTS["home"])
    nav    = NavigationManager(robot=robot)
    vis    = AsciiVisualiser(get_map())

    print("\nInitial map state:")
    vis.print(robot.position)
    print()

    names = list(WAYPOINTS.keys())

    while True:
        print("\nAvailable destinations:")
        for i, name in enumerate(names):
            wp = WAYPOINTS[name]
            print(f"  {i+1}. {name:20s}  ({wp.x:.1f}, {wp.y:.1f})")
        print("  q. Quit")

        choice = input("\nEnter number or 'q': ").strip().lower()
        if choice == "q":
            break

        try:
            idx = int(choice) - 1
            if not 0 <= idx < len(names):
                raise ValueError
        except ValueError:
            print("Invalid choice — try again.")
            continue

        destination = names[idx]
        goal_pose   = WAYPOINTS[destination]
        start_pose  = robot.position.copy()

        log.info(f"\nNavigating to {destination!r} …")
        t0     = _CLOCK.now
        result = nav.navigate(destination)
        elapsed = _CLOCK.now - t0

        print()
        print(f"Result: {'✓ Success' if result.success else '✗ Failed'}  "
              f"— {result.message}  (took {elapsed:.1f}s)")
        vis.print(robot.position, goal=goal_pose, start=start_pose)

    log.info("Phase 2 demo complete.")


def run_scripted_demo() -> None:
    """
    Non-interactive demo: navigate a pre-set sequence of waypoints.
    Used in Phase 3 internally.
    """
    log.info("=" * 60)
    log.info("Phase 2 — Scripted Navigation  (Simulation Mode)")
    log.info("=" * 60)

    robot = get_robot(WAYPOINTS["home"])
    nav   = NavigationManager(robot=robot)

    sequence = ["loading_area", "delivery_area", "home"]
    for dest in sequence:
        result = nav.navigate(dest)
        log.info(f"  {dest}: {'OK' if result.success else 'FAILED'}")
        time.sleep(0.5)

    log.info("Scripted navigation complete.")


def run_real_demo() -> None:
    """Guided checklist + Nav2 interactive navigation for real hardware."""
    log.info("=" * 60)
    log.info("Phase 2 — Navigation  (Real Robot Mode)")
    log.info("=" * 60)
    log.info("")
    log.info("Pre-flight:")
    log.info("  ros2 launch myagv_lab nav2_launch.py")
    log.info("  Set 2D Pose Estimate in RViz2")
    log.info("")
    input("Press Enter when Nav2 is running and initial pose is set …")

    nav = NavigationManager()

    names = list(WAYPOINTS.keys())
    while True:
        print("\nDestinations:")
        for i, name in enumerate(names):
            print(f"  {i+1}. {name}")
        print("  q. Quit")
        choice = input("Choice: ").strip().lower()
        if choice == "q":
            break
        try:
            dest = names[int(choice) - 1]
        except (ValueError, IndexError):
            continue
        result = nav.navigate(dest)
        print(f"Result: {'✓' if result.success else '✗'} {result.message}")

    nav.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 — Navigation")
    parser.add_argument("--sim",      action="store_true")
    parser.add_argument("--real",     action="store_true")
    parser.add_argument("--scripted", action="store_true",
                        help="Run the pre-set waypoint sequence (no user input)")
    args = parser.parse_args()

    if args.real:
        run_real_demo()
    elif args.scripted:
        run_scripted_demo()
    else:
        run_sim_demo()


if __name__ == "__main__":
    main()
