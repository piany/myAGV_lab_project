"""
myagv_lab/phase2_nav/nav_viz.py
================================
Matplotlib visualiser for Phase 2 navigation.

Shows the sim map, named waypoints, robot heading arrow, and a live
trail that updates while the robot is moving.

Usage (standalone demo):
  python3 -m myagv_lab.phase2_nav.nav_viz

Programmatic usage:
  from myagv_lab.phase2_nav.nav_viz import NavVisualizer
  viz = NavVisualizer(sim_map, waypoints)
  viz.draw(robot.position, goal=goal_pose, trail=pose_list)
  viz.live_navigate(robot, nav_manager, "delivery_area")

Architecture note
-----------------
Matplotlib's GUI must run on the main thread.  live_navigate() therefore
runs the navigation call in a daemon thread and pumps the event loop
(plt.pause) on the caller's thread.
"""

from __future__ import annotations

import math
import sys
import threading
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from myagv_lab.sim_layer import SimMap, SimRobot, Pose2D, _CLOCK

# ── Colour palette ────────────────────────────────────────────────────────────
_C = {
    "wall":    "#2d2d2d",
    "free":    "#f5f5f0",
    "trail":   "#4fc3f7",
    "robot":   "#ef5350",
    "goal":    "#66bb6a",
    "start":   "#ffa726",
    "wp":      "#7e57c2",
    "wp_text": "#4a148c",
}


class NavVisualizer:
    """
    Matplotlib-based navigation visualiser.

    Parameters
    ----------
    sim_map   : SimMap   — the occupancy grid
    waypoints : dict     — name → Pose2D (from nav_node.WAYPOINTS)
    title     : str      — window / figure title
    """

    ARROW_LEN  = 0.25   # metres — length of the robot heading arrow
    TRAIL_MAX  = 2000   # max trail points kept

    def __init__(self, sim_map: SimMap,
                 waypoints: dict[str, Pose2D],
                 title: str = "Phase 2 — Navigation Visualiser"):
        self._map       = sim_map
        self._waypoints = waypoints
        self._title     = title

        self._trail_x: list[float] = []
        self._trail_y: list[float] = []

        self._fig, self._ax = plt.subplots(figsize=(12, 5))
        self._fig.canvas.manager.set_window_title(title)
        plt.ion()

        self._map_img  = None   # rendered once
        self._robot_arrow: Optional[mpatches.FancyArrow] = None
        self._trail_line = None
        self._goal_marker = None
        self._start_marker = None

        self._build_static_layer()

    # ── Static layer (map + waypoints) ────────────────────────────────────────

    def _build_static_layer(self) -> None:
        ax = self._ax
        ax.set_facecolor(_C["free"])
        ax.set_title(self._title, fontsize=11, pad=8)
        ax.set_xlabel("x  (m)")
        ax.set_ylabel("y  (m)")
        ax.set_aspect("equal")

        # Draw obstacle cells as filled rectangles
        res = self._map.resolution
        for row_idx, row in enumerate(self._map.grid):
            for col_idx, cell in enumerate(row):
                if cell == 1:
                    ax.add_patch(plt.Rectangle(
                        (col_idx * res, row_idx * res),
                        res, res,
                        color=_C["wall"], zorder=1,
                    ))

        # Map boundary
        w = self._map.width  * res
        h = self._map.height * res
        ax.set_xlim(-0.1, w + 0.1)
        ax.set_ylim(-0.1, h + 0.1)

        # Waypoints
        for name, wp in self._waypoints.items():
            ax.plot(wp.x, wp.y, "o", color=_C["wp"],
                    markersize=8, zorder=3)
            ax.annotate(
                name, xy=(wp.x, wp.y),
                xytext=(4, 6), textcoords="offset points",
                fontsize=7.5, color=_C["wp_text"], zorder=4,
            )

        # Grid lines (light)
        ax.grid(True, linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)

        # Legend
        legend_handles = [
            mpatches.Patch(color=_C["wall"],  label="Wall"),
            mpatches.Patch(color=_C["free"],  label="Free space"),
            plt.Line2D([0], [0], color=_C["trail"],  lw=2,  label="Trail"),
            mpatches.Patch(color=_C["robot"], label="Robot"),
            mpatches.Patch(color=_C["goal"],  label="Goal"),
            mpatches.Patch(color=_C["start"], label="Start"),
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=_C["wp"], markersize=8, label="Waypoint"),
        ]
        ax.legend(handles=legend_handles, loc="upper right",
                  fontsize=7, framealpha=0.85)

    # ── Dynamic update ────────────────────────────────────────────────────────

    def draw(self, robot_pose: Pose2D,
             goal: Optional[Pose2D] = None,
             start: Optional[Pose2D] = None,
             trail: Optional[list[Pose2D]] = None,
             status: str = "") -> None:
        """
        Redraw the dynamic elements (robot, trail, goal, start).

        Call this whenever you want the plot to reflect a new state.
        """
        ax = self._ax

        # ── Trail ─────────────────────────────────────────────────────────────
        if trail is not None:
            self._trail_x = [p.x for p in trail]
            self._trail_y = [p.y for p in trail]
        else:
            # Accumulate automatically
            self._trail_x.append(robot_pose.x)
            self._trail_y.append(robot_pose.y)
            if len(self._trail_x) > self.TRAIL_MAX:
                self._trail_x = self._trail_x[-self.TRAIL_MAX:]
                self._trail_y = self._trail_y[-self.TRAIL_MAX:]

        if self._trail_line is not None:
            self._trail_line.remove()
        self._trail_line, = ax.plot(
            self._trail_x, self._trail_y,
            color=_C["trail"], linewidth=1.5, zorder=2, alpha=0.7,
        )

        # ── Start marker ──────────────────────────────────────────────────────
        if self._start_marker is not None:
            self._start_marker.remove()
            self._start_marker = None
        if start is not None:
            self._start_marker, = ax.plot(
                start.x, start.y, "*",
                color=_C["start"], markersize=14, zorder=4,
            )

        # ── Goal marker ───────────────────────────────────────────────────────
        if self._goal_marker is not None:
            self._goal_marker.remove()
            self._goal_marker = None
        if goal is not None:
            self._goal_marker, = ax.plot(
                goal.x, goal.y, "D",
                color=_C["goal"], markersize=10, zorder=4,
            )

        # ── Robot arrow ───────────────────────────────────────────────────────
        if self._robot_arrow is not None:
            self._robot_arrow.remove()
        dx = self.ARROW_LEN * math.cos(robot_pose.yaw)
        dy = self.ARROW_LEN * math.sin(robot_pose.yaw)
        self._robot_arrow = ax.annotate(
            "", xy=(robot_pose.x + dx, robot_pose.y + dy),
            xytext=(robot_pose.x, robot_pose.y),
            arrowprops=dict(arrowstyle="-|>", color=_C["robot"],
                            lw=2.5, mutation_scale=18),
            zorder=5,
        )
        # Robot body circle
        ax.add_patch(plt.Circle(
            (robot_pose.x, robot_pose.y), 0.10,
            color=_C["robot"], zorder=5,
        ))

        # Status text in top-left
        ax.set_title(
            f"{self._title}    {status}",
            fontsize=11, pad=8,
        )

        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    def clear_trail(self) -> None:
        """Reset the accumulated trail (call at the start of each trip)."""
        self._trail_x.clear()
        self._trail_y.clear()

    # ── Live navigation helper ────────────────────────────────────────────────

    def live_navigate(self, robot: SimRobot,
                      nav_manager,
                      destination: str,
                      waypoints: dict[str, Pose2D] = None) -> bool:
        """
        Navigate to *destination* while updating this visualiser live.

        Navigation runs in a daemon thread; the matplotlib event loop is
        driven on the calling thread via plt.pause().

        Returns True if navigation succeeded.
        """
        wps  = waypoints or self._waypoints
        goal = wps.get(destination)
        if goal is None:
            print(f"[NavViz] Unknown destination {destination!r}")
            return False

        start = robot.position.copy()
        self.clear_trail()
        result_box: list = []

        def _worker():
            result_box.append(nav_manager.navigate(destination))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        while thread.is_alive():
            self.draw(robot.position, goal=goal, start=start,
                      status=f"→ {destination}  …")
            plt.pause(0.08)

        thread.join()
        result = result_box[0] if result_box else None

        # Final frame
        ok  = result.success if result else False
        msg = result.message if result else "?"
        self.draw(robot.position, goal=goal, start=start,
                  status=f"{'✓' if ok else '✗'}  {destination}  — {msg}")
        return ok

    def save(self, path: str) -> None:
        """Save the current figure to a PNG file."""
        self._fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[NavViz] Saved → {path}")

    def show(self) -> None:
        """Block until the window is closed."""
        plt.ioff()
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDALONE DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import os, sys
    os.environ.setdefault("MYAGV_USE_SIM", "1")

    from myagv_lab.sim_layer import get_robot, get_map
    from myagv_lab.phase2_nav.nav_node import NavigationManager, WAYPOINTS

    robot = get_robot(WAYPOINTS["home"])
    nav   = NavigationManager(robot=robot)
    viz   = NavVisualizer(get_map(), WAYPOINTS)

    sequence = ["loading_area", "delivery_area", "storage_area", "home"]

    print("╔══════════════════════════════════════════╗")
    print("║  Phase 2 — Navigation Visualiser Demo    ║")
    print("╚══════════════════════════════════════════╝")
    print(f"Route: home → {' → '.join(sequence)}\n")

    for dest in sequence:
        print(f"Navigating to {dest!r} …")
        ok = viz.live_navigate(robot, nav, dest)
        print(f"  {'OK' if ok else 'FAILED'}\n")
        plt.pause(0.5)

    print("Route complete.  Close the window to exit.")
    viz.show()


if __name__ == "__main__":
    main()
