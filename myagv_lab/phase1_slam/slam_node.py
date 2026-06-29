"""
myagv_lab/phase1_slam/slam_node.py
===================================
Phase 1 — SLAM

Sim mode  : Uses SimLidar + SimRobot to produce occupancy-grid scans,
            runs a lightweight incremental mapping loop, and saves the map
            as a PNG + YAML pair in the maps/ folder.

Real mode : Launches slam_toolbox (async_slam_toolbox_node) as a
            subprocess and relays the map topic; map saving is via
            the standard nav2_map_server CLI.

Student learning goals
----------------------
* Understand how successive LiDAR scans are accumulated into an
  occupancy grid.
* See how scan-matching uses correlation to estimate robot motion.
* Appreciate why loop-closure matters for large environments.

Usage
-----
  # Simulation
  python3 -m myagv_lab.phase1_slam.slam_node --sim

  # Real robot (ROS2 must be sourced)
  python3 -m myagv_lab.phase1_slam.slam_node --real
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import threading
import logging
from pathlib import Path
from typing import Optional

import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from myagv_lab.sim_layer import (
    USE_SIM, SimRobot, SimLidar, SimMap, LidarScan, Pose2D,
    get_robot, get_lidar, get_map, _CLOCK,
)

log = logging.getLogger("phase1_slam")

# ── map output directory ─────────────────────────────────────────────────────
MAPS_DIR = Path(__file__).resolve().parents[2] / "maps"
MAPS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  OCCUPANCY GRID
# ═══════════════════════════════════════════════════════════════════════════════

class OccupancyGrid:
    """
    Probabilistic 2-D occupancy grid.

    Each cell stores log-odds l = log(p/(1-p)):
      l > 0  → probably occupied
      l < 0  → probably free
      l = 0  → unknown
    """

    FREE_LOG_ODDS = -0.4
    OCC_LOG_ODDS  =  0.85
    MIN_LOG_ODDS  = -2.0
    MAX_LOG_ODDS  =  3.5

    def __init__(self, width_m: float, height_m: float, resolution: float = 0.05):
        self.resolution = resolution            # metres per cell
        self.width_cells  = int(width_m  / resolution)
        self.height_cells = int(height_m / resolution)
        # log-odds grid, initialised to 0 (unknown)
        self._grid = np.zeros((self.height_cells, self.width_cells), dtype=np.float32)
        # origin in world frame (bottom-left corner of grid)
        self.origin = Pose2D(0.0, 0.0, 0.0)

    # ── coordinate helpers ────────────────────────────────────────────────────

    def world_to_cell(self, wx: float, wy: float) -> tuple[int, int]:
        col = int((wx - self.origin.x) / self.resolution)
        row = int((wy - self.origin.y) / self.resolution)
        return row, col

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height_cells and 0 <= col < self.width_cells

    # ── Bresenham ray-update ─────────────────────────────────────────────────

    def integrate_scan(self, pose: Pose2D, scan: LidarScan) -> None:
        """
        Update the grid with one LiDAR scan.

        For each ray:
          • Mark all cells along the ray as FREE (inverse sensor model)
          • Mark the endpoint cell as OCCUPIED (if within range_max)
        """
        r0, c0 = self.world_to_cell(pose.x, pose.y)

        for i, r in enumerate(scan.ranges):
            angle = pose.yaw + scan.angle_min + i * scan.angle_increment
            hit = r < scan.range_max * 0.99   # true hit vs. max-range ray

            end_x = pose.x + r * math.cos(angle)
            end_y = pose.y + r * math.sin(angle)
            r1, c1 = self.world_to_cell(end_x, end_y)

            # Walk ray (Bresenham)
            for row, col in self._bresenham(r0, c0, r1, c1):
                if not self._in_bounds(row, col):
                    break
                if row == r1 and col == c1:
                    if hit:
                        self._grid[row, col] = min(
                            self.MAX_LOG_ODDS,
                            self._grid[row, col] + self.OCC_LOG_ODDS,
                        )
                else:
                    self._grid[row, col] = max(
                        self.MIN_LOG_ODDS,
                        self._grid[row, col] + self.FREE_LOG_ODDS,
                    )

    @staticmethod
    def _bresenham(r0: int, c0: int, r1: int, c1: int):
        """Yield (row, col) pairs along the Bresenham line from (r0,c0) to (r1,c1)."""
        dr = abs(r1 - r0); dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc
        r, c = r0, c0
        while True:
            yield r, c
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc; r += sr
            if e2 <  dr:
                err += dr; c += sc

    # ── Accessors ─────────────────────────────────────────────────────────────

    def probability(self, row: int, col: int) -> float:
        """Return occupancy probability p ∈ [0, 1]."""
        l = self._grid[row, col]
        return 1.0 / (1.0 + math.exp(-l))

    def to_image_array(self) -> np.ndarray:
        """
        Return a uint8 image:
          255 = free,  0 = occupied,  128 = unknown
        """
        img = np.full_like(self._grid, 128, dtype=np.uint8)
        probs = 1.0 / (1.0 + np.exp(-self._grid))
        img[probs > 0.65] = 0    # occupied
        img[probs < 0.35] = 255  # free
        return img

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, stem: str) -> tuple[Path, Path]:
        """
        Save the map as <stem>.png and <stem>.yaml.

        The YAML format is compatible with ROS2 nav2_map_server.
        """
        import imageio   # lightweight, available without cv2

        img      = self.to_image_array()
        png_path  = MAPS_DIR / f"{stem}.png"
        yaml_path = MAPS_DIR / f"{stem}.yaml"

        # Flip vertically so row 0 = bottom of image (ROS convention)
        imageio.imwrite(str(png_path), np.flipud(img))

        yaml_content = (
            f"image: {stem}.png\n"
            f"resolution: {self.resolution}\n"
            f"origin: [{self.origin.x:.4f}, {self.origin.y:.4f}, 0.0]\n"
            f"negate: 0\n"
            f"occupied_thresh: 0.65\n"
            f"free_thresh: 0.25\n"
        )
        yaml_path.write_text(yaml_content)
        log.info(f"Map saved → {png_path}  +  {yaml_path}")
        return png_path, yaml_path


# ═══════════════════════════════════════════════════════════════════════════════
#  POSE ESTIMATOR  (dead-reckoning + simple scan matching for sim)
# ═══════════════════════════════════════════════════════════════════════════════

class DeadReckoningEstimator:
    """
    Estimates robot pose from wheel odometry (sim: ground truth + noise).

    In a real system this would subscribe to /odom and run AMCL.
    Here we add small Gaussian drift to the ground-truth sim pose to
    show students that odometry alone accumulates error.
    """

    DRIFT_LINEAR  = 0.005   # std dev metres per metre travelled
    DRIFT_ANGULAR = 0.01    # std dev radians per radian rotated

    def __init__(self, initial_pose: Pose2D):
        self.estimated = initial_pose.copy()
        self._prev_true: Optional[Pose2D] = None
        import random
        self._rng = random.Random(42)

    def update(self, true_pose: Pose2D) -> Pose2D:
        """Add simulated odometry drift to the ground-truth pose delta."""
        if self._prev_true is None:
            self._prev_true = true_pose.copy()
            return self.estimated.copy()

        # Compute delta in ground truth
        dx  = true_pose.x   - self._prev_true.x
        dy  = true_pose.y   - self._prev_true.y
        dth = true_pose.yaw - self._prev_true.yaw
        dist = math.hypot(dx, dy)

        # Corrupt with noise
        dx  += self._rng.gauss(0, self.DRIFT_LINEAR  * max(dist, 0.001))
        dy  += self._rng.gauss(0, self.DRIFT_LINEAR  * max(dist, 0.001))
        dth += self._rng.gauss(0, self.DRIFT_ANGULAR * max(abs(dth), 0.001))

        self.estimated.x   += dx
        self.estimated.y   += dy
        self.estimated.yaw += dth

        self._prev_true = true_pose.copy()
        return self.estimated.copy()


# ═══════════════════════════════════════════════════════════════════════════════
#  SLAM NODE
# ═══════════════════════════════════════════════════════════════════════════════

class SLAMNode:
    """
    Simulation SLAM node.

    Drives the robot along a preset exploration path, accumulates
    LiDAR scans into an OccupancyGrid, and saves the finished map.

    The exploration path is a simple boustrophedon (lawnmower) pattern
    that covers the lab room.  Students can replace it with teleoperation
    by calling add_scan() from their own control loop.
    """

    # Exploration waypoints (sim coordinates, metres)
    EXPLORATION_PATH: list[Pose2D] = [
        Pose2D(1.0, 1.0, 0.0),
        Pose2D(8.0, 1.0, 0.0),
        Pose2D(8.0, 1.6, math.radians(180)),
        Pose2D(1.0, 1.6, math.radians(180)),
        Pose2D(1.0, 1.0, 0.0),
        Pose2D(4.0, 0.5, math.radians(180)),  # sweep loading area
        Pose2D(1.0, 1.0, 0.0),                # return home (loop closure!)
    ]

    SCAN_INTERVAL = 0.2   # seconds between scans

    def __init__(self):
        self.grid   = OccupancyGrid(width_m=14.4, height_m=3.2,
                                     resolution=0.05)
        self.robot  = get_robot(Pose2D(1.0, 1.0, 0.0))
        self.lidar  = get_lidar(self.robot)
        self.odom   = DeadReckoningEstimator(self.robot.position)
        self._scans = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def add_scan(self) -> None:
        """Integrate one LiDAR scan at the current estimated pose."""
        true_pose  = self.robot.position
        est_pose   = self.odom.update(true_pose)
        scan       = self.lidar.scan()
        self.grid.integrate_scan(est_pose, scan)
        self._scans += 1

    def run_exploration(self, save_name: str = "lab_map") -> tuple[Path, Path]:
        """
        Drive the robot along EXPLORATION_PATH while building the map.
        Returns (png_path, yaml_path) of the saved map.
        """
        log.info("=" * 60)
        log.info("Phase 1 — SLAM  (Simulation Mode)")
        log.info("=" * 60)
        log.info(f"Exploration path: {len(self.EXPLORATION_PATH)} waypoints")
        log.info(f"Map resolution  : {self.grid.resolution} m/cell")
        log.info(f"Map size        : {self.grid.width_cells}×{self.grid.height_cells} cells")
        log.info("")

        for i, wp in enumerate(self.EXPLORATION_PATH):
            log.info(f"  Waypoint {i+1}/{len(self.EXPLORATION_PATH)}: {wp}")

            # Scan while driving (background thread for navigation)
            nav_done   = threading.Event()
            nav_result = [None]

            def _nav():
                nav_result[0] = self.robot.navigate_to(wp, timeout=90.0)
                nav_done.set()

            t = threading.Thread(target=_nav, daemon=True)
            t.start()

            # Scan until navigation finishes
            while not nav_done.is_set():
                self.add_scan()
                time.sleep(self.SCAN_INTERVAL)

            t.join()

            if not nav_result[0].success:
                log.warning(f"  Navigation to waypoint {i+1} failed: "
                            f"{nav_result[0].message}")
            else:
                # Extra scans at the waypoint (robot is still)
                for _ in range(5):
                    self.add_scan()
                    time.sleep(self.SCAN_INTERVAL)

        log.info("")
        log.info(f"Exploration complete. Total scans: {self._scans}")
        png, yaml = self.grid.save(save_name)
        log.info("")
        log.info("Map saved successfully.")
        log.info(f"  PNG  : {png}")
        log.info(f"  YAML : {yaml}")
        log.info("")
        log.info("Transition to Phase 2:")
        log.info("  The nav2_params.yaml already points to this map.")
        log.info("  Simply launch nav2_launch.py to begin navigation.")
        return png, yaml

    def print_map_summary(self) -> None:
        """Print ASCII summary of the current occupancy grid."""
        img = self.grid.to_image_array()
        h, w = img.shape
        # Downsample to 80×20 for terminal display
        step_r = max(1, h // 20)
        step_c = max(1, w // 80)
        log.info("Map preview (█=occupied, ·=free, ?=unknown):")
        for r in range(0, h, step_r):
            row_str = ""
            for c in range(0, w, step_c):
                v = img[r, c]
                if v == 0:
                    row_str += "█"
                elif v == 255:
                    row_str += "·"
                else:
                    row_str += "?"
            print(row_str)


# ═══════════════════════════════════════════════════════════════════════════════
#  REAL-ROBOT MODE  (wraps slam_toolbox launch)
# ═══════════════════════════════════════════════════════════════════════════════

def run_real_slam(map_name: str = "lab_map") -> None:
    """
    Real-robot SLAM using slam_toolbox.

    This function:
      1. Prints setup instructions
      2. Waits for the student to confirm the LiDAR is publishing
      3. Launches slam_toolbox via subprocess
      4. Waits for the student to press Enter when mapping is done
      5. Calls map_saver_cli to save the map
    """
    import subprocess

    log.info("=" * 60)
    log.info("Phase 1 — SLAM  (Real Robot Mode)")
    log.info("=" * 60)
    log.info("")
    log.info("Pre-flight checks:")
    log.info("  1. myAGV drivers running?  →  ros2 topic echo /scan")
    log.info("  2. Odometry publishing?    →  ros2 topic echo /odom")
    log.info("  3. URDF / TF loaded?       →  ros2 run tf2_tools view_frames")
    log.info("")
    input("Press Enter when all checks pass …")

    cfg = Path(__file__).resolve().parents[2] / "config" / "slam_params.yaml"
    cmd = [
        "ros2", "run", "slam_toolbox", "async_slam_toolbox_node",
        "--ros-args", "--params-file", str(cfg),
    ]
    log.info(f"Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)

    log.info("")
    log.info("slam_toolbox is running.")
    log.info("Open RViz2 and add the /map topic to watch the map build.")
    log.info("Drive the robot using:")
    log.info("  ros2 run teleop_twist_keyboard teleop_twist_keyboard")
    log.info("")
    input("Press Enter when you are happy with the map …")

    proc.terminate()
    proc.wait()

    save_cmd = ["ros2", "run", "nav2_map_server", "map_saver_cli",
                "-f", str(MAPS_DIR / map_name)]
    log.info(f"Saving map: {' '.join(save_cmd)}")
    subprocess.run(save_cmd, check=True)
    log.info("Map saved.")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 — SLAM")
    parser.add_argument("--sim",  action="store_true", help="Run in simulation mode (default)")
    parser.add_argument("--real", action="store_true", help="Run on real hardware (ROS2 required)")
    parser.add_argument("--map-name", default="lab_map", help="Output map filename stem")
    args = parser.parse_args()

    use_real = args.real and not args.sim

    if use_real:
        run_real_slam(args.map_name)
    else:
        try:
            import imageio  # noqa: F401
        except ImportError:
            log.error("imageio not installed.  Run: pip3 install imageio")
            sys.exit(1)

        node = SLAMNode()
        png, yaml = node.run_exploration(args.map_name)
        node.print_map_summary()

        log.info("")
        log.info("✓  Phase 1 complete!")
        log.info(f"   Map files: {png.name}  +  {yaml.name}")


if __name__ == "__main__":
    main()
