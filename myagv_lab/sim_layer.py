"""
myagv_lab/sim_layer.py
======================
Simulation layer for the myAGV lab.

When ROS2 is NOT available (e.g. running on a laptop for development),
every import from this module provides drop-in fakes that behave
identically to the real ROS2 / Nav2 interfaces from the student code's
perspective.

Sim-to-real transition:
  - Set  USE_SIM = False  (or env var MYAGV_USE_SIM=0)
  - Install ROS2 Humble + nav2 on the robot
  - All other code stays exactly the same

Architecture:
  ┌──────────────────────────────────────────────────────────┐
  │   Student code  (phase1, phase2, phase3 nodes)           │
  ├──────────────────────────────────────────────────────────┤
  │   sim_layer.py  ← this file                              │
  │   • SimRobot    – 2-D pose tracker with motion model     │
  │   • SimNav2     – Nav2 ActionClient fake                 │
  │   • SimLidar    – ray-cast LiDAR on a simple map         │
  │   • SimCobot    – cobot arm fake with load/unload states │
  │   • SimClock    – shared wall-clock for the simulation   │
  ├──────────────────────────────────────────────────────────┤
  │   ROS2 (real hardware)  /  rclpy stubs (sim)             │
  └──────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import math
import os
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger("sim_layer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Global sim flag ──────────────────────────────────────────────────────────
USE_SIM: bool = os.environ.get("MYAGV_USE_SIM", "1") != "0"


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA TYPES  (mirror ROS2 geometry_msgs / nav2_msgs structures)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Pose2D:
    """2-D robot pose: position (m) + heading (rad)."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0   # radians, CCW positive

    def __str__(self) -> str:
        return f"Pose2D(x={self.x:.2f}, y={self.y:.2f}, yaw={math.degrees(self.yaw):.1f}°)"

    def distance_to(self, other: "Pose2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def copy(self) -> "Pose2D":
        return Pose2D(self.x, self.y, self.yaw)


@dataclass
class LidarScan:
    """Simplified LiDAR scan (360 rays, 1° apart by default)."""
    ranges: list[float] = field(default_factory=list)
    angle_min: float = 0.0          # radians
    angle_max: float = 2 * math.pi  # radians
    angle_increment: float = math.radians(1.0)
    range_max: float = 12.0         # metres
    range_min: float = 0.12         # metres


@dataclass
class NavResult:
    """Result of a navigation action."""
    success: bool
    message: str = ""
    final_pose: Optional[Pose2D] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  SIM CLOCK
# ═══════════════════════════════════════════════════════════════════════════════

class SimClock:
    """Shared simulation clock (wall-time, not ROS time)."""

    def __init__(self) -> None:
        self._start = time.monotonic()

    @property
    def now(self) -> float:
        return time.monotonic() - self._start

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


_CLOCK = SimClock()


# ═══════════════════════════════════════════════════════════════════════════════
#  SIM MAP  (simple occupancy grid for ray-casting)
# ═══════════════════════════════════════════════════════════════════════════════

# Default lab map: 10 m × 8 m room with a central pillar and a doorway.
# '1' = obstacle, '0' = free.  Resolution = 0.1 m / cell.
_DEFAULT_MAP_ASCII = [
    "############################################",
    "#                                          #",
    "#   L         D                            #",
    "#                                          #",
    "#              ####                        #",
    "#              #  #                        #",
    "#              ####                        #",
    "#                          S               #",
    "#                                          #",
    "############################################",
]
# L=loading_area, D=delivery_area, S=storage_area, ####=central pillar obstacle

class SimMap:
    """Occupancy grid for ray-casting and collision detection."""

    def __init__(self, ascii_map: list[str] = None, resolution: float = 0.2):
        self.resolution = resolution  # metres per cell
        raw = ascii_map or _DEFAULT_MAP_ASCII
        self.grid = [[1 if c == '#' else 0 for c in row] for row in raw]
        self.height = len(self.grid)
        self.width  = max(len(row) for row in self.grid)
        # Named waypoints in metres (map frame, origin = top-left corner)
        self.waypoints: dict[str, Pose2D] = {
            "home":            Pose2D(0.4, 0.6, 0.0),
            "loading_area":    Pose2D(0.8, 0.4, 0.0),
            "delivery_area":   Pose2D(2.0, 0.4, math.radians(90)),
            "storage_area":    Pose2D(5.0, 0.4, math.radians(180)),
            "charger_station": Pose2D(7.0, 0.4, math.radians(270)),
        }

    def is_free(self, x: float, y: float) -> bool:
        col = int(x / self.resolution)
        row = int(y / self.resolution)
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        return self.grid[row][col] == 0

    def ray_cast(self, origin: Pose2D, angle: float, max_range: float = 12.0) -> float:
        """Return distance to first obstacle along a ray, up to max_range."""
        step = self.resolution * 0.5
        d = 0.0
        while d < max_range:
            x = origin.x + d * math.cos(angle)
            y = origin.y + d * math.sin(angle)
            if not self.is_free(x, y):
                return d
            d += step
        return max_range


_SIM_MAP = SimMap()


# ═══════════════════════════════════════════════════════════════════════════════
#  SIM ROBOT  (differential-drive motion model)
# ═══════════════════════════════════════════════════════════════════════════════

class SimRobot:
    """
    2-D robot with a simple unicycle motion model.

    navigate_to() blocks until the robot arrives or a timeout occurs,
    printing progress — exactly as the real Nav2 ActionClient would behave
    from the caller's perspective.
    """

    MAX_LINEAR_VEL  = 0.3    # m/s
    MAX_ANGULAR_VEL = 1.0    # rad/s
    GOAL_TOLERANCE  = 0.10   # metres
    DT              = 0.05   # simulation step (s)

    def __init__(self, start: Pose2D = None, sim_map: SimMap = None):
        self.pose   = (start or Pose2D(1.0, 1.0, 0.0)).copy()
        self._map   = sim_map or _SIM_MAP
        self._lock  = threading.Lock()
        self._carrying: Optional[str] = None   # package name or None

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def position(self) -> Pose2D:
        with self._lock:
            return self.pose.copy()

    @property
    def carrying(self) -> Optional[str]:
        return self._carrying

    # ── Navigation ────────────────────────────────────────────────────────────

    def navigate_to(self, goal: Pose2D, timeout: float = 60.0) -> NavResult:
        """
        Move the robot from its current pose to goal using a simple
        proportional controller.  Prints progress every 0.5 s.
        """
        log.info(f"[SimRobot] navigate_to {goal}")
        start_time = _CLOCK.now
        last_log   = start_time

        while True:
            elapsed = _CLOCK.now - start_time
            if elapsed > timeout:
                return NavResult(False, f"Navigation timeout after {timeout:.0f}s",
                                 self.pose.copy())

            with self._lock:
                dx  = goal.x - self.pose.x
                dy  = goal.y - self.pose.y
                dist = math.hypot(dx, dy)

                if dist < self.GOAL_TOLERANCE:
                    self.pose.yaw = goal.yaw   # snap to final heading
                    log.info(f"[SimRobot] Arrived at {goal}  (t={elapsed:.1f}s)")
                    return NavResult(True, "Goal reached", self.pose.copy())

                # Heading error
                desired_yaw = math.atan2(dy, dx)
                yaw_err = self._angle_diff(desired_yaw, self.pose.yaw)

                # Proportional control
                # Suppress linear velocity when heading error is large
                # (prevents arc-clipping walls during U-turns)
                heading_ok = abs(yaw_err) < math.radians(30)
                v = min(self.MAX_LINEAR_VEL, 0.5 * dist) if heading_ok else 0.0
                w = min(self.MAX_ANGULAR_VEL, max(-self.MAX_ANGULAR_VEL,
                                                   2.0 * yaw_err))

                # Step
                self.pose.x   += v * math.cos(self.pose.yaw) * self.DT
                self.pose.y   += v * math.sin(self.pose.yaw) * self.DT
                self.pose.yaw += w * self.DT
                self.pose.yaw  = self._wrap_angle(self.pose.yaw)

                # Collision check
                if not self._map.is_free(self.pose.x, self.pose.y):
                    log.warning("[SimRobot] Collision detected — stopping")
                    return NavResult(False, "Collision", self.pose.copy())

            # Progress log every 0.5 s
            now = _CLOCK.now
            if now - last_log >= 0.5:
                log.info(f"[SimRobot]   → dist={dist:.2f}m  "
                         f"pose=({self.pose.x:.2f},{self.pose.y:.2f})  "
                         f"t={elapsed:.1f}s")
                last_log = now

            time.sleep(self.DT)

    # ── Cargo ─────────────────────────────────────────────────────────────────

    def pick_up(self, package: str) -> bool:
        if self._carrying is not None:
            log.warning(f"[SimRobot] Already carrying {self._carrying!r}")
            return False
        self._carrying = package
        log.info(f"[SimRobot] Picked up {package!r}")
        return True

    def put_down(self) -> Optional[str]:
        pkg = self._carrying
        self._carrying = None
        log.info(f"[SimRobot] Put down {pkg!r}")
        return pkg

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Signed angular difference a - b in [-π, π]."""
        d = a - b
        return (d + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi


# ═══════════════════════════════════════════════════════════════════════════════
#  SIM LIDAR
# ═══════════════════════════════════════════════════════════════════════════════

class SimLidar:
    """
    Ray-cast LiDAR sensor attached to a SimRobot.
    Produces a LidarScan at a configurable rate.
    """

    def __init__(self, robot: SimRobot, sim_map: SimMap = None,
                 num_rays: int = 360, max_range: float = 12.0,
                 noise_std: float = 0.01):
        self._robot     = robot
        self._map       = sim_map or _SIM_MAP
        self._num_rays  = num_rays
        self._max_range = max_range
        self._noise_std = noise_std
        self._inc       = 2 * math.pi / num_rays

    def scan(self) -> LidarScan:
        """Return one LiDAR scan from the robot's current pose."""
        import random
        pose   = self._robot.position
        ranges = []
        for i in range(self._num_rays):
            angle = pose.yaw + i * self._inc
            r = self._map.ray_cast(pose, angle, self._max_range)
            # Add Gaussian noise
            r = max(0.12, r + random.gauss(0, self._noise_std))
            ranges.append(r)
        return LidarScan(
            ranges=ranges,
            angle_min=0.0,
            angle_max=2 * math.pi,
            angle_increment=self._inc,
            range_max=self._max_range,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  SIM COBOT  (manipulator arm)
# ═══════════════════════════════════════════════════════════════════════════════

class SimCobot:
    """
    Simulated cobot arm.

    Responds to load() / unload() calls with a configurable delay,
    prints progress, and publishes a /cobot_status string via callbacks.
    """

    LOAD_DURATION   = 3.0   # seconds
    UNLOAD_DURATION = 2.0

    def __init__(self, location: str = "loading_area",
                 on_status: Callable[[str], None] = None):
        self.location   = location
        self._on_status = on_status or (lambda s: None)
        self._state     = "IDLE"

    def _publish(self, status: str) -> None:
        self._state = status
        log.info(f"[SimCobot] status → {status}")
        self._on_status(status)

    def load(self, package: str, agv: SimRobot) -> bool:
        """
        Place package on the AGV.  Blocks for LOAD_DURATION seconds.
        Returns True on success.
        """
        self._publish("LOADING")
        log.info(f"[SimCobot] Loading {package!r} onto AGV …")
        time.sleep(self.LOAD_DURATION)
        agv.pick_up(package)
        self._publish("LOAD_COMPLETE")
        return True

    def unload(self, package: str, agv: SimRobot) -> bool:
        """
        Remove package from the AGV.  Blocks for UNLOAD_DURATION seconds.
        Returns True on success.
        """
        self._publish("UNLOADING")
        log.info(f"[SimCobot] Unloading {package!r} from AGV …")
        time.sleep(self.UNLOAD_DURATION)
        agv.put_down()
        self._publish("UNLOAD_COMPLETE")
        return True

    @property
    def state(self) -> str:
        return self._state


# ═══════════════════════════════════════════════════════════════════════════════
#  ROS2 STUBS  (used when USE_SIM=True so rclpy is not required)
# ═══════════════════════════════════════════════════════════════════════════════

class _StubNode:
    """Minimal Node stub so student code calling self.get_logger() works."""

    class _Logger:
        def info(self, msg):  log.info(msg)
        def warn(self, msg):  log.warning(msg)
        def warning(self, msg): log.warning(msg)
        def error(self, msg): log.error(msg)
        def debug(self, msg): log.debug(msg)

    def __init__(self, name: str = "sim_node"):
        self._name   = name
        self._logger = self._Logger()

    def get_logger(self):
        return self._logger

    def get_clock(self):
        class _C:
            def now(self):
                class _T:
                    def to_msg(self): return None
                return _T()
        return _C()

    def create_publisher(self, *a, **kw):
        class _P:
            def publish(self, msg): pass
        return _P()

    def create_subscription(self, *a, **kw):
        return None

    def create_timer(self, *a, **kw):
        return None

    def destroy_node(self):
        pass


def make_node(name: str):
    """Return a real rclpy Node or a stub depending on USE_SIM."""
    if USE_SIM:
        return _StubNode(name)
    import rclpy                          # noqa: F401
    from rclpy.node import Node
    class _RealNode(Node):
        def __init__(self):
            super().__init__(name)
    return _RealNode()


# ═══════════════════════════════════════════════════════════════════════════════
#  FACTORY  — one-stop shop to get sim or real objects
# ═══════════════════════════════════════════════════════════════════════════════

def get_robot(start: Pose2D = None) -> SimRobot:
    if USE_SIM:
        return SimRobot(start)
    raise RuntimeError("get_robot() called in real mode — use your ROS2 driver directly.")


def get_lidar(robot: SimRobot) -> SimLidar:
    if USE_SIM:
        return SimLidar(robot)
    raise RuntimeError("get_lidar() called in real mode.")


def get_cobot(on_status: Callable[[str], None] = None) -> SimCobot:
    if USE_SIM:
        return SimCobot(on_status=on_status)
    raise RuntimeError("get_cobot() called in real mode.")


def get_map() -> SimMap:
    return _SIM_MAP
