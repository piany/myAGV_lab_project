"""
myagv_lab/phase3_delivery/cobot_interface.py
============================================
ROS2 node that bridges /cobot_command → real cobot driver → /cobot_status.

In simulation, SimCobot handles this directly.
On real hardware, this node:
  1. Subscribes to /cobot_command  (std_msgs/String)
  2. Calls the real cobot motion API (e.g. pymycobot or MoveIt2)
  3. Publishes status to /cobot_status  (std_msgs/String)

Replace the placeholder blocks with your actual cobot driver calls.
"""

from __future__ import annotations

import logging
import time
import sys
from pathlib import Path

log = logging.getLogger("cobot_interface")

# ── Try to import ROS2; fall back to stub in sim ─────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    _HAS_ROS2 = True
except ImportError:
    _HAS_ROS2 = False
    log.warning("rclpy not available — cobot_interface will not run standalone.")


class CobotInterfaceNode:
    """
    Minimal ROS2 node for the cobot arm.

    Topics
    ------
    Subscribed : /cobot_command  (String)  — "LOAD" or "UNLOAD"
    Published  : /cobot_status   (String)  — "LOAD_COMPLETE", "UNLOAD_COMPLETE"
    """

    # ── Motion durations (real hardware) ─────────────────────────────────────
    LOAD_DURATION   = 5.0   # seconds for a full pick-and-place
    UNLOAD_DURATION = 3.0

    def __init__(self):
        if not _HAS_ROS2:
            raise RuntimeError("ROS2 (rclpy) is required for CobotInterfaceNode.")

        rclpy.init()
        self._node = Node("cobot_interface")
        self._pub  = self._node.create_publisher(String, "/cobot_status", 10)
        self._sub  = self._node.create_subscription(
            String, "/cobot_command", self._cmd_callback, 10
        )
        self._node.get_logger().info("CobotInterface ready — waiting for commands.")

    def _publish(self, status: str) -> None:
        msg = String(); msg.data = status
        self._pub.publish(msg)
        self._node.get_logger().info(f"/cobot_status → {status}")

    def _cmd_callback(self, msg: String) -> None:
        cmd = msg.data.strip().upper()
        self._node.get_logger().info(f"/cobot_command received: {cmd}")

        if cmd == "LOAD":
            self._execute_load()
        elif cmd == "UNLOAD":
            self._execute_unload()
        else:
            self._node.get_logger().warning(f"Unknown command: {cmd!r}")

    def _execute_load(self) -> None:
        """
        Execute the pick-and-place motion to load a package onto the AGV.

        ── Replace the time.sleep() below with your cobot driver calls ──

        Example (pymycobot / myCobot 280):
          from pymycobot.mycobot import MyCobot
          mc = MyCobot("/dev/ttyUSB0", 115200)
          mc.send_angles([0, 0, 0, 0, 0, 0], 50)     # home
          mc.send_angles([-10, -40, -80, 10, 0, 0], 50)  # above package
          mc.set_gripper_value(0, 50)                 # close gripper
          mc.send_angles([0, -20, -60, 0, 80, 0], 50)   # place on AGV
          mc.set_gripper_value(100, 50)               # open gripper
          mc.send_angles([0, 0, 0, 0, 0, 0], 50)     # retract to home

        Example (MoveIt2 via moveit2_tutorials):
          # self._moveit_group.set_named_target("pick_pose")
          # self._moveit_group.go(wait=True)
          # self._gripper.close()
          # self._moveit_group.set_named_target("place_on_agv")
          # self._moveit_group.go(wait=True)
          # self._gripper.open()
          # self._moveit_group.set_named_target("home")
          # self._moveit_group.go(wait=True)
        """
        self._publish("LOADING")
        self._node.get_logger().info(
            f"Executing LOAD (placeholder — replace with real driver)"
        )
        time.sleep(self.LOAD_DURATION)
        self._publish("LOAD_COMPLETE")

    def _execute_unload(self) -> None:
        """
        Execute the motion to remove a package from the AGV at delivery.

        ── Replace the time.sleep() below with your cobot driver calls ──
        """
        self._publish("UNLOADING")
        self._node.get_logger().info(
            f"Executing UNLOAD (placeholder — replace with real driver)"
        )
        time.sleep(self.UNLOAD_DURATION)
        self._publish("UNLOAD_COMPLETE")

    def spin(self) -> None:
        rclpy.spin(self._node)

    def shutdown(self) -> None:
        self._node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    node = CobotInterfaceNode()
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()
