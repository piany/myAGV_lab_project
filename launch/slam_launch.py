"""
launch/slam_launch.py
Real-robot SLAM launch (ROS2 Humble).
Not used in simulation — Phase 1 sim uses OccupancyGrid directly.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    cfg = os.path.join(
        get_package_share_directory("myagv_lab"),
        "config", "slam_params.yaml",
    )
    return LaunchDescription([
        # myAGV hardware driver (LiDAR + motor controllers)
        Node(
            package="myagv_bringup",
            executable="myagv_driver",
            name="myagv_driver",
            output="screen",
        ),
        # SLAM
        Node(
            package="slam_toolbox",
            executable="async_slam_toolbox_node",
            name="slam_toolbox",
            parameters=[cfg],
            output="screen",
        ),
    ])
