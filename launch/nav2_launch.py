"""
launch/nav2_launch.py
Real-robot navigation launch (ROS2 Humble).
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    nav2_dir  = get_package_share_directory("nav2_bringup")
    myagv_dir = get_package_share_directory("myagv_lab")

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_dir, "launch", "bringup_launch.py")
            ),
            launch_arguments={
                "map":          os.path.join(myagv_dir, "maps", "lab_map.yaml"),
                "use_sim_time": "false",
                "params_file":  os.path.join(myagv_dir, "config", "nav2_params.yaml"),
            }.items(),
        ),
    ])
