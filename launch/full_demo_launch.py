"""
launch/full_demo_launch.py
Launches Nav2 + cobot_interface + mission_manager for the real-robot demo.

Required environment variable (set before launching):
  export DEEPSEEK_API_KEY="sk-..."

Launch arguments:
  task        — Natural language task description (default: deliver_A fallback)
  use_fallback — "true" to skip the LLM and use hard-coded PDDL (default: false)
  scenario    — Fallback scenario name (default: deliver_A)

Examples:
  ros2 launch myagv_lab full_demo_launch.py \\
      task:="Deliver package_A to the delivery area and return home."

  ros2 launch myagv_lab full_demo_launch.py \\
      use_fallback:=true scenario:=deliver_AB
"""
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    nav2_dir  = get_package_share_directory("nav2_bringup")
    myagv_dir = get_package_share_directory("myagv_lab")

    # ── Launch arguments ──────────────────────────────────────────────────────
    task_arg = DeclareLaunchArgument(
        "task",
        default_value="Deliver package_A to the delivery area and return home.",
        description="Natural language task description for the mission manager",
    )
    fallback_arg = DeclareLaunchArgument(
        "use_fallback",
        default_value="false",
        description="Skip the LLM and use a hard-coded PDDL scenario",
    )
    scenario_arg = DeclareLaunchArgument(
        "scenario",
        default_value="deliver_A",
        description="Fallback scenario: deliver_A | deliver_AB | recharge_then_deliver",
    )

    # ── Nav2 ──────────────────────────────────────────────────────────────────
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_dir, "launch", "bringup_launch.py")
        ),
        launch_arguments={
            "map":          os.path.join(myagv_dir, "maps", "lab_map.yaml"),
            "use_sim_time": "false",
            "params_file":  os.path.join(myagv_dir, "config", "nav2_params.yaml"),
        }.items(),
    )

    # ── Cobot interface ───────────────────────────────────────────────────────
    cobot = Node(
        package="myagv_lab",
        executable="cobot_interface",
        output="screen",
    )

    # ── Mission manager (delayed 12 s to let Nav2 fully initialise) ───────────
    mission = TimerAction(
        period=12.0,
        actions=[Node(
            package="myagv_lab",
            executable="mission_manager",
            output="screen",
            arguments=[
                "--task",     LaunchConfiguration("task"),
                "--scenario", LaunchConfiguration("scenario"),
            ],
            additional_env={
                "MYAGV_USE_SIM":   "0",
                "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
            },
        )],
    )

    return LaunchDescription([
        task_arg, fallback_arg, scenario_arg,
        nav2, cobot, mission,
    ])
