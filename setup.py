from setuptools import setup, find_packages
import os
from glob import glob

package_name = "myagv_lab"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config",  glob("config/*.yaml")),
        (f"share/{package_name}/launch",  glob("launch/*.py")),
        (f"share/{package_name}/maps",    glob("maps/*")),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "imageio",
        "pyperplan",
        "openai",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            # Phase 1
            "slam_node         = myagv_lab.phase1_slam.slam_node:main",
            # Phase 2
            "nav_node          = myagv_lab.phase2_nav.nav_node:main",
            # Phase 3
            "mission_manager   = myagv_lab.phase3_delivery.mission_manager:main",
            "cobot_interface   = myagv_lab.phase3_delivery.cobot_interface:main",
        ],
    },
    zip_safe=True,
    author="UCL Robotics & AI",
    description="myAGV Summer School Lab — SLAM, Navigation, LLM-PDDL Delivery",
    license="MIT",
)
