from glob import glob
import os

from setuptools import find_packages, setup

package_name = "isaac_ros2_manager"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.json")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="CHIP-GT",
    maintainer_email="prakash.jamakatel@yahoo.com",
    description=(
        "Isaac Sim support utilities for the APE/Webots ROS contracts."
    ),
    license="Apache License 2.0",
    entry_points={
        "console_scripts": [
            "isaac_startup_ready_node = isaac_ros2_manager.startup_ready_node:main",
            "isaac_yolo_detection_adapter_node = isaac_ros2_manager.yolo_detection_adapter_node:main",
            "isaac_objective_state_node = isaac_ros2_manager.objective_state_node:main",
            "isaac_team_manager_node = isaac_ros2_manager.team_manager_node:main",
        ],
    },
)
