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
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="CHIP-GT",
    maintainer_email="prakash.jamakatel@yahoo.com",
    description=(
        "Isaac Sim counterpart to webots_ros2_manager: launches a Webots-style team "
        "into Isaac Sim and bridges the per-agent ROS control contracts."
    ),
    license="Apache License 2.0",
    entry_points={
        "console_scripts": [
            "isaac_agent_bridge_node = isaac_ros2_manager.agent_bridge_node:main",
            "isaac_env_manager_node = isaac_ros2_manager.env_manager_node:main",
            "isaac_team_manager_node = isaac_ros2_manager.team_manager_node:main",
        ],
    },
)
