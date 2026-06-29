#!/usr/bin/env bash
# Run tests with ROS2 paths stripped from PYTHONPATH so its pytest
# plugins don't crash before pytest.ini is read.
PYTHONPATH="" python3 -m pytest "$@"
