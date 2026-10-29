"""Tests for the cuMotion motion-planning backend (end2end/cumotion_backend.py).

The XRDF conversion test is pure Python. The planning test needs the cuMotion
wheel (``python end2end/setup_end2end_deps.py --cumotion``), a GPU and the
generated UR10e + Robotiq 2F-140 assets, and skips otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
E2E = REPO_ROOT / "end2end"
sys.path.insert(0, str(E2E))

from cumotion_backend import xrdf_from_curobo_robot_cfg  # noqa: E402

ROBOTIQ_YML = E2E / "curobo_assets" / "ur10e_robotiq_2f_140.yml"
ROBOTIQ_URDF = E2E / "curobo_assets" / "ur10e_robotiq_2f_140.urdf"
# Same table as end2end/envs/tabletop_single_nobin.yaml, in the robot base frame.
TABLE = {"table": {"dims": [0.8, 0.8, 0.5], "pose": [0.55, 0.0, -0.05, 1, 0, 0, 0]}}
Q_START = [0.0, -2.2, 1.9, -1.383, -1.57, 0.0]


def _curobo_cfg():
    return {
        "robot_cfg": {
            "kinematics": {
                "base_link": "base_link",
                "tool_frames": ["tool0"],
                "collision_spheres": {
                    "link_a": [{"center": [0, 0, 0.1], "radius": 0.05}],
                    "finger": [{"center": [0, 0.01, 0], "radius": 0.01}],
                },
                "self_collision_ignore": {"link_a": ["finger"]},
                "self_collision_buffer": {"link_a": 0.07, "finger": 0},
                "cspace": {
                    "joint_names": ["j1", "j2", "finger_joint"],
                    "default_joint_position": [0.1, -0.2, 0.0],
                    "max_acceleration": 12.0,
                    "max_jerk": 500.0,
                },
                "lock_joints": {"finger_joint": 0.3},
            }
        }
    }


def test_xrdf_locks_gripper_joint_and_keeps_spheres():
    xrdf = yaml.safe_load(xrdf_from_curobo_robot_cfg(_curobo_cfg()))
    assert xrdf["format"] == "xrdf"
    # Locked joint is out of the planned c-space but pinned to its lock value.
    assert xrdf["cspace"]["joint_names"] == ["j1", "j2"]
    assert xrdf["cspace"]["acceleration_limits"] == [12.0, 12.0]
    assert xrdf["default_joint_positions"] == {
        "j1": 0.1,
        "j2": -0.2,
        "finger_joint": 0.3,
    }
    assert xrdf["tool_frames"] == ["tool0"]
    assert xrdf["modifiers"] == [{"set_base_frame": "base_link"}]
    spheres = xrdf["geometry"]["collision_spheres"]["spheres"]
    assert spheres["link_a"] == [{"center": [0.0, 0.0, 0.1], "radius": 0.05}]
    assert xrdf["self_collision"]["ignore"] == {"link_a": ["finger"]}
    # Zero buffers are dropped; non-zero ones carried over.
    assert xrdf["self_collision"]["buffer_distance"] == {"link_a": 0.07}


def _cumotion_planner():
    cumotion = pytest.importorskip("cumotion")
    if not (ROBOTIQ_YML.is_file() and ROBOTIQ_URDF.is_file()):
        pytest.skip("UR10e+Robotiq assets missing — run end2end/setup_end2end_deps.py")
    from cumotion_backend import CuMotionGraspPlanner

    planner = CuMotionGraspPlanner(
        yaml.safe_load(ROBOTIQ_YML.read_text()), str(ROBOTIQ_URDF), "tool0", TABLE
    )
    return cumotion, planner


def test_cumotion_plans_straight_approach_and_lift():
    _, planner = _cumotion_planner()
    assert (
        planner.joint_names[0] == "shoulder_pan_joint" and len(planner.joint_names) == 6
    )
    # Top-down grasp 25 cm above the table top (tool +Z pointing down).
    grasp = np.eye(4)
    grasp[:3, :3] = np.diag([1.0, -1.0, -1.0])
    grasp[:3, 3] = [0.6, 0.0, 0.45]
    # An unreachable candidate first: the goalset must pick the reachable one.
    far = grasp.copy()
    far[:3, 3] = [3.0, 0.0, 0.45]
    res = planner.plan_grasp(
        [far, grasp], Q_START, approach_offset=-0.15, lift_offset=-0.20
    )
    assert res.success, res.status
    assert res.goalset_index == 1
    kin = planner.robot.kinematics()

    def tool_positions(seg):
        return np.array([kin.position(q.astype(float), "tool0") for q in seg])

    approach = tool_positions(res.segments["grasp"])
    lift = tool_positions(res.segments["lift"])
    np.testing.assert_allclose(approach[0], [0.6, 0.0, 0.60], atol=2e-3)  # pre-grasp
    np.testing.assert_allclose(approach[-1], [0.6, 0.0, 0.45], atol=1e-3)  # grasp
    np.testing.assert_allclose(lift[-1], [0.6, 0.0, 0.65], atol=1e-3)  # lifted 20 cm
    # Approach and lift are straight vertical lines (within 2 mm sideways).
    for seg in (approach, lift):
        assert np.linalg.norm(seg[:, :2] - [0.6, 0.0], axis=1).max() < 2e-3


def test_cumotion_rejects_grasp_inside_table():
    _, planner = _cumotion_planner()
    grasp = np.eye(4)
    grasp[:3, :3] = np.diag([1.0, -1.0, -1.0])
    grasp[:3, 3] = [0.6, 0.0, 0.25]  # fingers would end ~15 cm into the table
    res = planner.plan_grasp([grasp], Q_START, approach_offset=-0.15, lift_offset=-0.20)
    assert not res.success
