# SPDX-License-Identifier: Apache-2.0
"""cuMotion motion-planning backend for the single-object end2end demos.

`NVIDIA cuMotion <https://github.com/nvidia-isaac/cumotion>`_ is the
production motion-generation library behind Isaac ROS cuMotion; it carries
hardened versions of cuRobo's collision-aware IK and trajectory optimization.
This module lets ``e2e_grasp_demo.py --motion_backend cumotion`` plan the same
pick as the cuRobo path, reusing everything else (GraspGenX, scene, Newton
playback):

* The robot is described with the cuRobo YAML the demo already generates
  (collision spheres, self-collision ignores, locked gripper joint), converted
  on the fly to cuMotion's XRDF by :func:`xrdf_from_curobo_robot_cfg`.
* The scene's cuboid obstacles become cuMotion cuboids.
* :func:`plan_grasp_cumotion` mirrors cuRobo's ``MotionPlanner.plan_grasp``:
  a goalset plan to the pre-grasp poses of all candidate grasps (cuMotion picks
  the most reachable), then a straight-line approach along the tool +Z to the
  grasp, then a straight-line lift back along the tool Z.

Install: ``python end2end/setup_end2end_deps.py --cumotion`` (downloads the
release wheel; cuMotion is not on PyPI).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

log = logging.getLogger("e2e")

# Playback runs one trajectory waypoint per sim frame at 60 fps, so sample
# cuMotion's continuous-time trajectories at the same rate.
SAMPLE_DT = 1.0 / 60.0
# Max deviation of the tool origin from the straight approach / lift line.
LINEAR_PATH_DEVIATION_M = 0.002


def xrdf_from_curobo_robot_cfg(robot_dict: Dict[str, Any]) -> str:
    """Convert a cuRobo robot YAML (``{"robot_cfg": {"kinematics": ...}}``) to XRDF.

    Locked joints (e.g. the gripper's ``finger_joint``) are left out of the
    c-space and pinned via ``default_joint_positions``, which is how XRDF
    fixes non-planned joints.
    """
    kin = robot_dict["robot_cfg"]["kinematics"]
    cspace = kin["cspace"]
    locked = kin.get("lock_joints") or {}
    defaults = dict(zip(cspace["joint_names"], cspace["default_joint_position"]))
    defaults.update(locked)
    joints = [j for j in cspace["joint_names"] if j not in locked]

    def _limit(key: str, fallback: float) -> List[float]:
        v = cspace.get(key, fallback)
        return [float(v)] * len(joints) if np.isscalar(v) else [float(x) for x in v]

    spheres = {
        link: [
            {"center": [float(c) for c in s["center"]], "radius": float(s["radius"])}
            for s in sph
        ]
        for link, sph in kin["collision_spheres"].items()
    }
    self_buffer = {
        k: float(v) for k, v in (kin.get("self_collision_buffer") or {}).items() if v
    }
    self_collision: Dict[str, Any] = {
        "geometry": "collision_spheres",
        "ignore": kin.get("self_collision_ignore") or {},
    }
    if self_buffer:
        self_collision["buffer_distance"] = self_buffer
    xrdf = {
        "format": "xrdf",
        "format_version": 2.0,
        "default_joint_positions": {k: float(v) for k, v in defaults.items()},
        "cspace": {
            "joint_names": joints,
            "acceleration_limits": _limit("max_acceleration", 12.0),
            "jerk_limits": _limit("max_jerk", 500.0),
        },
        "tool_frames": list(kin.get("tool_frames") or [kin.get("ee_link")]),
        "modifiers": [{"set_base_frame": kin["base_link"]}],
        "world_collision": {"geometry": "collision_spheres"},
        "self_collision": self_collision,
        "geometry": {"collision_spheres": {"spheres": spheres}},
    }
    return yaml.safe_dump(xrdf, sort_keys=False)


@dataclass
class CuMotionGraspResult:
    """What ``plan_to_grasp`` needs from a plan: chosen candidate + segments."""

    success: bool
    status: str
    goalset_index: int = -1
    segments: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)


class CuMotionGraspPlanner:
    """cuMotion trajectory optimizer for one robot + a static cuboid world."""

    def __init__(
        self,
        robot_dict: Dict[str, Any],
        urdf_path: str,
        tool_frame: str,
        cuboids: Dict[str, Dict[str, Any]],
    ):
        import cumotion

        self._cm = cumotion
        self.robot = cumotion.load_robot_from_memory(
            xrdf_from_curobo_robot_cfg(robot_dict), open(urdf_path).read()
        )
        self.joint_names = [
            self.robot.cspace_coord_name(i)
            for i in range(self.robot.num_cspace_coords())
        ]
        self.world = cumotion.create_world()
        for name, cub in cuboids.items():
            ob = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
            ob.set_attribute(
                cumotion.Obstacle.Attribute.SIDE_LENGTHS,
                np.asarray(cub["dims"], dtype=np.float64),
            )
            x, y, z, qw, qx, qy, qz = cub["pose"]
            rot = cumotion.Rotation3(qw, qx, qy, qz)
            self.world.add_obstacle(ob, cumotion.Pose3(rot, np.array([x, y, z])))
            log.info("cuMotion obstacle %s dims=%s", name, cub["dims"])
        self.world_view = self.world.add_world_view()
        cfg = cumotion.create_default_trajectory_optimizer_config(
            self.robot, tool_frame, self.world_view
        )
        self.optimizer = cumotion.create_trajectory_optimizer(cfg)
        # Approach and lift are exact straight lines (like a UR ``moveL``):
        # task-space path -> c-space waypoints -> time-optimal trajectory,
        # then every sample is collision-checked against the world. The
        # trajectory optimizer's linear-path constraint is a soft penalty and
        # drifts 1-2 cm sideways on a 20 cm move, enough to push the object.
        self.tool_frame = tool_frame
        self.kinematics = self.robot.kinematics()
        self.trajectory_generator = cumotion.create_cspace_trajectory_generator(
            self.kinematics
        )
        self.inspector = cumotion.create_robot_world_inspector(
            self.robot, self.world_view
        )
        # Like cuRobo's plan_grasp, only consider grasps the arm can reach
        # without the gripper or arm intersecting the scene at the grasp pose.
        self.ik_solver = cumotion.create_collision_free_ik_solver(
            cumotion.create_default_collision_free_ik_solver_config(
                self.robot, tool_frame, self.world_view
            )
        )

    # -- helpers -----------------------------------------------------------

    def _rot(self, T: np.ndarray):
        return self._cm.Rotation3.from_matrix(np.asarray(T[:3, :3], dtype=np.float64))

    @staticmethod
    def _sample(traj) -> np.ndarray:
        dom = traj.domain()
        ts = np.append(np.arange(dom.lower, dom.upper, SAMPLE_DT), dom.upper)
        return np.stack(
            [np.asarray(traj.eval(float(t))).reshape(-1) for t in ts]
        ).astype(np.float32)

    def _linear(self, q0: np.ndarray, T_goal: np.ndarray):
        """Straight-line tool motion from FK(q0) to ``T_goal``.

        Returns ``(samples, reason)``; ``samples`` is None on failure.
        """
        cm = self._cm
        spec = cm.create_task_space_path_spec(self.kinematics.pose(q0, self.tool_frame))
        spec.add_linear_path(cm.Pose3(np.asarray(T_goal, dtype=np.float64)))
        conversion = cm.TaskSpacePathConversionConfig()
        conversion.max_position_deviation = LINEAR_PATH_DEVIATION_M
        conversion.min_position_deviation = LINEAR_PATH_DEVIATION_M / 2
        ik = cm.IkConfig()
        ik.cspace_seeds = [q0]
        try:
            path = cm.convert_task_space_path_spec_to_cspace(
                spec, self.kinematics, self.tool_frame, conversion, ik
            )
            traj = self.trajectory_generator.generate_trajectory(path.waypoints())
        except Exception as e:  # IK or time-parameterization failure
            return None, f"straight-line conversion failed ({e})"
        if traj is None:
            return (
                None,
                "no time-optimal trajectory through the straight-line waypoints",
            )
        samples = self._sample(traj)
        for q in samples:
            qd = q.astype(np.float64)
            if self.inspector.in_collision_with_obstacle(qd):
                return None, "straight line collides with the scene"
            if self.inspector.in_self_collision(qd):
                return None, "straight line is in self-collision"
        return samples, ""

    def collision_free_grasps(self, grasps_base: Sequence[np.ndarray]) -> List[int]:
        """Indices of grasps with a collision-free IK solution at the grasp pose."""
        S = self._cm.CollisionFreeIkSolver
        targets = S.TaskSpaceTargetArray(
            S.TranslationConstraintArray.target(
                [[np.asarray(T[:3, 3], dtype=np.float64)] for T in grasps_base]
            ),
            S.OrientationConstraintArray.target([[self._rot(T)] for T in grasps_base]),
        )
        res = self.ik_solver.solve_array(targets)
        return [
            i
            for i in range(len(grasps_base))
            if res.problem(i).status() == S.Results.Status.SUCCESS
        ]

    # -- main entry --------------------------------------------------------

    def plan_grasp(
        self,
        grasps_base: Sequence[np.ndarray],
        q_start: Sequence[float],
        approach_offset: float,
        lift_offset: float,
        plan_approach_to_grasp: bool = True,
        plan_grasp_to_lift: bool = True,
        max_goalset_retries: int = 5,
        candidates: Optional[Sequence[int]] = None,
    ) -> CuMotionGraspResult:
        """Plan pre-grasp (goalset) → linear approach → linear lift.

        ``grasps_base`` are tool-frame goal poses in the robot base frame.
        Offsets follow cuRobo's ``plan_grasp`` convention: translations along
        the tool Z axis, negative = backed away from the object. Only grasps in
        ``candidates`` (default: :meth:`collision_free_grasps`) are tried. If
        the approach or lift for the chosen grasp is infeasible, that grasp is
        dropped and the goalset is re-planned (up to ``max_goalset_retries``).
        """
        TO = self._cm.TrajectoryOptimizer
        Status = TO.Results.Status
        q0 = np.asarray(q_start, dtype=np.float64)

        def _offset(T: np.ndarray, dz: float) -> np.ndarray:
            off = np.eye(4)
            off[2, 3] = dz
            return T @ off

        if candidates is None:
            candidates = self.collision_free_grasps(grasps_base)
        remaining = list(candidates)
        status = "no grasp has a collision-free IK solution"
        for _ in range(max_goalset_retries):
            if not remaining:
                break
            pre = [_offset(grasps_base[i], approach_offset) for i in remaining]
            goal = TO.TaskSpaceTargetGoalset(
                TO.TranslationConstraintGoalset.target(
                    [np.asarray(T[:3, 3], dtype=np.float64) for T in pre]
                ),
                TO.OrientationConstraintGoalset.terminal_target(
                    [self._rot(T) for T in pre]
                ),
            )
            res = self.optimizer.plan_to_task_space_goalset(q0, goal)
            if res.status() != Status.SUCCESS:
                return CuMotionGraspResult(False, f"pre-grasp goalset: {res.status()}")
            k = int(res.target_index())
            gi = remaining[k]
            approach = self._sample(res.trajectory())
            segments: Dict[str, Optional[np.ndarray]] = {
                "approach": approach,
                "grasp": None,
                "lift": None,
            }
            if not plan_approach_to_grasp:
                return CuMotionGraspResult(True, "approach only", gi, segments)
            segments["grasp"], why = self._linear(
                approach[-1].astype(np.float64), grasps_base[gi]
            )
            if segments["grasp"] is None:
                status = f"approach to candidate {gi}: {why}"
                log.info("cuMotion: %s; dropping candidate and re-planning", status)
                remaining.pop(k)
                continue
            if not plan_grasp_to_lift:
                return CuMotionGraspResult(True, "approach+grasp", gi, segments)
            segments["lift"], why = self._linear(
                segments["grasp"][-1].astype(np.float64),
                _offset(grasps_base[gi], lift_offset),
            )
            if segments["lift"] is None:
                status = f"lift from candidate {gi}: {why}"
                log.info("cuMotion: %s; dropping candidate and re-planning", status)
                remaining.pop(k)
                continue
            return CuMotionGraspResult(
                True, "planned approach, grasp and lift", gi, segments
            )
        return CuMotionGraspResult(False, status)
