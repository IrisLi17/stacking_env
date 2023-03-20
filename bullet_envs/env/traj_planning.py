#!/usr/bin/env python
# -*- coding: utf-8 -*-


from typing import Tuple, List, Dict, Optional
import numpy as np
import pinocchio
import pybullet as p
from bullet_envs.env.bullet_rotations import mat2quat

def _min_jerk_spaces(
    N: int, T: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a 1-dim minimum jerk trajectory from 0 to 1 in N steps & T seconds.
    Assumes zero velocity & acceleration at start & goal.
    The resulting trajectories can be scaled for different start & goals.

    Args:
        N: Length of resulting trajectory in steps
        T: Duration of resulting trajectory in seconds

    Returns:
        p_traj: Position trajectory of shape (N,)
        pd_traj: Velocity trajectory of shape (N,)
        pdd_traj: Acceleration trajectory of shape (N,)
    """
    assert N > 1, "Number of planning steps must be larger than 1."

    t_traj = np.linspace(0, 1, N)
    p_traj = 10 * t_traj**3 - 15 * t_traj**4 + 6 * t_traj**5
    pd_traj = (30 * t_traj**2 - 60 * t_traj**3 + 30 * t_traj**4) / T
    pdd_traj = (60 * t_traj - 180 * t_traj**2 + 120 * t_traj**3) / (T**2)

    return p_traj, pd_traj, pdd_traj


def _compute_num_steps(time_to_go: float, hz: float):
    return int(time_to_go * hz)


def generate_joint_space_min_jerk(
    start: np.ndarray, goal: np.ndarray, time_to_go: float, hz: float
) -> List[Dict]:
    """
    Primitive joint space minimum jerk trajectory planner.
    Assumes zero velocity & acceleration at start & goal.

    Args:
        start: Start joint position of shape (N,)
        goal: Goal joint position of shape (N,)
        time_to_go: Trajectory duration in seconds
        hz: Frequency of output trajectory

    Returns:
        waypoints: List of waypoints
    """
    steps = _compute_num_steps(time_to_go, hz)
    dt = 1.0 / hz

    p_traj, pd_traj, pdd_traj = _min_jerk_spaces(steps, time_to_go)

    D = goal - start
    q_traj = start[None, :] + D[None, :] * p_traj[:, None]
    qd_traj = D[None, :] * pd_traj[:, None]
    qdd_traj = D[None, :] * pdd_traj[:, None]

    waypoints = [
        {
            "time_from_start": i * dt,
            "position": q_traj[i, :],
            "velocity": qd_traj[i, :],
            "acceleration": qdd_traj[i, :],
        }
        for i in range(steps)
    ]

    return waypoints


def generate_cartesian_space_min_jerk(
    start: pinocchio.SE3,
    goal: pinocchio.SE3,
    time_to_go: float,
    hz: float,
) -> List[Dict]:
    """Initializes planner object and plans the trajectory

    Args:
        start: Start pose
        goal: Goal pose
        time_to_go: Trajectory duration in seconds
        hz: Frequency of output trajectory

    Returns:
        q_traj: Joint position trajectory
        qd_traj: Joint velocity trajectory
        qdd_traj: Joint acceleration trajectory
    """
    steps = _compute_num_steps(time_to_go, hz)
    dt = 1.0 / hz

    p_traj, pd_traj, pdd_traj = _min_jerk_spaces(steps, time_to_go)

    # Plan translation
    x_start = start.translation
    x_goal = goal.translation

    D = x_goal - x_start
    x_traj = x_start[None, :] + D[None, :] * p_traj[:, None]
    xd_traj = D[None, :] * pd_traj[:, None]
    xdd_traj = D[None, :] * pdd_traj[:, None]

    # Plan rotation
    r_start = start.rotation
    r_goal = goal.rotation
    r_delta = r_goal @ np.linalg.inv(r_start)
    r_axis, r_angle = p.getAxisAngleFromQuaternion(mat2quat(r_delta))
    rv_delta = r_angle * np.array(r_axis)
    # rv_delta = r_delta.as_rotvec()

    r_traj = np.empty([steps, 4])
    for i in range(steps):
        _rv = rv_delta * p_traj[i]
        if np.linalg.norm(_rv) < 1e-6:
            r = r_start
        else:
            _angle = np.linalg.norm(_rv)
            _axis = _rv / _angle
            _quat = np.concatenate([np.sin(_angle / 2) * _axis, [np.cos(_angle / 2)]])
            r = np.array(p.getMatrixFromQuaternion(_quat)).reshape((3, 3)) @ r_start
        r_traj[i, :] = mat2quat(r)
    rd_traj = rv_delta[None, :] * pd_traj[:, None]
    rdd_traj = rv_delta[None, :] * pdd_traj[:, None]

    # Combine results
    ee_twist_traj = np.concatenate([xd_traj, rd_traj], axis=-1)
    ee_accel_traj = np.concatenate([xdd_traj, rdd_traj], axis=-1)

    waypoints = [
        {
            "time_from_start": i * dt,
            "pose": pinocchio.SE3(
                np.array(p.getMatrixFromQuaternion(r_traj[i, :])).reshape((3, 3)), translation=x_traj[i, :]
            ),
            "twist": ee_twist_traj[i, :],
            "acceleration": ee_accel_traj[i, :],
        }
        for i in range(steps)
    ]

    return waypoints


def generate_position_min_jerk(start, goal, time_to_go: float, hz: float) -> List[Dict]:
    """
    Minimum jerk trajectory planner through XYZ space.
    Assumes zero velocity & acceleration at start & goal.
    Equivalent to a joint space planner with 3 joints.

    Args:
        start: start joint position of shape (3,)
        goal: goal joint position of shape (3,)
        time_to_go: Trajectory duration in seconds
        hz: Frequency of output trajectory

    Returns:
        q_traj: Position trajectory
        qd_traj: Velocity trajectory
        qdd_traj: Acceleration trajectory
    """
    assert start.shape == (3,)
    assert goal.shape == (3,)
    return generate_joint_space_min_jerk(start, goal, time_to_go, hz)


def generate_cartesian_target_joint_min_jerk(
    joint_pos_start: np.ndarray,
    ee_pose_goal: pinocchio.SE3,
    time_to_go: float,
    hz: float,
    robot_model,
    home_pose: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Cartesian space minimum jerk trajectory planner, but outputs plan in joint space.
    Assumes zero velocity & acceleration at start & goal.

    Args:
        start: Start pose
        goal: Goal pose
        time_to_go: Trajectory duration in seconds
        hz: Frequency of output trajectory
        robot_model: A valid robot model module from torchcontrol.models
        home_pose: Default pose of robot to stabilize around in null (elbow) space

    Returns:
        q_traj: Joint position trajectory
        qd_traj: Joint velocity trajectory
        qdd_traj: Joint acceleration trajectory
    """
    steps = _compute_num_steps(time_to_go, hz)
    dt = 1.0 / hz
    home_pose = np.zeros_like(joint_pos_start) if home_pose is None else home_pose

    # Compute start pose
    ee_pos_start, ee_quat_start = robot_model.forward_kinematics(joint_pos_start)
    ee_pose_start = pinocchio.SE3(
        rotation=np.array(p.getMatrixFromQuaternion(ee_quat_start)).reshape((3, 3)), translation=ee_pos_start
    )
    cartesian_waypoints = generate_cartesian_space_min_jerk(
        ee_pose_start, ee_pose_goal, time_to_go, hz
    )

    # Extract plan & convert to joint space
    q_traj = np.zeros((steps, joint_pos_start.shape[0]))
    qd_traj = np.zeros((steps, joint_pos_start.shape[0]))
    qdd_traj = np.zeros((steps, joint_pos_start.shape[0]))

    q_traj[0, :] = joint_pos_start
    for i in range(0, steps - 1):
        # Get current joint state & jacobian
        joint_pos_current = q_traj[i, :]
        jacobian = robot_model.compute_jacobian(joint_pos_current)
        jacobian_pinv = np.linalg.pinv(jacobian)

        # Query Cartesian plan for next step & compute diff
        ee_pose_desired = cartesian_waypoints[i + 1]["pose"]
        ee_twist_desired = cartesian_waypoints[i + 1]["twist"]
        ee_accel_desired = cartesian_waypoints[i + 1]["acceleration"]

        # Convert next step to joint plan
        qdd_traj[i + 1, :] = jacobian_pinv @ ee_accel_desired
        qd_traj[i + 1, :] = jacobian_pinv @ ee_twist_desired
        q_delta = qd_traj[i + 1, :] * dt
        q_traj[i + 1, :] = joint_pos_current + q_delta

        # Null space correction
        null_space_proj = np.eye(joint_pos_start.shape[0]) - jacobian_pinv @ jacobian
        q_null_err = null_space_proj @ (home_pose - q_traj[i + 1, :])
        q_null_err_norm = np.linalg.norm(q_null_err) + 1e-27  # prevent zero division
        q_null_err_clamped = (
            q_null_err / q_null_err_norm * min(q_null_err_norm, np.linalg.norm(q_delta))
        )  # norm of correction clamped to norm of current action
        q_traj[i + 1, :] = q_traj[i + 1, :] + q_null_err_clamped

    waypoints = [
        {
            "time_from_start": i * dt,
            "position": q_traj[i, :],
            "velocity": qd_traj[i, :],
            "acceleration": qdd_traj[i, :],
        }
        for i in range(steps)
    ]

    return waypoints
