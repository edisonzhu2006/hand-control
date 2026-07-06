"""
Robotic arm kinematics module with forward and inverse kinematics solvers.

Supports configurable arms with arbitrary DOF, joint limits, and link parameters.
Uses DH (Denavit-Hartenberg) parameters for arm definition.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import json
from pathlib import Path


@dataclass
class DHParameter:
    """Denavit-Hartenberg parameter for a single joint."""
    a: float  # Link length (mm)
    alpha: float  # Link twist (radians)
    d: float  # Link offset (mm)
    theta: float  # Joint angle (radians)
    joint_type: str = 'revolute'  # 'revolute' or 'prismatic'
    theta_min: float = -np.pi  # Min angle (revolute) or min extension (prismatic)
    theta_max: float = np.pi   # Max angle
    velocity_limit: float = 1.0  # rad/s for revolute, mm/s for prismatic


class Kinematics:
    """Forward and inverse kinematics solver for robotic arms."""

    def __init__(self, dh_params: List[DHParameter], base_transform: np.ndarray = None):
        """Initialize kinematics solver.

        Args:
            dh_params: List of DH parameters for each joint
            base_transform: Base frame transformation (4x4 matrix)
        """
        self.dh_params = dh_params
        self.num_joints = len(dh_params)
        self.base_transform = base_transform if base_transform is not None else np.eye(4)

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute end effector pose from joint angles.

        Args:
            joint_angles: Joint angles/displacements (rad or mm)

        Returns:
            transform: 4x4 transformation matrix (base to end effector)
        """
        if len(joint_angles) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joint angles, got {len(joint_angles)}")

        # Clamp to joint limits
        joint_angles = self._clamp_joints(joint_angles)

        # Compute cumulative transformation
        T = self.base_transform.copy()

        for i, (angle, dh) in enumerate(zip(joint_angles, self.dh_params)):
            # Update DH parameter with current joint value
            current_dh = DHParameter(
                a=dh.a,
                alpha=dh.alpha,
                d=dh.d if dh.joint_type == 'revolute' else angle,
                theta=angle if dh.joint_type == 'revolute' else dh.theta,
                joint_type=dh.joint_type,
                theta_min=dh.theta_min,
                theta_max=dh.theta_max,
            )

            # Compute transformation for this joint
            T_i = self._dh_transform(current_dh)
            T = T @ T_i

        return T

    def inverse_kinematics(self, target_pose: np.ndarray, initial_guess: Optional[np.ndarray] = None,
                          max_iterations: int = 100, tolerance: float = 1e-4) -> Tuple[np.ndarray, bool]:
        """Compute joint angles for target end effector pose.

        Uses iterative Jacobian-based method (damped least squares).

        Args:
            target_pose: 4x4 target transformation matrix
            initial_guess: Initial joint angle guess
            max_iterations: Maximum iterations
            tolerance: Position/orientation tolerance (mm/rad)

        Returns:
            joint_angles: Joint angles to reach target
            success: Whether solution converged
        """
        target_pos = target_pose[:3, 3]

        # Try the caller's guess plus a spread of offset/elbow seeds. Position-only
        # IK on a multi-link arm has folded local minima (the arm doubling back on
        # itself, which is what left the old solver stuck at the origin); restarting
        # from several bent poses reliably escapes them.
        alt = np.array([(-1.0) ** i for i in range(self.num_joints)])
        seeds = []
        if initial_guess is not None:
            seeds.append(np.asarray(initial_guess, dtype=float).copy())
        seeds += [
            np.zeros(self.num_joints),
            np.full(self.num_joints, 0.6),
            np.full(self.num_joints, -0.6),
            np.full(self.num_joints, 1.4),
            np.full(self.num_joints, -1.4),
            0.9 * alt,
            -0.9 * alt,
        ]

        best_angles = self._clamp_joints(seeds[0].copy())
        best_err = np.inf

        for seed in seeds:
            theta = self._clamp_joints(np.asarray(seed, dtype=float).copy())
            err_vec = target_pos - self.forward_kinematics(theta)[:3, 3]
            err = float(np.linalg.norm(err_vec))

            for _ in range(max_iterations):
                if err < best_err:
                    best_err = err
                    best_angles = theta.copy()
                if err < tolerance:
                    return theta, True

                # Position-only damped least squares. These arms don't constrain
                # end-effector orientation, so the old 6D error (with zero
                # orientation rows) was ill-posed; solving the 3D position system
                # directly is well-conditioned.
                J = self._compute_jacobian(theta)[:3, :]
                damping = 1.0  # mm-scale Levenberg-Marquardt term; stable near singularities
                JT = J.T
                delta_theta = JT @ np.linalg.solve(J @ JT + damping * np.eye(3), err_vec)

                # Backtracking line search: shrink the step until the error
                # actually drops, so each iteration is a guaranteed improvement
                # and the solver can't oscillate into a folded pose.
                step = 1.0
                improved = False
                for _ls in range(10):
                    candidate = self._clamp_joints(theta + step * delta_theta)
                    cand_err_vec = target_pos - self.forward_kinematics(candidate)[:3, 3]
                    cand_err = float(np.linalg.norm(cand_err_vec))
                    if cand_err < err:
                        theta, err_vec, err = candidate, cand_err_vec, cand_err
                        improved = True
                        break
                    step *= 0.5

                if not improved:
                    break  # local minimum for this seed; move on to the next

            if best_err < tolerance:
                return best_angles, True

        return best_angles, best_err < tolerance

    def jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix (end effector velocity w.r.t. joint velocities).

        Args:
            joint_angles: Current joint angles

        Returns:
            jacobian: 6xN Jacobian matrix
        """
        return self._compute_jacobian(joint_angles)

    def end_effector_position(self, joint_angles: np.ndarray) -> np.ndarray:
        """Get end effector position (xyz).

        Args:
            joint_angles: Joint angles

        Returns:
            position: 3D position vector
        """
        T = self.forward_kinematics(joint_angles)
        return T[:3, 3]

    def end_effector_orientation(self, joint_angles: np.ndarray) -> np.ndarray:
        """Get end effector orientation (3x3 rotation matrix).

        Args:
            joint_angles: Joint angles

        Returns:
            orientation: 3x3 rotation matrix
        """
        T = self.forward_kinematics(joint_angles)
        return T[:3, :3]

    def joint_positions(self, joint_angles: np.ndarray) -> np.ndarray:
        """Return Cartesian positions of the base and every joint / end effector.

        Accumulates the per-joint transforms so callers can render the arm as a
        connected chain of links.

        Args:
            joint_angles: Joint angles/displacements (rad or mm)

        Returns:
            points: (num_joints + 1, 3) array of XYZ positions, base first, end
                effector last.
        """
        angles = self._clamp_joints(np.asarray(joint_angles, dtype=float))
        T = self.base_transform.copy()
        points = [T[:3, 3].copy()]

        for angle, dh in zip(angles, self.dh_params):
            current_dh = DHParameter(
                a=dh.a,
                alpha=dh.alpha,
                d=dh.d if dh.joint_type == 'revolute' else angle,
                theta=angle if dh.joint_type == 'revolute' else dh.theta,
                joint_type=dh.joint_type,
            )
            T = T @ self._dh_transform(current_dh)
            points.append(T[:3, 3].copy())

        return np.array(points)

    def _dh_transform(self, dh: DHParameter) -> np.ndarray:
        """Compute 4x4 transformation matrix from DH parameters.

        Args:
            dh: DH parameter

        Returns:
            transform: 4x4 transformation matrix
        """
        c_theta = np.cos(dh.theta)
        s_theta = np.sin(dh.theta)
        c_alpha = np.cos(dh.alpha)
        s_alpha = np.sin(dh.alpha)

        T = np.array([
            [c_theta, -s_theta * c_alpha, s_theta * s_alpha, dh.a * c_theta],
            [s_theta, c_theta * c_alpha, -c_theta * s_alpha, dh.a * s_theta],
            [0, s_alpha, c_alpha, dh.d],
            [0, 0, 0, 1],
        ])

        return T

    def _compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute numerical Jacobian using finite differences.

        Args:
            joint_angles: Current joint angles

        Returns:
            jacobian: 6xN Jacobian matrix
        """
        delta = 1e-5
        jacobian = np.zeros((6, self.num_joints))

        # Current end effector pose
        T0 = self.forward_kinematics(joint_angles)
        p0 = T0[:3, 3]

        for i in range(self.num_joints):
            # Perturb joint i
            joint_angles_perturbed = joint_angles.copy()
            joint_angles_perturbed[i] += delta

            T_perturbed = self.forward_kinematics(joint_angles_perturbed)
            p_perturbed = T_perturbed[:3, 3]

            # Linear velocity Jacobian (position derivative)
            jacobian[:3, i] = (p_perturbed - p0) / delta

        return jacobian

    def _clamp_joints(self, joint_angles: np.ndarray) -> np.ndarray:
        """Clamp joint angles to limits.

        Args:
            joint_angles: Joint angles

        Returns:
            clamped_angles: Clamped joint angles
        """
        clamped = joint_angles.copy()

        for i in range(self.num_joints):
            dh = self.dh_params[i]
            clamped[i] = np.clip(clamped[i], dh.theta_min, dh.theta_max)

        return clamped

    def _rotation_error(self, R_desired: np.ndarray, R_current: np.ndarray) -> float:
        """Compute rotation error magnitude between two rotation matrices.

        Args:
            R_desired: Desired rotation matrix
            R_current: Current rotation matrix

        Returns:
            error: Error magnitude
        """
        # Use trace-based error metric
        trace = np.trace(R_desired.T @ R_current)
        angle_error = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        return float(angle_error)

    @classmethod
    def from_config(cls, config_path: str) -> 'Kinematics':
        """Load kinematics from JSON configuration file.

        Args:
            config_path: Path to config file

        Returns:
            kinematics: Kinematics instance
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        dh_params = [
            DHParameter(
                a=p['a'],
                alpha=p['alpha'],
                d=p['d'],
                theta=p['theta'],
                joint_type=p.get('joint_type', 'revolute'),
                theta_min=p.get('theta_min', -np.pi),
                theta_max=p.get('theta_max', np.pi),
            )
            for p in config['dh_parameters']
        ]

        base_T = np.array(config.get('base_transform', np.eye(4).tolist()))

        return cls(dh_params, base_T)

    def save_config(self, filepath: str):
        """Save arm configuration to JSON file.

        Args:
            filepath: Path to save
        """
        config = {
            'dh_parameters': [
                {
                    'a': float(dh.a),
                    'alpha': float(dh.alpha),
                    'd': float(dh.d),
                    'theta': float(dh.theta),
                    'joint_type': dh.joint_type,
                    'theta_min': float(dh.theta_min),
                    'theta_max': float(dh.theta_max),
                }
                for dh in self.dh_params
            ],
            'base_transform': self.base_transform.tolist(),
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
