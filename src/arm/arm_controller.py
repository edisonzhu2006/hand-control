"""
Robotic arm controller for mapping hand pose to arm movements.

Supports multiple control modes: position control, velocity control, and hybrid.
Includes trajectory generation, workspace management, and hardware abstraction.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any
from enum import Enum
import json
from pathlib import Path
import time

from .kinematics import Kinematics, DHParameter


class ControlMode(Enum):
    """Arm control modes."""
    POSITION = "position"        # Direct position control
    VELOCITY = "velocity"        # Velocity-based control
    HYBRID = "hybrid"           # Position + orientation control
    IMPEDANCE = "impedance"     # Force/compliance control


@dataclass
class WorkspaceConfig:
    """Workspace boundaries and constraints."""
    min_x: float = -500  # mm
    max_x: float = 500
    min_y: float = -500
    max_y: float = 500
    min_z: float = 0
    max_z: float = 1000
    max_joint_speed: List[float] = None  # Per-joint speed limits


class ArmController:
    """Control robotic arm with hand gestures."""

    def __init__(self, kinematics: Kinematics, control_mode: ControlMode = ControlMode.POSITION):
        """Initialize arm controller.

        Args:
            kinematics: Kinematics solver instance
            control_mode: Initial control mode
        """
        self.kinematics = kinematics
        self.control_mode = control_mode
        self.num_joints = kinematics.num_joints

        # Current state
        self.current_joints = np.zeros(self.num_joints)
        self.target_joints = np.zeros(self.num_joints)
        self.current_velocity = np.zeros(self.num_joints)

        # Hand to arm mapping
        self.hand_frame_origin = np.array([0, 0, 500])  # Default hand origin in arm space
        self.hand_frame_scale = 1.0  # Scale factor for hand movements
        self.hand_tracking_enabled = False

        # Workspace
        self.workspace = WorkspaceConfig()

        # Trajectory
        self.trajectory_generator = TrajectoryGenerator(max_accel=100)  # mm/s^2
        self.last_trajectory_time = 0

        # Safety limits
        self.collision_enabled = True
        self.joint_limits_enabled = True

        # Callbacks
        self.on_joint_target = None

    def set_control_mode(self, mode: ControlMode):
        """Set control mode.

        Args:
            mode: New control mode
        """
        self.control_mode = mode

    def enable_hand_tracking(self, hand_origin: np.ndarray = None):
        """Enable hand-to-arm mapping.

        Args:
            hand_origin: Origin of hand frame in arm space (mm)
        """
        self.hand_tracking_enabled = True
        if hand_origin is not None:
            self.hand_frame_origin = hand_origin

    def disable_hand_tracking(self):
        """Disable hand-to-arm mapping."""
        self.hand_tracking_enabled = False

    def map_hand_to_arm(self, hand_data: Dict, frame_w: int, frame_h: int) -> np.ndarray:
        """Map hand position to arm workspace.

        Args:
            hand_data: Hand data from detector
            frame_w: Camera frame width
            frame_h: Camera frame height

        Returns:
            target_position: 3D position in arm space (mm)
        """
        if 'palm_center' not in hand_data:
            return self.hand_frame_origin

        palm_x, palm_y = hand_data['palm_center']

        # Normalize to -1 to 1
        norm_x = (palm_x / frame_w) * 2 - 1
        norm_y = (palm_y / frame_h) * 2 - 1

        # Map to workspace
        workspace_x = self.hand_frame_origin[0] + norm_x * 200  # ±200mm from origin
        workspace_y = self.hand_frame_origin[1] + norm_y * 200
        workspace_z = self.hand_frame_origin[2]  # Keep z constant initially

        target = np.array([workspace_x, workspace_y, workspace_z])
        return self._clamp_to_workspace(target)

    def set_hand_target_position(self, hand_data: Dict, frame_w: int, frame_h: int):
        """Set arm target based on hand position.

        Args:
            hand_data: Hand data from detector
            frame_w: Camera frame width
            frame_h: Camera frame height
        """
        if not self.hand_tracking_enabled:
            return

        target_pos = self.map_hand_to_arm(hand_data, frame_w, frame_h)
        self.set_target_position(target_pos)

    def set_target_position(self, position: np.ndarray, orientation: Optional[np.ndarray] = None):
        """Set target end effector position.

        Args:
            position: Target 3D position (mm)
            orientation: Target 3x3 rotation matrix (optional)
        """
        # Clamp to workspace
        position = self._clamp_to_workspace(position)

        # Build target pose
        if orientation is None:
            orientation = np.eye(3)

        target_pose = np.eye(4)
        target_pose[:3, 3] = position
        target_pose[:3, :3] = orientation

        # Solve inverse kinematics
        initial_guess = self.current_joints
        target_joints, success = self.kinematics.inverse_kinematics(
            target_pose,
            initial_guess=initial_guess,
        )

        if success or np.linalg.norm(target_joints) > 0:  # Accept even if not fully converged
            self.target_joints = target_joints
            if self.on_joint_target:
                self.on_joint_target(target_joints)

    def set_target_joints(self, joint_angles: np.ndarray):
        """Set target joint angles directly.

        Args:
            joint_angles: Target joint angles (rad or mm)
        """
        self.target_joints = np.clip(joint_angles, -np.pi, np.pi)

    def get_current_position(self) -> np.ndarray:
        """Get current end effector position.

        Returns:
            position: 3D position (mm)
        """
        return self.kinematics.end_effector_position(self.current_joints)

    def get_current_orientation(self) -> np.ndarray:
        """Get current end effector orientation.

        Returns:
            orientation: 3x3 rotation matrix
        """
        return self.kinematics.end_effector_orientation(self.current_joints)

    def get_current_pose(self) -> np.ndarray:
        """Get current end effector pose.

        Returns:
            pose: 4x4 transformation matrix
        """
        return self.kinematics.forward_kinematics(self.current_joints)

    def step_simulation(self, dt: float = 0.01):
        """Simulate arm motion toward target.

        Args:
            dt: Time step (seconds)
        """
        # Simple linear interpolation toward target
        alpha = 0.1  # Blending factor
        self.current_joints = (
            (1 - alpha) * self.current_joints +
            alpha * self.target_joints
        )

    def get_workspace_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get workspace boundaries.

        Returns:
            bounds: Dictionary of min/max for each axis
        """
        return {
            'x': (self.workspace.min_x, self.workspace.max_x),
            'y': (self.workspace.min_y, self.workspace.max_y),
            'z': (self.workspace.min_z, self.workspace.max_z),
        }

    def set_workspace(self, bounds: Dict[str, Tuple[float, float]]):
        """Set workspace boundaries.

        Args:
            bounds: Dictionary with 'x', 'y', 'z' keys and (min, max) tuples
        """
        if 'x' in bounds:
            self.workspace.min_x, self.workspace.max_x = bounds['x']
        if 'y' in bounds:
            self.workspace.min_y, self.workspace.max_y = bounds['y']
        if 'z' in bounds:
            self.workspace.min_z, self.workspace.max_z = bounds['z']

    def _clamp_to_workspace(self, position: np.ndarray) -> np.ndarray:
        """Clamp position to workspace bounds.

        Args:
            position: 3D position

        Returns:
            clamped_position: Clamped position
        """
        clamped = position.copy()
        clamped[0] = np.clip(clamped[0], self.workspace.min_x, self.workspace.max_x)
        clamped[1] = np.clip(clamped[1], self.workspace.min_y, self.workspace.max_y)
        clamped[2] = np.clip(clamped[2], self.workspace.min_z, self.workspace.max_z)
        return clamped

    def save_state(self, filepath: str):
        """Save arm state to file.

        Args:
            filepath: Path to save
        """
        state = {
            'current_joints': self.current_joints.tolist(),
            'target_joints': self.target_joints.tolist(),
            'control_mode': self.control_mode.value,
            'hand_frame_origin': self.hand_frame_origin.tolist(),
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load arm state from file.

        Args:
            filepath: Path to load
        """
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.current_joints = np.array(state['current_joints'])
        self.target_joints = np.array(state['target_joints'])
        self.control_mode = ControlMode(state['control_mode'])
        self.hand_frame_origin = np.array(state['hand_frame_origin'])


class TrajectoryGenerator:
    """Generate smooth trajectories for arm motion."""

    def __init__(self, max_accel: float = 100):
        """Initialize trajectory generator.

        Args:
            max_accel: Maximum acceleration (mm/s^2)
        """
        self.max_accel = max_accel
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)

    def update(self, target_position: np.ndarray, dt: float) -> np.ndarray:
        """Compute next position in trajectory.

        Args:
            target_position: Target position
            dt: Time step

        Returns:
            next_position: Next position in trajectory
        """
        error = target_position - self.current_position
        distance = np.linalg.norm(error)

        if distance < 1:  # Close enough
            self.current_position = target_position
            self.current_velocity = np.zeros(3)
            return self.current_position

        # Direction to target
        direction = error / distance if distance > 0 else np.zeros(3)

        # Update velocity with acceleration limit
        max_velocity = 500  # mm/s
        target_velocity = direction * max_velocity
        velocity_error = target_velocity - self.current_velocity

        accel = np.clip(
            velocity_error / dt,
            -self.max_accel,
            self.max_accel,
        )

        self.current_velocity += accel * dt
        self.current_position += self.current_velocity * dt

        return self.current_position
