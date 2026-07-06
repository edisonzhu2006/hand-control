"""
Tests for robotic arm kinematics.
"""

import sys
import numpy as np

sys.path.insert(0, '/Users/edison.zhu/hand-control')

from src.arm.kinematics import Kinematics, DHParameter


def test_forward_kinematics_3dof():
    """Test forward kinematics for 3-DOF arm."""
    dh_params = [
        DHParameter(a=200, alpha=0, d=0, theta=0),
        DHParameter(a=200, alpha=0, d=0, theta=0),
        DHParameter(a=100, alpha=0, d=0, theta=0),
    ]

    kinematics = Kinematics(dh_params)

    # All joints zero
    joints = np.array([0, 0, 0])
    pose = kinematics.forward_kinematics(joints)

    # End effector should be at (500, 0, 0) in base frame
    expected_x = 200 + 200 + 100  # Sum of link lengths
    assert np.isclose(pose[0, 3], expected_x, atol=1.0), f"Expected x={expected_x}, got {pose[0, 3]}"
    assert np.isclose(pose[1, 3], 0, atol=1.0), f"Expected y=0, got {pose[1, 3]}"
    assert np.isclose(pose[2, 3], 0, atol=1.0), f"Expected z=0, got {pose[2, 3]}"

    print("✓ Forward kinematics (all zeros) passed")


def test_forward_kinematics_rotation():
    """Test forward kinematics with rotation."""
    dh_params = [
        DHParameter(a=100, alpha=0, d=0, theta=0),
        DHParameter(a=100, alpha=0, d=0, theta=0),
    ]

    kinematics = Kinematics(dh_params)

    # 90 degree rotation at first joint
    joints = np.array([np.pi / 2, 0])
    pose = kinematics.forward_kinematics(joints)

    # End effector should be at (0, 100, 0) then (0, 200, 0)
    assert np.isclose(pose[0, 3], 0, atol=1.0), f"Expected x=0, got {pose[0, 3]}"
    assert np.isclose(pose[1, 3], 200, atol=1.0), f"Expected y=200, got {pose[1, 3]}"

    print("✓ Forward kinematics (with rotation) passed")


def test_inverse_kinematics_reachability():
    """Test inverse kinematics convergence."""
    dh_params = [
        DHParameter(a=200, alpha=0, d=0, theta=0),
        DHParameter(a=200, alpha=0, d=0, theta=0),
    ]

    kinematics = Kinematics(dh_params)

    # Target position reachable by arm
    target_x, target_y = 300, 100
    target_pose = np.eye(4)
    target_pose[0, 3] = target_x
    target_pose[1, 3] = target_y

    joints, success = kinematics.inverse_kinematics(target_pose, max_iterations=100)

    # Verify solution
    pose = kinematics.forward_kinematics(joints)
    error = np.linalg.norm(pose[:2, 3] - np.array([target_x, target_y]))

    assert error < 10, f"IK error too large: {error}"
    print(f"✓ Inverse kinematics passed (error: {error:.2f}mm)")


def test_jacobian_computation():
    """Test Jacobian computation."""
    dh_params = [
        DHParameter(a=100, alpha=0, d=0, theta=0),
        DHParameter(a=100, alpha=0, d=0, theta=0),
    ]

    kinematics = Kinematics(dh_params)

    joints = np.array([0, 0])
    jacobian = kinematics.jacobian(joints)

    # Jacobian should be 6x2
    assert jacobian.shape == (6, 2), f"Expected shape (6, 2), got {jacobian.shape}"

    # Check rank is appropriate
    rank = np.linalg.matrix_rank(jacobian)
    assert rank == 2, f"Expected rank 2, got {rank}"

    print("✓ Jacobian computation passed")


def test_joint_limits():
    """Test joint limit enforcement."""
    dh_params = [
        DHParameter(
            a=100, alpha=0, d=0, theta=0,
            theta_min=-np.pi/2,
            theta_max=np.pi/2
        ),
    ]

    kinematics = Kinematics(dh_params)

    # Request angle beyond limits
    joints = np.array([2.0])  # Beyond pi/2
    clamped = kinematics._clamp_joints(joints)

    assert clamped[0] <= np.pi/2, "Joint exceeds max limit"
    print("✓ Joint limits passed")


def test_dh_parameter_loading():
    """Test loading DH parameters from config."""
    config_path = '/Users/edison.zhu/hand-control/data/arm_configs/3dof_arm.json'

    kinematics = Kinematics.from_config(config_path)

    assert kinematics.num_joints == 3, f"Expected 3 joints, got {kinematics.num_joints}"
    print("✓ DH parameter loading passed")


if __name__ == "__main__":
    print("Running Kinematics Tests")
    print("=" * 50)

    try:
        test_forward_kinematics_3dof()
        test_forward_kinematics_rotation()
        test_inverse_kinematics_reachability()
        test_jacobian_computation()
        test_joint_limits()
        test_dh_parameter_loading()

        print("=" * 50)
        print("All tests passed! ✓")

    except AssertionError as e:
        print(f"Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
