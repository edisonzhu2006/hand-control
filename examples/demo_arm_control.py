#!/usr/bin/env python3
"""Demo: Control simulated robotic arm with hand gestures."""

import sys
import cv2
import numpy as np

sys.path.insert(0, '/Users/edison.zhu/hand-control')
from src.hand_detection.detector import HandDetector
from src.arm.kinematics import Kinematics
from src.arm.arm_controller import ArmController, ControlMode
from src.gestures.recognizer import GestureRecognizer


def draw_arm_pose(frame, joints, kinematics, frame_w, frame_h, color=(0, 255, 255)):
    """Draw arm configuration on frame.

    Args:
        frame: Video frame
        joints: Joint angles
        kinematics: Kinematics solver
        frame_w: Frame width
        frame_h: Frame height
        color: Drawing color
    """
    try:
        pose = kinematics.forward_kinematics(joints)
        end_effector = pose[:3, 3]

        # Scale arm coordinates to screen (simplified 2D projection)
        # Assume arm coordinates are in mm, scale to screen
        arm_x = int(frame_w / 2 + end_effector[0] / 1000 * 100)
        arm_y = int(frame_h / 2 - end_effector[2] / 1000 * 100)

        # Draw end effector
        cv2.circle(frame, (arm_x, arm_y), 8, color, -1)
        cv2.putText(
            frame,
            f"EE: ({end_effector[0]:.0f}, {end_effector[1]:.0f}, {end_effector[2]:.0f})",
            (arm_x - 100, arm_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        # Draw joint angles
        for i, angle in enumerate(joints[:3]):
            cv2.putText(
                frame,
                f"J{i+1}: {np.degrees(angle):.1f}°",
                (10, 100 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    except Exception as e:
        pass


def main():
    print("Arm Control Demo - Hand to Arm Mapping")
    print("=" * 50)

    # Initialize components
    detector = HandDetector(max_hands=1, confidence=0.6)
    recognizer = GestureRecognizer()

    # Load 3-DOF arm
    kinematics = Kinematics.from_config(
        '/Users/edison.zhu/hand-control/data/arm_configs/3dof_arm.json'
    )
    arm_controller = ArmController(kinematics, ControlMode.POSITION)
    arm_controller.enable_hand_tracking(hand_origin=np.array([0, 0, 500]))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    frame_count = 0
    tracking_enabled = False

    print("Controls:")
    print("  'e' - Enable/disable hand tracking")
    print("  'r' - Reset arm")
    print("  'q' - Quit")
    print("=" * 50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_h, frame_w, _ = frame.shape

            # Detect hands
            results, _, _, _ = detector.detect(frame)
            hands = detector.get_hand_landmarks(results, frame_h, frame_w)

            # Draw landmarks
            frame = detector.draw_landmarks(frame, results)

            # Process first hand
            if hands:
                hand = hands[0]

                # Map hand to arm if tracking enabled
                if tracking_enabled:
                    arm_controller.set_hand_target_position(hand, frame_w, frame_h)
                    arm_controller.step_simulation(dt=0.01)

                # Draw hand info
                palm_center = hand['palm_center']
                cv2.circle(frame, palm_center, 8, (0, 255, 0), -1)

                # Recognize gesture
                matches = recognizer.recognize(hand, top_k=1)
                if matches:
                    gesture = matches[0]
                    cv2.putText(
                        frame,
                        f"Gesture: {gesture.gesture_name} ({gesture.confidence:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

            # Draw arm state
            cv2.putText(
                frame,
                f"Tracking: {'ON' if tracking_enabled else 'OFF'}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if tracking_enabled else (0, 0, 255),
                2,
            )

            draw_arm_pose(frame, arm_controller.current_joints, kinematics, frame_w, frame_h)

            cv2.putText(
                frame,
                f"Frame: {frame_count}",
                (frame_w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Arm Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                tracking_enabled = not tracking_enabled
                print(f"Tracking: {'enabled' if tracking_enabled else 'disabled'}")
            elif key == ord('r'):
                arm_controller.current_joints = np.zeros(3)
                arm_controller.target_joints = np.zeros(3)
                print("Arm reset")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

        print("\n" + "=" * 50)
        print(f"Processed {frame_count} frames")
        print("Demo complete")


if __name__ == "__main__":
    main()
