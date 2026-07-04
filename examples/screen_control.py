#!/usr/bin/env python3
"""Screen control demo - move mouse with hand position."""

import sys
import cv2
import time

sys.path.insert(0, '/Users/edison.zhu/hand-control')
from src.hand_detection.detector import HandDetector
from src.control.screen_control import ScreenController


def main():
    detector = HandDetector(max_hands=1, confidence=0.6)
    controller = ScreenController(smoothing=0.7)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Get frame dimensions
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    last_click_time = 0
    click_threshold = 0.5  # Seconds

    print("Starting screen control demo...")
    print("Press 'q' to quit")
    print(f"Screen: {controller.screen_w}x{controller.screen_h}")
    print(f"Webcam: {frame_w}x{frame_h}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detect hands
            results, frame_h, frame_w, frame_c = detector.detect(frame)
            hands = detector.get_hand_landmarks(results, frame_h, frame_w)

            # Draw landmarks
            frame = detector.draw_landmarks(frame, results)

            # Control mouse with right hand
            for hand in hands:
                if hand['handedness'] == 'Right':
                    # Move mouse
                    controller.move_mouse(hand, frame_w, frame_h, smooth=True, invert_x=True)

                    # Check for pinch gesture to click
                    pinch_dist = detector.get_pinch_distance(hand)
                    current_time = time.time()

                    # Pinch distance < 30px triggers click
                    if pinch_dist < 30 and (current_time - last_click_time) > click_threshold:
                        controller.click('left')
                        last_click_time = current_time
                        cv2.putText(
                            frame,
                            "CLICK!",
                            (frame_w // 2 - 50, frame_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 0, 255),
                            3,
                        )

                    # Draw status
                    palm_center = hand['palm_center']
                    cv2.circle(frame, palm_center, 8, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        f"Pinch: {pinch_dist:.0f}px",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            # Display frame info
            cv2.putText(
                frame,
                f"Frame: {frame_count}",
                (frame_w - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Screen Control", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    main()
