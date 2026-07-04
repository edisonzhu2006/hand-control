#!/usr/bin/env python3
"""Simple hand detection demo from webcam."""

import sys
import cv2

sys.path.insert(0, '/Users/edison.zhu/hand-control')
from src.hand_detection.detector import HandDetector


def main():
    detector = HandDetector(max_hands=2, confidence=0.5)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    frame_count = 0

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

            # Display hand info
            for hand in hands:
                handedness = hand['handedness']
                palm_center = hand['palm_center']
                pinch_dist = detector.get_pinch_distance(hand)

                # Draw palm center
                cv2.circle(frame, palm_center, 8, (0, 255, 0), -1)

                # Display text
                cv2.putText(
                    frame,
                    f"{handedness} Hand",
                    (10, 30 + hand['id'] * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Pinch: {pinch_dist:.0f}px",
                    (10, 55 + hand['id'] * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            # Display FPS
            cv2.putText(
                frame,
                f"Frame: {frame_count}",
                (frame_w - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Hand Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    main()
