#!/usr/bin/env python3
"""Demonstration of gesture recognition system."""

import sys
import cv2
import numpy as np

sys.path.insert(0, '/Users/edison.zhu/hand-control')
from src.hand_detection.detector import HandDetector
from src.gestures.recognizer import GestureRecognizer


def main():
    detector = HandDetector(max_hands=2, confidence=0.6)
    recognizer = GestureRecognizer()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    frame_count = 0
    gesture_history = []

    print("Gesture Recognition Demo")
    print("=" * 50)
    print(f"Available gestures: {', '.join(recognizer.list_gestures())}")
    print("Press 'q' to quit")
    print("=" * 50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_h, frame_w, _ = frame.shape

            # Detect hands
            results, frame_h, frame_w, frame_c = detector.detect(frame)
            hands = detector.get_hand_landmarks(results, frame_h, frame_w)

            # Draw landmarks
            frame = detector.draw_landmarks(frame, results)

            # Recognize gestures
            for hand_idx, hand in enumerate(hands):
                # Recognize gesture
                matches = recognizer.recognize(hand, top_k=3)

                # Draw hand info
                palm_center = hand['palm_center']
                cv2.circle(frame, palm_center, 10, (0, 255, 0), -1)

                y_offset = 30 + hand_idx * 80

                # Handedness
                handedness = hand['handedness']
                cv2.putText(
                    frame,
                    f"{handedness} Hand",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Gesture matches
                if matches:
                    cv2.putText(
                        frame,
                        "Gestures:",
                        (10, y_offset + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        1,
                    )

                    for rank, match in enumerate(matches):
                        text = f"{rank+1}. {match.gesture_name} ({match.confidence:.2f})"
                        color = (0, 255, 0) if rank == 0 else (100, 255, 0)
                        cv2.putText(
                            frame,
                            text,
                            (20, y_offset + 45 + rank * 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                        )

                        # Record top gesture
                        if rank == 0 and match.confidence > 0.7:
                            gesture_history.append({
                                'frame': frame_count,
                                'gesture': match.gesture_name,
                                'confidence': match.confidence,
                            })

            # Display frame info
            cv2.putText(
                frame,
                f"Frame: {frame_count}",
                (frame_w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Gestures detected: {len(gesture_history)}",
                (frame_w - 200, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            cv2.imshow("Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

        print("\n" + "=" * 50)
        print(f"Processed {frame_count} frames")
        print(f"Total gestures recognized: {len(gesture_history)}")

        if gesture_history:
            print("\nGesture Summary:")
            gesture_counts = {}
            for entry in gesture_history:
                gesture = entry['gesture']
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

            for gesture, count in sorted(gesture_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {gesture}: {count}")


if __name__ == "__main__":
    main()
