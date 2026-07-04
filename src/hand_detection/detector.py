import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """Real-time hand detection and pose estimation using MediaPipe."""

    def __init__(self, max_hands=2, confidence=0.5):
        """Initialize hand detector.

        Args:
            max_hands: Maximum number of hands to detect (1 or 2)
            confidence: Confidence threshold for detections (0-1)
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.max_hands = max_hands

    def detect(self, frame):
        """Detect hands in frame.

        Args:
            frame: Input image (BGR format from OpenCV)

        Returns:
            results: MediaPipe hand detection results
            frame_h: Frame height
            frame_w: Frame width
            frame_c: Frame channels
        """
        frame_h, frame_w, frame_c = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results, frame_h, frame_w, frame_c

    def get_hand_landmarks(self, results, frame_h, frame_w):
        """Extract hand landmarks as normalized and pixel coordinates.

        Args:
            results: MediaPipe detection results
            frame_h: Frame height
            frame_w: Frame width

        Returns:
            hands: List of hand data dicts with landmarks and position
        """
        hands = []

        if not results.multi_hand_landmarks:
            return hands

        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_data = {
                'id': hand_idx,
                'landmarks': [],
                'handedness': results.multi_handedness[hand_idx].classification[0].label,
            }

            # Extract all 21 landmarks
            for landmark in hand_landmarks.landmark:
                x_norm = landmark.x
                y_norm = landmark.y
                z_norm = landmark.z

                x_pixel = int(x_norm * frame_w)
                y_pixel = int(y_norm * frame_h)

                hand_data['landmarks'].append({
                    'norm': (x_norm, y_norm, z_norm),
                    'pixel': (x_pixel, y_pixel),
                    'z': z_norm,
                })

            # Calculate palm center (average of wrist and hand center points)
            wrist = hand_data['landmarks'][0]['pixel']
            middle_mcp = hand_data['landmarks'][9]['pixel']
            palm_center = (
                (wrist[0] + middle_mcp[0]) // 2,
                (wrist[1] + middle_mcp[1]) // 2,
            )
            hand_data['palm_center'] = palm_center

            hands.append(hand_data)

        return hands

    def get_pinch_distance(self, hand_data):
        """Calculate distance between thumb and index finger (pinch gesture).

        Args:
            hand_data: Hand data dict from get_hand_landmarks

        Returns:
            distance: Euclidean distance in pixels
        """
        if len(hand_data['landmarks']) < 9:
            return 0

        thumb_tip = hand_data['landmarks'][4]['pixel']
        index_tip = hand_data['landmarks'][8]['pixel']

        distance = np.sqrt(
            (thumb_tip[0] - index_tip[0]) ** 2 +
            (thumb_tip[1] - index_tip[1]) ** 2
        )
        return distance

    def draw_landmarks(self, frame, results):
        """Draw hand landmarks and connections on frame.

        Args:
            frame: Input frame
            results: MediaPipe detection results

        Returns:
            frame: Frame with drawn landmarks
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )
        return frame

    def close(self):
        """Clean up resources."""
        self.hands.close()
