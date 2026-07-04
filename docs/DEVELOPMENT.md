# Development Guide

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running Examples

### Hand Detection Demo
```bash
python examples/detect_hand.py
```
Shows real-time hand pose estimation with landmark visualization.

**Output:**
- Skeleton overlay with 21 hand landmarks
- Handedness detection (Left/Right)
- Pinch distance measurement

### Screen Control Demo
```bash
python examples/screen_control.py
```
Move your mouse with your right hand. Pinch your thumb and index finger to click.

**Controls:**
- Hand position → Mouse position
- Pinch gesture (thumb + index < 30px) → Left click
- Press 'q' to quit

## Architecture

### Hand Detection Module (`src/hand_detection/`)
- **detector.py**: HandDetector class using MediaPipe Hands
  - `detect()`: Run inference on a frame
  - `get_hand_landmarks()`: Extract normalized and pixel coordinates
  - `get_pinch_distance()`: Measure thumb-to-index distance

### Screen Control Module (`src/control/`)
- **screen_control.py**: ScreenController class for mouse control
  - `hand_to_screen()`: Convert hand position to screen coordinates
  - `smooth_position()`: Exponential moving average smoothing
  - `move_mouse()`: Move cursor
  - `click()`: Perform click

### Key Concepts

**Hand Landmarks (21 points):**
```
0   - Wrist
1-4 - Thumb
5-8 - Index
9-12- Middle
13-16- Ring
17-20- Pinky
```

**Coordinate Systems:**
- Normalized: (0-1) relative to frame size
- Pixel: (0-width, 0-height) in frame
- Screen: (0-screen_width, 0-screen_height)

**Smoothing:**
Exponential moving average: `smooth_value = α * last + (1 - α) * new`
- α = 0.5: 50/50 blend (responsive)
- α = 0.7: 70/30 blend (smooth)
- α = 0.9: 90/10 blend (very smooth, laggy)

## Next Steps

1. **Gesture Recognition**: Build classifiers for custom gestures
2. **Multi-hand Support**: Independent control for left/right hands
3. **Robotic Arm Interface**: Map hand pose to arm kinematics
4. **Real-time Feedback**: Add visual feedback for gesture states
5. **Performance Optimization**: Profile and optimize detection loop

## Troubleshooting

**Webcam not opening:**
- Check device permissions (macOS: System Preferences → Security & Privacy)
- Try `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

**Hand detection not working:**
- Ensure good lighting
- Keep hands in view
- Try adjusting confidence threshold (0.3-0.9)

**Mouse movement is jittery:**
- Increase smoothing factor (0.7 → 0.8)
- Check lighting and camera quality

**Pinch detection is unreliable:**
- Adjust pinch_dist threshold (currently 30px)
- Ensure clear thumb-index separation
