# Hand Control

A progressive hand gesture and movement recognition system, starting with screen control and evolving toward robotic arm manipulation.

## Vision

Build a system that can understand and respond to human hand movements, enabling intuitive control of digital and physical systems through natural gestures.

### Roadmap

**Stage 1: Hand Detection & Tracking**
- Real-time hand pose estimation from webcam
- Smooth hand trajectory tracking
- Multi-hand detection

**Stage 2: Screen Control**
- Mouse cursor control via hand position
- Click/drag operations via gestures
- Gesture recognition (pinch, point, wave, etc.)
- System integration (macOS, Linux, Windows)

**Stage 3: Gesture Recognition**
- Custom gesture definition and training
- Real-time gesture classification
- Confidence scoring and filtering

**Stage 4: Robotic Arm Interface**
- Arm kinematics and forward/inverse kinematics
- Gesture-to-arm-movement mapping
- Real-time control loop
- Feedback integration (camera, sensors)

## Project Structure

```
hand-control/
├── src/
│   ├── hand_detection/      # Pose estimation, hand tracking
│   ├── gestures/            # Gesture recognition & classification
│   ├── control/             # Screen/system control interface
│   ├── arm/                 # Robotic arm interface (future)
│   └── utils/               # Common utilities
├── data/
│   ├── gestures/            # Training data for custom gestures
│   └── models/              # Trained ML models
├── tests/
├── examples/                # Demo scripts
├── docs/                    # Documentation
└── requirements.txt
```

## Getting Started

```bash
# Clone and install
cd hand-control
pip install -r requirements.txt

# Run hand detection demo
python examples/detect_hand.py

# Run screen control
python examples/screen_control.py
```

## Dependencies

- **OpenCV** — Video capture and processing
- **MediaPipe** — Hand pose estimation (initial)
- **NumPy/SciPy** — Math and signal processing
- **PyAutoGUI** (macOS/Linux) — Screen control
- Future: Computer vision models, arm control libraries

## Current Status

🔧 Project initialized. Stage 1 (hand detection setup) coming next.
