# Hand Control

Complete hand gesture recognition and robotic arm control system. From webcam input to physical arm manipulation, with intuitive gesture-based interfaces.

## Vision

Build a comprehensive system for human-computer and human-robot interaction through natural hand gestures, progressing from screen control to precise robotic arm manipulation.

## Features

✅ **Stage 1: Hand Detection & Tracking** — Real-time pose estimation with trajectory history, velocity/acceleration calculation, stability detection
✅ **Stage 2: Screen Control** — Gesture-based mouse control with pinch, point, swipe, and two-finger operations
✅ **Stage 3: Gesture Recognition** — Template-based gesture matching (static and temporal), built-in gestures, custom training
✅ **Stage 4: Robotic Arm Interface** — 3-DOF and 6-DOF arm kinematics, forward/inverse kinematics, hand-to-arm mapping, trajectory planning

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run examples
python examples/detect_hand.py          # Hand detection & tracking
python examples/screen_control.py       # Hand → mouse control
python examples/demo_gestures.py        # Gesture recognition
python examples/demo_arm_control.py     # Hand → arm control

# Run tests
python tests/test_kinematics.py
python tests/test_gestures.py
```

## Architecture

### Hand Detection Module (`src/hand_detection/detector.py`)

Real-time hand pose estimation using MediaPipe with advanced tracking:

- **21-point hand landmarks** — Fingertips, finger bases, palm
- **Multi-hand support** — Track up to 2 hands independently
- **Motion metrics** — Velocity, acceleration, jitter, stability scoring
- **Gesture state machine** — Foundation for temporal gesture recognition
- **Trajectory history** — Configurable frame buffer for historical analysis

**Key Methods:**
```python
detector = HandDetector(max_hands=2, confidence=0.6, history_frames=30)
results, h, w, c = detector.detect(frame)
hands = detector.get_hand_landmarks(results, h, w)
pinch_distance = detector.get_pinch_distance(hand_data)
velocity = detector.get_hand_velocity(hand_id)
stability = detector.get_stability_score(hand_id)
```

### Screen Control Module (`src/control/screen_control.py`)

Gesture-based mouse and system control with state management:

- **Point gesture** — Index finger extended → cursor control
- **Pinch gesture** — Thumb + index close → click action
- **Palm open** — All fingers extended → drag mode
- **Swipe gestures** — Left/right/up/down detection
- **Two-finger pinch** — Scroll wheel control
- **Gesture recording** — Save gesture sequences for training

**Gesture State Machine:**
- IDLE → POINTING → moving cursor
- POINTING → PINCHING → click
- PINCHING → IDLE → release
- POINTING → SWIPING → directional action

### Gesture Recognition Engine (`src/gestures/recognizer.py`)

Template-based gesture recognition with static and temporal support:

- **Built-in gestures** — Thumbs up/down, OK sign, peace, rock, fist, palm, pointing, victory
- **Custom gestures** — Train from recorded samples using averaging
- **Distance metrics** — Euclidean distance for static poses, DTW for temporal sequences
- **Confidence scoring** — 0-1 score indicating recognition confidence
- **Dynamic gestures** — Multi-frame temporal sequences with temporal alignment

**Example Usage:**
```python
recognizer = GestureRecognizer()

# Static gesture recognition
matches = recognizer.recognize(hand_data, top_k=3)
for match in matches:
    print(f"{match.gesture_name}: {match.confidence:.2f}")

# Train custom gesture
samples = [record_gesture() for _ in range(5)]
recognizer.train_gesture('custom_gesture', samples)

# Temporal gesture recognition
trajectory = [detector.get_hand_landmarks(...) for _ in range(10)]
matches = recognizer.recognize_temporal(trajectory)
```

### Robotic Arm Interface

#### Kinematics (`src/arm/kinematics.py`)

Forward and inverse kinematics solvers for robotic arms:

- **Denavit-Hartenberg (DH) parameterization** — Standard arm representation
- **Forward kinematics** — Joint angles → end effector pose
- **Inverse kinematics** — End effector pose → joint angles (numerical solver)
- **Jacobian computation** — Velocity transformation
- **Joint limits** — Automatic enforcement of angular/position limits
- **Configurable DOF** — Support 3-DOF, 6-DOF, or custom arms

**Example:**
```python
kinematics = Kinematics.from_config('data/arm_configs/3dof_arm.json')

# Forward kinematics
pose = kinematics.forward_kinematics(np.array([0, 0.5, -0.5]))
end_effector = pose[:3, 3]  # 3D position

# Inverse kinematics
target_pose = np.eye(4)
target_pose[:3, 3] = [300, 100, 500]  # Target position (mm)
joints, success = kinematics.inverse_kinematics(target_pose)
```

#### Arm Controller (`src/arm/arm_controller.py`)

High-level arm control with hand mapping and trajectory generation:

- **Hand-to-arm mapping** — Project hand position to arm workspace
- **Position control mode** — Direct end effector positioning
- **Velocity control mode** — Continuous velocity-based motion
- **Workspace management** — Define reachable boundaries
- **Trajectory generation** — Smooth motion with acceleration limits
- **State management** — Save/load arm configuration

**Example:**
```python
controller = ArmController(kinematics, mode=ControlMode.POSITION)
controller.enable_hand_tracking(hand_origin=[0, 0, 500])

# Each frame:
controller.set_hand_target_position(hand_data, frame_w, frame_h)
controller.step_simulation(dt=0.01)
current_joints = controller.current_joints
```

## Project Structure

```
hand-control/
├── src/
│   ├── hand_detection/
│   │   ├── detector.py          # HandDetector class
│   │   └── __init__.py
│   ├── gestures/
│   │   ├── recognizer.py        # GestureRecognizer class
│   │   └── __init__.py
│   ├── control/
│   │   ├── screen_control.py    # ScreenController class
│   │   └── __init__.py
│   ├── arm/
│   │   ├── kinematics.py        # Kinematics solver
│   │   ├── arm_controller.py    # ArmController class
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── data/
│   ├── arm_configs/
│   │   ├── 3dof_arm.json        # 3-DOF arm DH parameters
│   │   └── 6dof_arm.json        # 6-DOF industrial arm
│   ├── gestures/                # Training data (populated by user)
│   └── models/                  # Trained gesture models
├── config/
│   └── control_config.json      # System configuration
├── tests/
│   ├── test_kinematics.py       # Kinematics unit tests
│   └── test_gestures.py         # Gesture recognition tests
├── examples/
│   ├── detect_hand.py           # Hand detection demo
│   ├── screen_control.py        # Screen control demo
│   ├── demo_gestures.py         # Gesture recognition demo
│   └── demo_arm_control.py      # Arm control demo
├── docs/
│   └── DEVELOPMENT.md           # Developer guide
└── README.md
```

## Configuration

All system parameters are configurable via `config/control_config.json`:

```json
{
  "camera": {"index": 0, "width": 1280, "height": 720, "fps": 30},
  "hand_detection": {"max_hands": 2, "confidence_threshold": 0.6},
  "screen_control": {"smoothing_factor": 0.7, "pinch_distance_threshold": 30},
  "gesture_recognition": {"confidence_threshold": 0.7},
  "arm_control": {
    "control_mode": "position",
    "workspace": {"min_x": -500, "max_x": 500, ...}
  }
}
```

## Gesture Library

### Built-in Static Gestures
- **thumbs_up** — Thumb extended upward
- **thumbs_down** — Thumb extended downward
- **ok_sign** — Thumb + index form circle
- **peace** — Index + middle extended
- **rock** — Index + pinky extended
- **fist** — Closed hand
- **open_palm** — All fingers extended
- **pointing** — Index finger extended

### Built-in Screen Control Gestures
- **point** — Move cursor
- **pinch** — Click
- **palm_open** — Drag mode
- **swipe_left/right/up/down** — Scroll/navigate
- **two_finger_pinch** — Scroll wheel

## Arm Configurations

### 3-DOF Planar Arm
- Configuration: Shoulder, elbow, wrist
- Workspace: XY plane, 500mm reach
- Use case: Desktop manipulation, basic tasks
- File: `data/arm_configs/3dof_arm.json`

### 6-DOF Industrial Arm
- Configuration: ±310mm base, 6 revolute joints
- Workspace: 1.9m reach, full 3D positioning + orientation
- Use case: Precision manipulation, complex tasks
- File: `data/arm_configs/6dof_arm.json`

## Dependencies

```
opencv-python>=4.8.0      # Video capture and processing
mediapipe>=0.10.0         # Hand pose estimation
numpy>=1.24.0             # Numerical computing
scipy>=1.11.0             # Scientific computing
pyautogui>=0.9.53         # Screen control (macOS/Linux)
```

## Examples

### 1. Real-time Hand Detection
```bash
python examples/detect_hand.py
# Shows: Hand landmarks, confidence scores, motion metrics
```

### 2. Hand-to-Mouse Control
```bash
python examples/screen_control.py
# Controls: Point gesture → move mouse, Pinch → click
```

### 3. Gesture Recognition
```bash
python examples/demo_gestures.py
# Shows: Real-time gesture matching with confidence scores
```

### 4. Hand-Controlled Robotic Arm
```bash
python examples/demo_arm_control.py
# Controls: Hand position → 3-DOF arm end effector
```

## Testing

Run the test suite to verify all components:

```bash
# Kinematics tests (forward/inverse, Jacobian, limits)
python tests/test_kinematics.py

# Gesture recognition tests (matching, training, temporal)
python tests/test_gestures.py
```

## Development

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for:
- Setup instructions
- Architecture details
- Coordinate systems and conventions
- Troubleshooting guide
- Contributing guidelines

## Performance

Typical performance on modern hardware:

- **Hand detection:** 30 FPS @ 1280×720 (M-series Mac)
- **Gesture recognition:** <50ms per frame
- **Inverse kinematics:** 20-100ms (depends on convergence)
- **Arm simulation:** Real-time (limited by display refresh)

## Future Extensions

- [ ] Real hardware integration (serial/Ethernet arm interface)
- [ ] Force feedback and haptics
- [ ] Multi-modal sensors (depth, IMU)
- [ ] Advanced gesture learning (neural networks)
- [ ] Grasp planning and object manipulation
- [ ] Collaborative robot safety

## License

MIT License - See LICENSE file

## Contributing

Contributions welcome! Please:
1. Follow the code style (PEP 8)
2. Add tests for new features
3. Update documentation
4. Submit pull requests

---

**Ready to control the world with your hands! 🤖**
