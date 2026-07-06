"""
Tests for gesture recognition engine.
"""

import sys
import numpy as np

sys.path.insert(0, '/Users/edison.zhu/hand-control')

from src.gestures.recognizer import GestureRecognizer, GestureTemplate


def test_gesture_recognizer_initialization():
    """Test gesture recognizer initialization."""
    recognizer = GestureRecognizer()

    assert recognizer.num_joints is None, "Should initialize without specific arm"
    assert len(recognizer.templates) > 0, "Should have built-in gesture templates"

    print("✓ Gesture recognizer initialization passed")


def test_add_custom_gesture():
    """Test adding custom gesture template."""
    recognizer = GestureRecognizer()

    # Create custom gesture template
    template = np.random.randn(21, 3) * 0.1 + 0.5
    recognizer.add_gesture('test_gesture', template, gesture_type='custom')

    assert 'test_gesture' in recognizer.templates, "Gesture not added"
    print("✓ Add custom gesture passed")


def test_recognize_gesture():
    """Test gesture recognition."""
    recognizer = GestureRecognizer()

    # Create synthetic hand data
    hand_data = {
        'landmarks': [
            {'norm': (x, y, z)}
            for x, y, z in np.random.randn(21, 3) * 0.1 + 0.5
        ]
    }

    # Recognize
    matches = recognizer.recognize(hand_data, top_k=3)

    assert isinstance(matches, list), "Should return list"
    assert len(matches) <= 3, "Should return at most top_k matches"

    if matches:
        match = matches[0]
        assert 0 <= match.confidence <= 1, f"Invalid confidence: {match.confidence}"

    print("✓ Gesture recognition passed")


def test_temporal_gesture_training():
    """Test training temporal (dynamic) gestures."""
    recognizer = GestureRecognizer()

    # Create synthetic temporal gesture data (swipe-like motion)
    num_frames = 10
    samples = [
        np.random.randn(num_frames, 21, 3) * 0.05 + 0.5
        for _ in range(3)
    ]

    recognizer.train_gesture(
        'swipe_motion',
        samples,
        gesture_type='swipe',
        is_dynamic=True
    )

    assert 'swipe_motion' in recognizer.templates, "Temporal gesture not trained"
    template = recognizer.templates['swipe_motion']
    assert template.is_dynamic, "Gesture should be marked as dynamic"

    print("✓ Temporal gesture training passed")


def test_recognize_temporal_gesture():
    """Test temporal gesture recognition."""
    recognizer = GestureRecognizer()

    # Create synthetic temporal gesture
    num_frames = 8
    trajectory = [
        {
            'landmarks': [
                {'norm': (x, y, z)}
                for x, y, z in np.random.randn(21, 3) * 0.05 + 0.5
            ]
        }
        for _ in range(num_frames)
    ]

    # Train simple temporal gesture
    samples = [np.random.randn(num_frames, 21, 3) * 0.05 + 0.5 for _ in range(2)]
    recognizer.train_gesture('test_swipe', samples, is_dynamic=True)

    # Recognize
    matches = recognizer.recognize_temporal(trajectory, top_k=1)

    assert isinstance(matches, list), "Should return list"
    print("✓ Temporal gesture recognition passed")


def test_distance_to_confidence():
    """Test distance-to-confidence conversion."""
    recognizer = GestureRecognizer()

    # Test edge cases
    conf_0 = recognizer._distance_to_confidence(0.0)
    conf_max = recognizer._distance_to_confidence(0.5)
    conf_beyond = recognizer._distance_to_confidence(1.0)

    assert conf_0 == 1.0, "Zero distance should give confidence 1.0"
    assert conf_max == 0.0, "Max distance should give confidence 0.0"
    assert conf_beyond == 0.0, "Beyond max distance should be clamped"

    print("✓ Distance to confidence conversion passed")


def test_list_gestures():
    """Test listing available gestures."""
    recognizer = GestureRecognizer()

    gestures = recognizer.list_gestures()

    assert isinstance(gestures, list), "Should return list"
    assert len(gestures) > 0, "Should have gestures available"
    assert 'thumbs_up' in gestures, "Should have built-in gestures"

    print(f"✓ List gestures passed ({len(gestures)} gestures)")


def test_gesture_info():
    """Test getting gesture information."""
    recognizer = GestureRecognizer()

    info = recognizer.get_gesture_info('thumbs_up')

    assert 'name' in info, "Info should have name"
    assert 'type' in info, "Info should have type"
    assert 'is_dynamic' in info, "Info should have is_dynamic"
    assert 'confidence_threshold' in info, "Info should have confidence_threshold"

    print("✓ Gesture info passed")


def test_gesture_save_load():
    """Test saving and loading gestures."""
    import tempfile
    import os

    recognizer = GestureRecognizer()

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save gesture
        gesture_path = os.path.join(tmpdir, 'test_gesture.pkl')
        recognizer.save_gesture('thumbs_up', gesture_path)

        assert os.path.exists(gesture_path), "Gesture file not created"

        # Create new recognizer and load
        recognizer2 = GestureRecognizer()
        recognizer2.load_gesture('thumbs_up_loaded', gesture_path)

        assert 'thumbs_up_loaded' in recognizer2.templates, "Gesture not loaded"

    print("✓ Gesture save/load passed")


if __name__ == "__main__":
    print("Running Gesture Recognition Tests")
    print("=" * 50)

    try:
        test_gesture_recognizer_initialization()
        test_add_custom_gesture()
        test_recognize_gesture()
        test_temporal_gesture_training()
        test_recognize_temporal_gesture()
        test_distance_to_confidence()
        test_list_gestures()
        test_gesture_info()
        test_gesture_save_load()

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
