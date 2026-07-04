import pyautogui
import numpy as np


class ScreenController:
    """Control mouse cursor based on hand position."""

    def __init__(self, screen_w=None, screen_h=None, smoothing=0.6):
        """Initialize screen controller.

        Args:
            screen_w: Screen width (auto-detect if None)
            screen_h: Screen height (auto-detect if None)
            smoothing: Exponential moving average smoothing factor (0-1)
        """
        if screen_w is None or screen_h is None:
            screen_size = pyautogui.size()
            self.screen_w = screen_w or screen_size[0]
            self.screen_h = screen_h or screen_size[1]
        else:
            self.screen_w = screen_w
            self.screen_h = screen_h

        self.smoothing = smoothing
        self.last_x = None
        self.last_y = None

        # Disable pyautogui safety (corners of screen won't stop movement)
        pyautogui.FAILSAFE = False

    def hand_to_screen(self, hand_data, frame_w, frame_h, invert_x=False):
        """Convert hand palm center position to screen coordinates.

        Args:
            hand_data: Hand data dict from get_hand_landmarks
            frame_w: Frame width
            frame_h: Frame height
            invert_x: Flip horizontal axis (mirror)

        Returns:
            screen_x, screen_y: Screen coordinates
        """
        palm_x, palm_y = hand_data['palm_center']

        # Normalize to 0-1
        norm_x = palm_x / frame_w
        norm_y = palm_y / frame_h

        if invert_x:
            norm_x = 1 - norm_x

        # Map to screen coordinates
        screen_x = int(norm_x * self.screen_w)
        screen_y = int(norm_y * self.screen_h)

        return screen_x, screen_y

    def smooth_position(self, x, y):
        """Apply exponential moving average smoothing.

        Args:
            x, y: New position

        Returns:
            smoothed_x, smoothed_y: Smoothed position
        """
        if self.last_x is None:
            self.last_x = x
            self.last_y = y
            return x, y

        smoothed_x = int(self.last_x * self.smoothing + x * (1 - self.smoothing))
        smoothed_y = int(self.last_y * self.smoothing + y * (1 - self.smoothing))

        self.last_x = smoothed_x
        self.last_y = smoothed_y

        return smoothed_x, smoothed_y

    def move_mouse(self, hand_data, frame_w, frame_h, smooth=True, invert_x=False):
        """Move mouse cursor to hand position.

        Args:
            hand_data: Hand data dict
            frame_w: Frame width
            frame_h: Frame height
            smooth: Apply smoothing
            invert_x: Flip horizontal axis
        """
        screen_x, screen_y = self.hand_to_screen(hand_data, frame_w, frame_h, invert_x)

        if smooth:
            screen_x, screen_y = self.smooth_position(screen_x, screen_y)

        pyautogui.moveTo(screen_x, screen_y, duration=0)

    def click(self, button='left', duration=0.1):
        """Perform mouse click.

        Args:
            button: 'left', 'right', or 'middle'
            duration: Click duration
        """
        pyautogui.click(button=button, duration=duration)

    def drag(self, start_x, start_y, end_x, end_y, duration=0.5):
        """Drag from one position to another.

        Args:
            start_x, start_y: Starting position
            end_x, end_y: Ending position
            duration: Duration of drag
        """
        pyautogui.moveTo(start_x, start_y, duration=0)
        pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration)

    def reset_smoothing(self):
        """Reset smoothing state."""
        self.last_x = None
        self.last_y = None
