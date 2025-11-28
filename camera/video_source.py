"""
Video source handling for gait analysis system.
Supports local webcam and RTSP/RTMP streams for future DJI camera integration.
"""

import cv2
import time
from typing import Optional, Tuple
import numpy as np


class VideoSource:
    """
    Handles video input from various sources (webcam, RTSP streams).

    Attributes:
        source: Video source (camera index or URL)
        cap: OpenCV VideoCapture object
        fps: Frames per second of the video source
        frame_width: Width of video frames
        frame_height: Height of video frames
    """

    def __init__(self, source: str = 0):
        """
        Initialize video source.

        Args:
            source: Camera index (0, 1, 2...) or RTSP/RTMP URL
                   For webcam: 0 (default camera), 1 (second camera), etc.
                   For streams: "rtsp://192.168.1.100:8554/stream"
        """
        self.source = source
        self.cap = None
        self.fps = 30.0  # Default FPS, will be updated from actual source
        self.frame_width = 640
        self.frame_height = 480
        self.is_connected = False

    def connect(self) -> bool:
        """
        Connect to the video source.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                print(f"Error: Could not open video source {self.source}")
                return False

            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.is_connected = True
            print(f"Connected to video source: {self.source}")
            print(f"Resolution: {self.frame_width}x{self.frame_height} @ {self.fps} FPS")

            return True

        except Exception as e:
            print(f"Error connecting to video source: {e}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the video source.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
                success: True if frame was read successfully
                frame: BGR image as numpy array, None if failed
        """
        if not self.is_connected or self.cap is None:
            return False, None

        ret, frame = self.cap.read()

        if not ret:
            print("Warning: Failed to read frame from video source")
            return False, None

        return True, frame

    def get_properties(self) -> dict:
        """
        Get video source properties.

        Returns:
            dict: Video properties including fps, width, height
        """
        return {
            'fps': self.fps,
            'width': self.frame_width,
            'height': self.frame_height,
            'source': self.source,
            'connected': self.is_connected
        }

    def set_resolution(self, width: int, height: int) -> bool:
        """
        Set video capture resolution (works for some webcams).

        Args:
            width: Desired frame width
            height: Desired frame height

        Returns:
            bool: True if resolution was set successfully
        """
        if not self.is_connected or self.cap is None:
            return False

        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Verify the resolution was actually set
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.frame_width = actual_width
            self.frame_height = actual_height

            print(f"Resolution set to: {actual_width}x{actual_height}")
            return True

        except Exception as e:
            print(f"Error setting resolution: {e}")
            return False

    def disconnect(self):
        """Release the video source and clean up resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.is_connected = False
        print(f"Disconnected from video source: {self.source}")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def test_video_source():
    """Test function to verify video source functionality."""
    print("Testing video source...")

    # Test with default webcam
    with VideoSource(0) as video:
        if not video.is_connected:
            print("No webcam found")
            return

        print("Video properties:", video.get_properties())

        # Read a few test frames
        for i in range(5):
            success, frame = video.read_frame()
            if success:
                print(f"Frame {i+1}: {frame.shape}")
                # Optionally display frame
                # cv2.imshow('Test Frame', frame)
                # cv2.waitKey(1)
            else:
                print(f"Failed to read frame {i+1}")

            time.sleep(0.1)

    print("Video source test completed")


if __name__ == "__main__":
    test_video_source()