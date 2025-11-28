"""
Skeleton overlay visualization for recorded walks.
Creates video with pose keypoints overlaid for visual validation of gait analysis.
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
from camera.video_source import VideoSource
from pose.pose_estimator import PoseEstimator


class SkeletonOverlay:
    """
    Creates skeleton overlay videos from recorded walk data.
    Useful for visualizing pose estimation quality and gait patterns.
    """

    def __init__(self):
        """Initialize skeleton overlay generator."""
        self.pose_estimator = PoseEstimator()

        # Colors for different body parts (BGR format)
        self.colors = {
            'joints': (0, 255, 0),      # Green circles for joints
            'connections': (255, 0, 0),  # Blue lines for connections
            'text': (255, 255, 255),     # White text
            'background': (0, 0, 0)      # Black background for text
        }

        # Joint connections for skeleton drawing
        self.connections = [
            ('left_hip', 'right_hip'),      # Hip line
            ('left_hip', 'left_knee'),      # Left leg
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),    # Right leg
            ('right_knee', 'right_ankle')
        ]

    def create_overlay_video(self,
                           frames: List[np.ndarray],
                           keypoints_sequence: List[Optional[Dict]],
                           output_path: str,
                           fps: float = 30.0,
                           show_metrics: bool = True) -> bool:
        """
        Create a video with skeleton overlay from frames and keypoints.

        Args:
            frames: List of video frames (BGR format)
            keypoints_sequence: List of pose keypoints for each frame
            output_path: Path to save the output video
            fps: Video frame rate
            show_metrics: Whether to show gait metrics on video

        Returns:
            bool: True if video was created successfully
        """
        if not frames or len(frames) != len(keypoints_sequence):
            print("Error: Frames and keypoints must have the same length")
            return False

        try:
            # Get video dimensions
            height, width = frames[0].shape[:2]

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not video_writer.isOpened():
                print(f"Error: Could not open video writer for {output_path}")
                return False

            # Process each frame
            for i, (frame, keypoints) in enumerate(zip(frames, keypoints_sequence)):
                # Create overlay frame
                overlay_frame = self._create_frame_overlay(
                    frame, keypoints, i, len(frames), show_metrics
                )

                # Write frame to video
                video_writer.write(overlay_frame)

            # Release video writer
            video_writer.release()

            print(f"Skeleton overlay video saved: {output_path}")
            return True

        except Exception as e:
            print(f"Error creating overlay video: {e}")
            return False

    def create_live_preview(self,
                          camera_index: int = 0,
                          duration_seconds: int = 10) -> Optional[Tuple[List[np.ndarray], List[Dict]]]:
        """
        Create a live preview with skeleton overlay and return data for video creation.

        Args:
            camera_index: Camera device index
            duration_seconds: Duration to record in seconds

        Returns:
            Optional[Tuple[List[np.ndarray], List[Dict]]]: (frames, keypoints) or None
        """
        frames = []
        keypoints_sequence = []

        try:
            with VideoSource(camera_index) as video:
                if not video.is_connected:
                    print("Error: Could not connect to camera")
                    return None

                fps = video.get_properties()['fps']
                total_frames = int(duration_seconds * fps)

                print(f"Recording {duration_seconds} seconds at {fps} FPS")
                print("Press 'q' to stop early, 's' to skip to analysis")

                for frame_idx in range(total_frames):
                    success, frame = video.read_frame()
                    if not success:
                        break

                    # Run pose estimation
                    pose_success, keypoints = self.pose_estimator.estimate_pose(frame)

                    if pose_success:
                        keypoints_sequence.append(keypoints)
                    else:
                        keypoints_sequence.append(None)

                    # Create overlay for display
                    display_frame = self._create_frame_overlay(
                        frame, keypoints if pose_success else None,
                        frame_idx, total_frames, show_metrics=True
                    )

                    # Store original frame for video creation
                    frames.append(frame.copy())

                    # Display preview
                    cv2.imshow('Gait Recording (Press q to stop, s to skip)', display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Recording stopped by user")
                        break
                    elif key == ord('s'):
                        print("Skipping to analysis")
                        break

                cv2.destroyAllWindows()

                print(f"Recorded {len(frames)} frames with pose data")
                return frames, keypoints_sequence

        except Exception as e:
            print(f"Error in live preview: {e}")
            return None

    def _create_frame_overlay(self,
                            frame: np.ndarray,
                            keypoints: Optional[Dict],
                            frame_idx: int,
                            total_frames: int,
                            show_metrics: bool = True) -> np.ndarray:
        """
        Create a single frame with skeleton overlay and metrics.

        Args:
            frame: Original video frame
            keypoints: Pose keypoints for this frame
            frame_idx: Current frame index
            total_frames: Total number of frames
            show_metrics: Whether to show metrics overlay

        Returns:
            np.ndarray: Frame with overlay
        """
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Draw skeleton if pose detected
        if keypoints is not None:
            self._draw_skeleton(overlay, keypoints, width, height)

            # Draw pose confidence indicator
            confidence = self._calculate_pose_confidence(keypoints)
            confidence_color = self._get_confidence_color(confidence)
            cv2.circle(overlay, (30, 30), 15, confidence_color, -1)
        else:
            # Draw red indicator for no pose
            cv2.circle(overlay, (30, 30), 15, (0, 0, 255), -1)

        # Draw metrics overlay
        if show_metrics:
            self._draw_metrics_overlay(overlay, frame_idx, total_frames)

        return overlay

    def _draw_skeleton(self,
                      frame: np.ndarray,
                      keypoints: Dict,
                      width: int,
                      height: int):
        """Draw skeleton connections and joints on frame."""

        # Convert normalized coordinates to pixel coordinates
        pixel_keypoints = {}
        for joint_name, joint_data in keypoints.items():
            x = int(joint_data['x'] * width)
            y = int(joint_data['y'] * height)
            pixel_keypoints[joint_name] = (x, y)

        # Draw connections
        for joint1, joint2 in self.connections:
            if (joint1 in pixel_keypoints and joint2 in pixel_keypoints and
                keypoints[joint1]['visibility'] > 0.5 and keypoints[joint2]['visibility'] > 0.5):

                pt1 = pixel_keypoints[joint1]
                pt2 = pixel_keypoints[joint2]
                cv2.line(frame, pt1, pt2, self.colors['connections'], 3)

        # Draw joints
        for joint_name, (x, y) in pixel_keypoints.items():
            if keypoints[joint_name]['visibility'] > 0.5:
                cv2.circle(frame, (x, y), 6, self.colors['joints'], -1)
                cv2.circle(frame, (x, y), 6, (0, 0, 0), 2)  # Black border

    def _draw_metrics_overlay(self,
                            frame: np.ndarray,
                            frame_idx: int,
                            total_frames: int):
        """Draw metrics and progress information on frame."""
        height, width = frame.shape[:2]

        # Progress bar
        progress = frame_idx / total_frames if total_frames > 0 else 0
        bar_width = 300
        bar_height = 10
        bar_x = width - bar_width - 20
        bar_y = 20

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)

        # Progress
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height),
                     (0, 255, 0), -1)

        # Progress text
        progress_text = f"Frame: {frame_idx + 1}/{total_frames} ({progress:.1%})"
        self._draw_text(frame, progress_text, (bar_x, bar_y - 5))

        # Instructions
        instructions = [
            "Recording Gait Analysis",
            "Keep walking naturally",
            "Stay in camera view"
        ]

        for i, instruction in enumerate(instructions):
            y_pos = height - 80 + (i * 25)
            self._draw_text(frame, instruction, (20, y_pos))

    def _draw_text(self,
                  frame: np.ndarray,
                  text: str,
                  position: Tuple[int, int],
                  font_scale: float = 0.6,
                  thickness: int = 1):
        """Draw text with background for better visibility."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        x, y = position
        # Draw background rectangle
        cv2.rectangle(frame,
                     (x - 2, y - text_height - 2),
                     (x + text_width + 2, y + baseline + 2),
                     self.colors['background'], -1)

        # Draw text
        cv2.putText(frame, text, (x, y), font, font_scale,
                   self.colors['text'], thickness, cv2.LINE_AA)

    def _calculate_pose_confidence(self, keypoints: Dict) -> float:
        """Calculate overall confidence of pose detection."""
        if not keypoints:
            return 0.0

        confidences = [joint['visibility'] for joint in keypoints.values()]
        return sum(confidences) / len(confidences)

    def _get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """Get color based on confidence level (BGR format)."""
        if confidence > 0.8:
            return (0, 255, 0)      # Green - high confidence
        elif confidence > 0.5:
            return (0, 255, 255)    # Yellow - medium confidence
        else:
            return (0, 0, 255)      # Red - low confidence

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'pose_estimator'):
            self.pose_estimator.cleanup()


def test_skeleton_overlay():
    """Test skeleton overlay functionality."""
    print("Testing skeleton overlay...")

    overlay_gen = SkeletonOverlay()

    # Test live preview
    print("Starting live preview test (10 seconds)")
    print("Stand in front of camera and walk in place")

    result = overlay_gen.create_live_preview(camera_index=0, duration_seconds=10)

    if result is not None:
        frames, keypoints_sequence = result

        # Create overlay video
        output_path = "test_gait_overlay.mp4"
        success = overlay_gen.create_overlay_video(
            frames, keypoints_sequence, output_path, fps=30.0
        )

        if success:
            print(f"Test video saved: {output_path}")
        else:
            print("Failed to create test video")
    else:
        print("Failed to capture test data")

    overlay_gen.cleanup()
    print("Skeleton overlay test completed")


if __name__ == "__main__":
    test_skeleton_overlay()