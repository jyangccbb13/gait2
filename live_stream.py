"""
Live gait analysis with real-time pose overlay.
Shows live video feed with skeleton overlay and basic gait metrics.
"""

import cv2
import numpy as np
import time
from typing import Dict, Optional
from camera.video_source import VideoSource
from pose.pose_estimator import PoseEstimator
from gait.gait_features import GaitAnalyzer


class LiveGaitStream:
    """
    Real-time gait analysis with live video display and pose overlay.
    """

    def __init__(self, video_source_index=0):
        """
        Initialize live gait stream.

        Args:
            video_source_index: Camera index (0 for MacBook, 1 for external, or RTMP URL)
        """
        self.video_source = VideoSource(video_source_index)
        self.pose_estimator = PoseEstimator()
        self.gait_analyzer = GaitAnalyzer(fps=30.0)

        # Tracking variables
        self.keypoint_history = []
        self.frame_count = 0
        self.start_time = time.time()

        # Display settings
        self.show_metrics = True
        self.show_skeleton = True
        self.show_confidence = True

        # Colors (BGR format)
        self.colors = {
            'skeleton_joints': (0, 255, 0),     # Green joints
            'skeleton_bones': (255, 0, 0),      # Blue bones
            'text': (255, 255, 255),            # White text
            'background': (0, 0, 0),            # Black background
            'high_confidence': (0, 255, 0),     # Green
            'medium_confidence': (0, 255, 255), # Yellow
            'low_confidence': (0, 0, 255)       # Red
        }

    def start_stream(self):
        """Start the live video stream with pose overlay."""
        print("Starting live gait analysis stream...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Toggle skeleton overlay")
        print("  'm' - Toggle metrics display")
        print("  'c' - Toggle confidence indicators")
        print("  'r' - Reset gait history")

        if not self.video_source.connect():
            print("Error: Could not connect to camera")
            return

        # Get actual FPS from camera
        camera_fps = self.video_source.get_properties()['fps']
        self.gait_analyzer.fps = camera_fps
        print(f"Camera connected: {camera_fps} FPS")

        try:
            while True:
                success, frame = self.video_source.read_frame()
                if not success:
                    print("Failed to read frame")
                    break

                # Process frame
                processed_frame = self._process_frame(frame)

                # Display frame
                cv2.imshow('Live Gait Analysis (Press q to quit)', processed_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.show_skeleton = not self.show_skeleton
                    print(f"Skeleton overlay: {'ON' if self.show_skeleton else 'OFF'}")
                elif key == ord('m'):
                    self.show_metrics = not self.show_metrics
                    print(f"Metrics display: {'ON' if self.show_metrics else 'OFF'}")
                elif key == ord('c'):
                    self.show_confidence = not self.show_confidence
                    print(f"Confidence indicators: {'ON' if self.show_confidence else 'OFF'}")
                elif key == ord('r'):
                    self._reset_tracking()
                    print("Gait tracking reset")

        except KeyboardInterrupt:
            print("Stream interrupted by user")

        finally:
            self._cleanup()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with pose estimation and overlay."""
        self.frame_count += 1
        processed_frame = frame.copy()

        # Run pose estimation
        pose_success, keypoints = self.pose_estimator.estimate_pose(frame)

        if pose_success and self.pose_estimator.is_pose_valid(keypoints):
            # Store keypoints for gait analysis
            self.keypoint_history.append(keypoints)

            # Keep only recent keypoints (last 10 seconds)
            max_history = int(self.gait_analyzer.fps * 10)
            if len(self.keypoint_history) > max_history:
                self.keypoint_history = self.keypoint_history[-max_history:]

            # Draw skeleton overlay
            if self.show_skeleton:
                processed_frame = self._draw_skeleton_overlay(processed_frame, keypoints)

            # Draw confidence indicator
            if self.show_confidence:
                self._draw_confidence_indicator(processed_frame, keypoints)
        else:
            # Add None for failed detection
            self.keypoint_history.append(None)

        # Draw metrics overlay
        if self.show_metrics:
            self._draw_metrics_overlay(processed_frame)

        return processed_frame

    def _draw_skeleton_overlay(self, frame: np.ndarray, keypoints: Dict) -> np.ndarray:
        """Draw skeleton joints and connections on frame."""
        height, width = frame.shape[:2]

        # Convert normalized coordinates to pixel coordinates
        pixel_joints = {}
        for joint_name, joint_data in keypoints.items():
            x = int(joint_data['x'] * width)
            y = int(joint_data['y'] * height)
            visibility = joint_data['visibility']

            # Only draw if visibility is good enough
            if visibility > 0.5:
                pixel_joints[joint_name] = (x, y, visibility)

        # Draw bone connections
        connections = [
            ('left_hip', 'right_hip'),      # Hip line
            ('left_hip', 'left_knee'),      # Left leg
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),    # Right leg
            ('right_knee', 'right_ankle')
        ]

        for joint1, joint2 in connections:
            if joint1 in pixel_joints and joint2 in pixel_joints:
                pt1 = pixel_joints[joint1][:2]
                pt2 = pixel_joints[joint2][:2]

                # Line thickness based on average confidence
                avg_confidence = (pixel_joints[joint1][2] + pixel_joints[joint2][2]) / 2
                thickness = max(1, int(avg_confidence * 4))

                cv2.line(frame, pt1, pt2, self.colors['skeleton_bones'], thickness)

        # Draw joints
        for joint_name, (x, y, visibility) in pixel_joints.items():
            # Joint size based on confidence
            radius = max(3, int(visibility * 8))

            # Draw joint circle
            cv2.circle(frame, (x, y), radius, self.colors['skeleton_joints'], -1)
            cv2.circle(frame, (x, y), radius, (0, 0, 0), 1)  # Black border

            # Draw joint label
            if visibility > 0.7:  # Only label high-confidence joints
                label = joint_name.replace('_', ' ').title()
                self._draw_text(frame, label, (x + 10, y - 10), scale=0.4)

        return frame

    def _draw_confidence_indicator(self, frame: np.ndarray, keypoints: Dict):
        """Draw confidence indicator in corner."""
        # Calculate overall confidence
        confidences = [joint['visibility'] for joint in keypoints.values()]
        avg_confidence = sum(confidences) / len(confidences)

        # Choose color based on confidence
        if avg_confidence > 0.8:
            color = self.colors['high_confidence']
            status = "HIGH"
        elif avg_confidence > 0.5:
            color = self.colors['medium_confidence']
            status = "MEDIUM"
        else:
            color = self.colors['low_confidence']
            status = "LOW"

        # Draw confidence indicator
        cv2.circle(frame, (30, 30), 20, color, -1)
        cv2.circle(frame, (30, 30), 20, (0, 0, 0), 2)

        # Draw confidence text
        confidence_text = f"Confidence: {status} ({avg_confidence:.2f})"
        self._draw_text(frame, confidence_text, (60, 35))

    def _draw_metrics_overlay(self, frame: np.ndarray):
        """Draw real-time gait metrics on frame."""
        height, width = frame.shape[:2]

        # Calculate current metrics if we have enough data
        metrics_text = []

        # Basic frame info
        elapsed_time = time.time() - self.start_time
        metrics_text.append(f"Time: {elapsed_time:.1f}s")
        metrics_text.append(f"Frames: {self.frame_count}")
        metrics_text.append(f"FPS: {self.frame_count / elapsed_time:.1f}")

        # Gait metrics (if enough data)
        if len(self.keypoint_history) > 30:  # At least 1 second of data
            try:
                # Analyze recent gait pattern
                gait_results = self.gait_analyzer.analyze_walk(self.keypoint_history[-90:])  # Last 3 seconds

                metrics_text.append("")  # Spacer
                metrics_text.append("=== GAIT METRICS ===")
                metrics_text.append(f"Speed: {gait_results['average_speed']:.4f}")

                stride_metrics = gait_results['stride_metrics']
                if stride_metrics.get('stride_length_left', 0) > 0:
                    metrics_text.append(f"Stride Length: {stride_metrics['stride_length_left']:.3f}")
                if stride_metrics.get('stride_time_variability', 0) > 0:
                    metrics_text.append(f"Variability: {stride_metrics['stride_time_variability']:.3f}")

                # Stride events
                stride_events = gait_results['stride_events']
                left_strides = len(stride_events['left'])
                right_strides = len(stride_events['right'])
                metrics_text.append(f"Strides: L{left_strides} R{right_strides}")

            except Exception as e:
                metrics_text.append(f"Analysis error: {str(e)[:20]}...")

        # Draw metrics panel
        panel_width = 300
        panel_height = len(metrics_text) * 25 + 20
        panel_x = width - panel_width - 10
        panel_y = 10

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text
        for i, text in enumerate(metrics_text):
            y_pos = panel_y + 20 + (i * 20)
            self._draw_text(frame, text, (panel_x + 10, y_pos), scale=0.5)

    def _draw_text(self, frame: np.ndarray, text: str, position, scale=0.6):
        """Draw text with background for visibility."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)

        x, y = position

        # Draw text
        cv2.putText(frame, text, (x, y), font, scale, self.colors['text'], thickness, cv2.LINE_AA)

    def _reset_tracking(self):
        """Reset gait tracking data."""
        self.keypoint_history = []
        self.frame_count = 0
        self.start_time = time.time()

    def _cleanup(self):
        """Clean up resources."""
        cv2.destroyAllWindows()
        self.video_source.disconnect()
        self.pose_estimator.cleanup()
        print("Live stream ended")


def main():
    """Main function to start live gait stream."""
    print("Live Gait Analysis Stream")
    print("=" * 40)

    # Ask user for camera source
    print("Camera options:")
    print("  0 - MacBook built-in camera")
    print("  1 - External USB camera")
    print("  Or enter RTMP URL for DJI camera")

    source_input = input("Enter camera source (default: 0): ").strip()

    if source_input == "":
        source = 0
    elif source_input.isdigit():
        source = int(source_input)
    else:
        source = source_input  # Assume it's a URL

    # Start live stream
    live_stream = LiveGaitStream(source)
    live_stream.start_stream()


if __name__ == "__main__":
    main()