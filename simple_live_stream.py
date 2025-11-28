"""
Simple live gait analysis - automatically uses MacBook camera.
Shows live video with pose overlay and real-time metrics.
"""

import cv2
import numpy as np
import time
from camera.video_source import VideoSource
from pose.pose_estimator import PoseEstimator
from gait.gait_features import GaitAnalyzer


def main():
    """Start live gait analysis with MacBook camera."""
    print("🚶‍♂️ Starting Live Gait Analysis...")
    print("Controls: 'q' to quit, 's' to toggle skeleton, 'm' to toggle metrics")

    # Initialize components
    video_source = VideoSource(0)  # MacBook camera
    pose_estimator = PoseEstimator()
    gait_analyzer = GaitAnalyzer(fps=30.0)

    # Tracking variables
    keypoint_history = []
    frame_count = 0
    start_time = time.time()

    # Display settings
    show_skeleton = True
    show_metrics = True

    if not video_source.connect():
        print("❌ Error: Could not connect to camera")
        return

    print("✅ Camera connected! Press 'q' to quit")

    try:
        while True:
            success, frame = video_source.read_frame()
            if not success:
                print("Failed to read frame")
                break

            frame_count += 1
            processed_frame = frame.copy()

            # Run pose estimation
            pose_success, keypoints = pose_estimator.estimate_pose(frame)

            if pose_success and keypoints:
                # Store keypoints
                keypoint_history.append(keypoints)

                # Keep only recent keypoints (last 5 seconds)
                max_history = 150
                if len(keypoint_history) > max_history:
                    keypoint_history = keypoint_history[-max_history:]

                # Draw skeleton overlay
                if show_skeleton:
                    processed_frame = draw_skeleton(processed_frame, keypoints)

                # Draw confidence indicator
                draw_confidence_indicator(processed_frame, keypoints)
            else:
                keypoint_history.append(None)

            # Draw metrics
            if show_metrics:
                draw_metrics(processed_frame, keypoint_history, frame_count, start_time, gait_analyzer)

            # Show frame
            cv2.imshow('🚶‍♂️ Live Gait Analysis (q=quit, s=skeleton, m=metrics)', processed_frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_skeleton = not show_skeleton
                print(f"Skeleton: {'ON' if show_skeleton else 'OFF'}")
            elif key == ord('m'):
                show_metrics = not show_metrics
                print(f"Metrics: {'ON' if show_metrics else 'OFF'}")

    except KeyboardInterrupt:
        print("Stream interrupted")

    finally:
        cv2.destroyAllWindows()
        video_source.disconnect()
        pose_estimator.cleanup()
        print("✅ Live stream ended")


def draw_skeleton(frame, keypoints):
    """Draw skeleton on frame."""
    height, width = frame.shape[:2]

    # Convert to pixel coordinates
    pixel_joints = {}
    for joint_name, joint_data in keypoints.items():
        if joint_data['visibility'] > 0.5:
            x = int(joint_data['x'] * width)
            y = int(joint_data['y'] * height)
            pixel_joints[joint_name] = (x, y)

    # Draw connections
    connections = [
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle')
    ]

    # Draw lines
    for joint1, joint2 in connections:
        if joint1 in pixel_joints and joint2 in pixel_joints:
            cv2.line(frame, pixel_joints[joint1], pixel_joints[joint2], (255, 0, 0), 3)

    # Draw joints
    for joint_name, (x, y) in pixel_joints.items():
        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 8, (0, 0, 0), 2)

    return frame


def draw_confidence_indicator(frame, keypoints):
    """Draw confidence indicator."""
    confidences = [joint['visibility'] for joint in keypoints.values()]
    avg_confidence = sum(confidences) / len(confidences)

    # Choose color
    if avg_confidence > 0.8:
        color = (0, 255, 0)  # Green
        status = "HIGH"
    elif avg_confidence > 0.5:
        color = (0, 255, 255)  # Yellow
        status = "MED"
    else:
        color = (0, 0, 255)  # Red
        status = "LOW"

    # Draw circle
    cv2.circle(frame, (50, 50), 25, color, -1)
    cv2.circle(frame, (50, 50), 25, (0, 0, 0), 2)

    # Draw text
    cv2.putText(frame, status, (85, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_metrics(frame, keypoint_history, frame_count, start_time, gait_analyzer):
    """Draw metrics overlay."""
    height, width = frame.shape[:2]

    # Basic info
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    metrics_text = [
        f"Time: {elapsed_time:.1f}s",
        f"Frames: {frame_count}",
        f"FPS: {fps:.1f}",
        "",
        "=== GAIT DATA ===",
        f"Poses: {len([k for k in keypoint_history if k is not None])}"
    ]

    # Analyze if enough data
    valid_keypoints = [k for k in keypoint_history if k is not None]
    if len(valid_keypoints) > 30:
        try:
            results = gait_analyzer.analyze_walk(valid_keypoints[-90:])  # Last 3 seconds
            metrics_text.extend([
                f"Speed: {results['average_speed']:.4f}",
                f"Duration: {results['duration']:.1f}s"
            ])

            stride_metrics = results['stride_metrics']
            if stride_metrics.get('stride_length_variability', 0) > 0:
                metrics_text.append(f"Variability: {stride_metrics['stride_length_variability']:.3f}")

        except Exception as e:
            metrics_text.append(f"Analysis: {str(e)[:15]}...")

    # Draw background panel
    panel_width = 250
    panel_height = len(metrics_text) * 25 + 20
    panel_x = width - panel_width - 10
    panel_y = 10

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                 (panel_x + panel_width, panel_y + panel_height),
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw text
    for i, text in enumerate(metrics_text):
        y_pos = panel_y + 20 + (i * 20)
        cv2.putText(frame, text, (panel_x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()