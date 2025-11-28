"""
Live gait analysis with DJI Osmo 5 camera.
Real-time pose detection and gait metrics from your DJI camera.
"""

import cv2
import numpy as np
import time
from camera.video_source import VideoSource
from pose.pose_estimator import PoseEstimator
from gait.gait_features import GaitAnalyzer


def main():
    """Start live gait analysis with DJI Osmo 5."""
    print("🎥 DJI Osmo 5 Live Gait Analysis")
    print("=" * 50)
    print("📱 Make sure DJI is connected in USB webcam mode")
    print("🚶‍♂️ Walk in front of camera for gait analysis")
    print("Controls: 'q' to quit, 's' to toggle skeleton, 'm' to toggle metrics, 'r' to reset")

    # Initialize components with DJI camera (index 0)
    video_source = VideoSource(0)  # DJI Osmo 5
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
        print("❌ Error: Could not connect to DJI camera")
        print("💡 Check USB connection and camera mode")
        return

    print("✅ DJI camera connected!")
    print("📹 Camera properties:", video_source.get_properties())

    try:
        while True:
            success, frame = video_source.read_frame()
            if not success:
                print("❌ Failed to read frame from DJI")
                break

            frame_count += 1
            processed_frame = frame.copy()

            # Run pose estimation
            pose_success, keypoints = pose_estimator.estimate_pose(frame)

            if pose_success and pose_estimator.is_pose_valid(keypoints):
                # Store keypoints
                keypoint_history.append(keypoints)
                pose_detected = True

                print(f"✅ Frame {frame_count}: Pose detected!")
            else:
                keypoint_history.append(None)
                pose_detected = False

            # Keep only recent keypoints (last 5 seconds)
            max_history = 150
            if len(keypoint_history) > max_history:
                keypoint_history = keypoint_history[-max_history:]

            # Draw skeleton overlay
            if show_skeleton and pose_detected:
                processed_frame = draw_skeleton(processed_frame, keypoints)

            # Draw confidence indicator
            if pose_detected:
                draw_confidence_indicator(processed_frame, keypoints)
            else:
                # Red indicator for no pose
                cv2.circle(processed_frame, (50, 50), 25, (0, 0, 255), -1)
                cv2.putText(processed_frame, "NO POSE", (85, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw metrics
            if show_metrics:
                draw_metrics(processed_frame, keypoint_history, frame_count, start_time, gait_analyzer)

            # DJI camera info overlay
            cv2.putText(processed_frame, "DJI Osmo 5 - Live Gait Analysis", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Show frame
            cv2.imshow('🎥 DJI Live Gait Analysis (q=quit, s=skeleton, m=metrics, r=reset)', processed_frame)

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
            elif key == ord('r'):
                keypoint_history = []
                frame_count = 0
                start_time = time.time()
                print("🔄 Reset gait tracking")

    except KeyboardInterrupt:
        print("Stream interrupted")

    finally:
        cv2.destroyAllWindows()
        video_source.disconnect()
        pose_estimator.cleanup()
        print("✅ DJI live stream ended")


def draw_skeleton(frame, keypoints):
    """Draw skeleton on frame with DJI camera."""
    height, width = frame.shape[:2]

    # Convert to pixel coordinates
    pixel_joints = {}
    for joint_name, joint_data in keypoints.items():
        if joint_data['visibility'] > 0.5:
            x = int(joint_data['x'] * width)
            y = int(joint_data['y'] * height)
            pixel_joints[joint_name] = (x, y)

    # Draw connections (bright colors for DJI video)
    connections = [
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle')
    ]

    # Draw lines (cyan for visibility)
    for joint1, joint2 in connections:
        if joint1 in pixel_joints and joint2 in pixel_joints:
            cv2.line(frame, pixel_joints[joint1], pixel_joints[joint2], (255, 255, 0), 4)

    # Draw joints (bright green)
    for joint_name, (x, y) in pixel_joints.items():
        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 10, (255, 255, 255), 3)

    return frame


def draw_confidence_indicator(frame, keypoints):
    """Draw confidence indicator for DJI feed."""
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
    cv2.circle(frame, (50, 50), 25, (255, 255, 255), 2)

    # Draw text
    cv2.putText(frame, status, (85, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_metrics(frame, keypoint_history, frame_count, start_time, gait_analyzer):
    """Draw real-time gait metrics on DJI feed."""
    height, width = frame.shape[:2]

    # Basic info
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    metrics_text = [
        "=== DJI LIVE GAIT ===",
        f"Time: {elapsed_time:.1f}s",
        f"FPS: {fps:.1f}",
        f"Frames: {frame_count}",
        "",
        "=== POSE DATA ===",
        f"Valid Poses: {len([k for k in keypoint_history if k is not None])}"
    ]

    # Analyze if enough data
    valid_keypoints = [k for k in keypoint_history if k is not None]
    if len(valid_keypoints) > 30:
        try:
            results = gait_analyzer.analyze_walk(valid_keypoints[-90:])  # Last 3 seconds
            metrics_text.extend([
                "",
                "=== GAIT METRICS ===",
                f"Speed: {results['average_speed']:.4f}",
                f"Duration: {results['duration']:.1f}s"
            ])

            stride_metrics = results['stride_metrics']
            if stride_metrics.get('stride_length_variability', 0) > 0:
                metrics_text.append(f"Variability: {stride_metrics['stride_length_variability']:.3f}")

            # Show stride events
            stride_events = results['stride_events']
            left_strides = len(stride_events['left'])
            right_strides = len(stride_events['right'])
            if left_strides + right_strides > 0:
                metrics_text.append(f"Strides: L{left_strides} R{right_strides}")

        except Exception as e:
            metrics_text.extend([
                "",
                "=== ANALYSIS ===",
                f"Processing..."
            ])

    # Draw metrics panel with DJI branding
    panel_width = 280
    panel_height = len(metrics_text) * 25 + 20
    panel_x = width - panel_width - 10
    panel_y = 80  # Below DJI title

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                 (panel_x + panel_width, panel_y + panel_height),
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # DJI-style border (cyan)
    cv2.rectangle(frame, (panel_x, panel_y),
                 (panel_x + panel_width, panel_y + panel_height),
                 (255, 255, 0), 2)

    # Draw text
    for i, text in enumerate(metrics_text):
        y_pos = panel_y + 20 + (i * 20)
        color = (0, 255, 255) if "===" in text else (255, 255, 255)
        cv2.putText(frame, text, (panel_x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()