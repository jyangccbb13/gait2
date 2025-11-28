"""
Demo live gait analysis with synthetic walking data.
Shows what the live stream would look like with real pose data.
"""

import cv2
import numpy as np
import time
import math


def main():
    """Demo live gait analysis with synthetic data."""
    print("🚶‍♂️ Demo Live Gait Analysis")
    print("Shows synthetic walking person with pose overlay and metrics")
    print("Controls: 'q' to quit, 's' to toggle skeleton, 'm' to toggle metrics")

    # Display settings
    show_skeleton = True
    show_metrics = True
    frame_count = 0
    start_time = time.time()

    # Gait tracking
    keypoint_history = []
    speed_history = []

    try:
        while True:
            # Create synthetic frame
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            frame[:] = (40, 40, 40)  # Dark gray background

            # Add title
            cv2.putText(frame, "DEMO: Live Gait Analysis", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Synthetic walking person with pose overlay", (50, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            frame_count += 1
            current_time = time.time() - start_time

            # Generate synthetic walking motion
            keypoints = generate_walking_keypoints(current_time, frame.shape)

            # Store for analysis
            keypoint_history.append(keypoints)
            if len(keypoint_history) > 150:  # Keep last 5 seconds
                keypoint_history = keypoint_history[-150:]

            # Calculate speed
            if len(keypoint_history) >= 2:
                speed = calculate_speed(keypoint_history[-2:], 1/30)  # 30 FPS
                speed_history.append(speed)
                if len(speed_history) > 90:  # Keep last 3 seconds
                    speed_history = speed_history[-90:]

            # Draw skeleton
            if show_skeleton:
                draw_skeleton(frame, keypoints)

            # Draw confidence (always high for demo)
            draw_confidence_indicator(frame, 0.95)

            # Draw metrics
            if show_metrics:
                draw_demo_metrics(frame, keypoint_history, speed_history, frame_count, current_time)

            # Draw walking path
            draw_walking_path(frame, current_time)

            # Show frame
            cv2.imshow('🚶‍♂️ DEMO: Live Gait Analysis (q=quit, s=skeleton, m=metrics)', frame)

            # Handle keys
            key = cv2.waitKey(33) & 0xFF  # ~30 FPS
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_skeleton = not show_skeleton
                print(f"Skeleton: {'ON' if show_skeleton else 'OFF'}")
            elif key == ord('m'):
                show_metrics = not show_metrics
                print(f"Metrics: {'ON' if show_metrics else 'OFF'}")

    except KeyboardInterrupt:
        print("Demo interrupted")

    finally:
        cv2.destroyAllWindows()
        print("✅ Demo ended")


def generate_walking_keypoints(t, frame_shape):
    """Generate synthetic walking keypoints."""
    height, width = frame_shape[:2]

    # Walking parameters
    walking_speed = 0.5  # Cycles per second
    forward_speed = 50   # Pixels per second
    step_height = 30     # Ankle lift in pixels
    sway_amount = 15     # Hip sway in pixels

    # Calculate walking cycle
    cycle = math.sin(2 * math.pi * walking_speed * t)
    left_cycle = math.sin(2 * math.pi * walking_speed * t)
    right_cycle = math.sin(2 * math.pi * walking_speed * t + math.pi)  # Opposite phase

    # Base position (person walks left to right)
    base_x = (forward_speed * t) % (width + 200) - 100
    base_y = height // 2

    # Generate keypoints
    keypoints = {
        'left_hip': {
            'x': (base_x + sway_amount * cycle * 0.3) / width,
            'y': (base_y - 50) / height,
            'visibility': 0.95
        },
        'right_hip': {
            'x': (base_x + 40 + sway_amount * cycle * 0.3) / width,
            'y': (base_y - 50) / height,
            'visibility': 0.95
        },
        'left_knee': {
            'x': (base_x + 10 + left_cycle * 20) / width,
            'y': (base_y + 80) / height,
            'visibility': 0.9
        },
        'right_knee': {
            'x': (base_x + 30 + right_cycle * 20) / width,
            'y': (base_y + 80) / height,
            'visibility': 0.9
        },
        'left_ankle': {
            'x': (base_x + 5 + left_cycle * 30) / width,
            'y': (base_y + 200 - abs(left_cycle) * step_height) / height,
            'visibility': 0.85
        },
        'right_ankle': {
            'x': (base_x + 35 + right_cycle * 30) / width,
            'y': (base_y + 200 - abs(right_cycle) * step_height) / height,
            'visibility': 0.85
        }
    }

    return keypoints


def draw_skeleton(frame, keypoints):
    """Draw skeleton on frame."""
    height, width = frame.shape[:2]

    # Convert to pixel coordinates
    pixel_joints = {}
    for joint_name, joint_data in keypoints.items():
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

    # Draw lines (blue bones)
    for joint1, joint2 in connections:
        if joint1 in pixel_joints and joint2 in pixel_joints:
            cv2.line(frame, pixel_joints[joint1], pixel_joints[joint2], (255, 100, 0), 4)

    # Draw joints (green circles)
    for joint_name, (x, y) in pixel_joints.items():
        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)

        # Label joints
        label = joint_name.replace('_', ' ').title()[:5]
        cv2.putText(frame, label, (x + 15, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def draw_confidence_indicator(frame, confidence):
    """Draw confidence indicator."""
    color = (0, 255, 0)  # Always green for demo
    status = "HIGH"

    cv2.circle(frame, (80, 150), 30, color, -1)
    cv2.circle(frame, (80, 150), 30, (255, 255, 255), 3)

    cv2.putText(frame, status, (120, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"{confidence:.2f}", (120, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_demo_metrics(frame, keypoint_history, speed_history, frame_count, elapsed_time):
    """Draw metrics overlay."""
    height, width = frame.shape[:2]

    # Calculate metrics
    avg_speed = sum(speed_history[-30:]) / len(speed_history[-30:]) if speed_history else 0
    current_speed = speed_history[-1] if speed_history else 0

    # Simulate stride detection
    stride_count = int(elapsed_time * 1.2)  # ~1.2 strides per second

    # Simulate variability
    speed_variability = 0.05 + 0.03 * math.sin(elapsed_time * 0.5)

    metrics_text = [
        "=== LIVE GAIT METRICS ===",
        f"Time: {elapsed_time:.1f}s",
        f"FPS: {frame_count / elapsed_time:.1f}" if elapsed_time > 0 else "FPS: 0",
        "",
        f"Poses Detected: {len([k for k in keypoint_history if k])}/150",
        "",
        "=== GAIT ANALYSIS ===",
        f"Current Speed: {current_speed:.4f}",
        f"Average Speed: {avg_speed:.4f}",
        f"Total Strides: {stride_count}",
        f"Speed Variability: {speed_variability:.3f}",
        "",
        "=== PATTERN QUALITY ===",
        "✅ Consistent stride timing",
        "✅ Normal speed range",
        "✅ Balanced left/right",
        "",
        "Demo Mode: Synthetic Data"
    ]

    # Draw panel
    panel_width = 300
    panel_height = len(metrics_text) * 22 + 20
    panel_x = width - panel_width - 20
    panel_y = 20

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                 (panel_x + panel_width, panel_y + panel_height),
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Border
    cv2.rectangle(frame, (panel_x, panel_y),
                 (panel_x + panel_width, panel_y + panel_height),
                 (100, 150, 255), 2)

    # Text
    for i, text in enumerate(metrics_text):
        y_pos = panel_y + 18 + (i * 18)
        color = (255, 255, 255)

        # Highlight headers
        if "===" in text:
            color = (100, 255, 255)
        elif "✅" in text:
            color = (100, 255, 100)

        cv2.putText(frame, text, (panel_x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_walking_path(frame, t):
    """Draw walking path visualization."""
    height, width = frame.shape[:2]

    # Draw ground line
    ground_y = height // 2 + 220
    cv2.line(frame, (0, ground_y), (width, ground_y), (100, 100, 100), 2)

    # Draw footsteps
    step_distance = 60
    for i in range(0, width + 100, step_distance):
        x = (i + t * 50) % (width + 200) - 100
        if 0 <= x <= width:
            # Alternate left/right footsteps
            if (i // step_distance) % 2 == 0:
                cv2.ellipse(frame, (int(x), ground_y), (15, 8), 0, 0, 360, (0, 150, 255), 2)
                cv2.putText(frame, "L", (int(x) - 5, ground_y - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 255), 1)
            else:
                cv2.ellipse(frame, (int(x), ground_y), (15, 8), 0, 0, 360, (255, 150, 0), 2)
                cv2.putText(frame, "R", (int(x) - 5, ground_y - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)


def calculate_speed(keypoint_sequence, dt):
    """Calculate walking speed between two keypoints."""
    if len(keypoint_sequence) < 2:
        return 0

    kp1, kp2 = keypoint_sequence[-2], keypoint_sequence[-1]
    if not kp1 or not kp2:
        return 0

    # Calculate hip center movement
    hip1_x = (kp1['left_hip']['x'] + kp1['right_hip']['x']) / 2
    hip2_x = (kp2['left_hip']['x'] + kp2['right_hip']['x']) / 2

    dx = abs(hip2_x - hip1_x)
    speed = dx / dt

    return speed


if __name__ == "__main__":
    main()