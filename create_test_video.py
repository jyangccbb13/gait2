"""
Create a test walking video with synthetic person for gait analysis testing.
"""

import cv2
import numpy as np
import math


def create_test_video(output_path="test_walking.mp4", duration_seconds=10, fps=30):
    """Create a synthetic walking video for testing."""
    width, height = 640, 480
    total_frames = int(duration_seconds * fps)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creating test video: {output_path}")
    print(f"Duration: {duration_seconds}s at {fps} FPS ({total_frames} frames)")

    for frame_idx in range(total_frames):
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark gray background

        # Add grid for reference
        for x in range(0, width, 50):
            cv2.line(frame, (x, 0), (x, height), (80, 80, 80), 1)
        for y in range(0, height, 50):
            cv2.line(frame, (0, y), (width, y), (80, 80, 80), 1)

        # Add title
        cv2.putText(frame, "Test Walking Video", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Time in seconds
        t = frame_idx / fps

        # Create walking person
        create_walking_person(frame, t, width, height)

        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out.write(frame)

        # Progress
        if frame_idx % 30 == 0:
            print(f"  Progress: {frame_idx}/{total_frames} frames")

    out.release()
    print(f"✅ Test video created: {output_path}")


def create_walking_person(frame, t, width, height):
    """Draw a walking stick figure on the frame."""
    # Walking parameters
    walking_speed = 1.5  # cycles per second
    forward_speed = 60   # pixels per second
    step_height = 25     # ankle lift
    sway = 10           # body sway

    # Walking cycle
    cycle = math.sin(2 * math.pi * walking_speed * t)
    left_cycle = math.sin(2 * math.pi * walking_speed * t)
    right_cycle = math.sin(2 * math.pi * walking_speed * t + math.pi)

    # Person position (walks left to right)
    person_x = (forward_speed * t) % (width + 200) - 100
    person_y = height // 2

    # Body parts positions
    head = (int(person_x + sway * cycle * 0.2), int(person_y - 120))
    torso_top = (int(person_x + sway * cycle * 0.3), int(person_y - 80))
    torso_bottom = (int(person_x + sway * cycle * 0.3), int(person_y - 20))

    left_hip = (int(person_x - 15 + sway * cycle * 0.3), int(person_y - 20))
    right_hip = (int(person_x + 15 + sway * cycle * 0.3), int(person_y - 20))

    left_knee = (int(person_x - 15 + left_cycle * 15), int(person_y + 40))
    right_knee = (int(person_x + 15 + right_cycle * 15), int(person_y + 40))

    left_ankle = (int(person_x - 10 + left_cycle * 25), int(person_y + 100 - abs(left_cycle) * step_height))
    right_ankle = (int(person_x + 10 + right_cycle * 25), int(person_y + 100 - abs(right_cycle) * step_height))

    # Draw person (only if visible)
    if -50 < person_x < width + 50:
        # Head
        cv2.circle(frame, head, 15, (255, 200, 150), -1)
        cv2.circle(frame, head, 15, (255, 255, 255), 2)

        # Torso
        cv2.line(frame, torso_top, torso_bottom, (100, 150, 255), 8)

        # Arms (simple)
        arm_swing = cycle * 20
        left_hand = (int(torso_top[0] - 30 - arm_swing), int(torso_top[1] + 30))
        right_hand = (int(torso_top[0] + 30 + arm_swing), int(torso_top[1] + 30))

        cv2.line(frame, torso_top, left_hand, (255, 150, 100), 4)
        cv2.line(frame, torso_top, right_hand, (255, 150, 100), 4)

        # Legs - Hip to knee
        cv2.line(frame, left_hip, left_knee, (100, 150, 255), 6)
        cv2.line(frame, right_hip, right_knee, (100, 150, 255), 6)

        # Legs - Knee to ankle
        cv2.line(frame, left_knee, left_ankle, (100, 150, 255), 6)
        cv2.line(frame, right_knee, right_ankle, (100, 150, 255), 6)

        # Hip line
        cv2.line(frame, left_hip, right_hip, (100, 150, 255), 6)

        # Joint circles (these are what pose estimation would detect)
        joints = [
            (left_hip, "L_HIP"),
            (right_hip, "R_HIP"),
            (left_knee, "L_KNEE"),
            (right_knee, "R_KNEE"),
            (left_ankle, "L_ANKLE"),
            (right_ankle, "R_ANKLE")
        ]

        for (x, y), label in joints:
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)

        # Add ground line
        ground_y = person_y + 120
        cv2.line(frame, (0, ground_y), (width, ground_y), (150, 150, 150), 2)

        # Footprints
        if abs(left_cycle) < 0.3:  # Foot on ground
            cv2.ellipse(frame, left_ankle, (20, 10), 0, 0, 360, (255, 100, 100), 2)

        if abs(right_cycle) < 0.3:  # Foot on ground
            cv2.ellipse(frame, right_ankle, (20, 10), 0, 0, 360, (100, 100, 255), 2)


if __name__ == "__main__":
    create_test_video("test_walking.mp4", duration_seconds=10, fps=30)
    print("\n🎬 Test video created!")
    print("Now run: python3 video_analyzer.py")
    print("And use 'test_walking.mp4' as the input video")