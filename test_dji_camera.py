"""
Test DJI camera feeds and select the best one.
"""

import cv2
import time


def test_camera(camera_index, duration=5):
    """Test a camera for the specified duration."""
    print(f"\n🎥 Testing Camera {camera_index}...")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Camera {camera_index} failed to open")
        return False

    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"📹 Properties: {width}x{height} @ {fps} FPS")
    print(f"📺 Showing feed for {duration} seconds...")
    print("👀 Look for DJI camera feed!")
    print("Press 'q' to quit early, 's' to select this camera")

    frame_count = 0
    start_time = time.time()
    selected = False

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        frame_count += 1

        # Add overlay info
        cv2.putText(frame, f"Camera {camera_index}: {width}x{height} @ {fps:.1f}FPS",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 's' to select, 'q' to skip",
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(f'Camera {camera_index} Test', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("⏭️  Skipping this camera")
            break
        elif key == ord('s'):
            print("✅ Camera selected!")
            selected = True
            break

    cap.release()
    cv2.destroyAllWindows()

    actual_fps = frame_count / (time.time() - start_time) if time.time() - start_time > 0 else 0
    print(f"📊 Actual FPS: {actual_fps:.1f}")

    return selected


def main():
    """Test all available cameras to find DJI."""
    print("🎯 DJI Camera Detection Test")
    print("=" * 40)
    print("Look for the DJI camera feed and press 's' to select it")
    print("Or press 'q' to skip to the next camera")

    # Test cameras in order of likelihood (highest FPS first)
    cameras_to_test = [0, 2, 1]  # Camera 0 (30fps) most likely DJI

    selected_camera = None

    for camera_idx in cameras_to_test:
        if test_camera(camera_idx, duration=8):
            selected_camera = camera_idx
            break

    if selected_camera is not None:
        print(f"\n🎉 DJI Camera found at index: {selected_camera}")
        print(f"💡 Use this in your code: VideoSource({selected_camera})")

        # Test with pose detection
        test_with_pose = input("\n🤖 Test with pose detection? (y/n): ").lower().strip()
        if test_with_pose == 'y':
            print("🔄 Starting pose detection test...")
            print("Walk in front of the camera to test pose detection!")

            import sys
            sys.path.append('.')
            from camera.video_source import VideoSource
            from pose.pose_estimator import PoseEstimator

            video_source = VideoSource(selected_camera)
            pose_estimator = PoseEstimator()

            if video_source.connect():
                print("✅ DJI connected! Walk in front of camera...")

                for i in range(100):  # ~3 seconds at 30fps
                    success, frame = video_source.read_frame()
                    if success:
                        pose_success, keypoints = pose_estimator.estimate_pose(frame)

                        if pose_success and pose_estimator.is_pose_valid(keypoints):
                            print(f"✅ Frame {i}: Pose detected!")
                            # Draw simple skeleton
                            frame_with_pose = pose_estimator.draw_skeleton(frame, keypoints)
                            cv2.imshow('DJI + Pose Detection', frame_with_pose)
                        else:
                            cv2.putText(frame, "No pose detected", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.imshow('DJI + Pose Detection', frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print(f"❌ Frame {i}: Failed to read")

                video_source.disconnect()
                cv2.destroyAllWindows()
                print("🎉 DJI + Pose detection test completed!")
            else:
                print("❌ Failed to connect to DJI camera")

    else:
        print("\n❌ No camera selected")
        print("💡 Try adjusting DJI settings or reconnecting USB")


if __name__ == "__main__":
    main()