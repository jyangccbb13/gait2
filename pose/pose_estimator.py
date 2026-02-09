"""
Pose estimation using rtmlib RTMW for gait analysis.
Extracts 2D keypoints for lower body joints including foot landmarks.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple

try:
    from rtmlib import Wholebody
    RTMLIB_AVAILABLE = True
except ImportError:
    RTMLIB_AVAILABLE = False
    print("Warning: rtmlib not available. Using fallback pose estimator.")


# COCO-WholeBody keypoint indices for gait-relevant joints
KEYPOINT_MAP = {
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
    'left_big_toe': 17,
    'left_small_toe': 18,
    'left_heel': 19,
    'right_big_toe': 20,
    'right_small_toe': 21,
    'right_heel': 22,
}


class PoseEstimator:
    """
    Wrapper for rtmlib RTMW to extract lower body keypoints for gait analysis.

    Key joints extracted (12 total):
    - Left/Right Hip, Knee, Ankle
    - Left/Right Big Toe, Small Toe, Heel
    """

    def __init__(self):
        self.keypoint_map = KEYPOINT_MAP

        if RTMLIB_AVAILABLE:
            self.wholebody = Wholebody(
                to_openpose=False,
                mode='balanced',
                backend='onnxruntime',
                device='cpu'
            )
            print("PoseEstimator initialized with rtmlib RTMW")
        else:
            self.wholebody = None
            print("PoseEstimator initialized in FALLBACK mode (no rtmlib)")

    def estimate_pose(self, frame: np.ndarray) -> Tuple[bool, Optional[Dict]]:
        """
        Run pose estimation on a single frame.

        Args:
            frame: Input image as BGR numpy array

        Returns:
            Tuple[bool, Optional[Dict]]: (success, keypoints)
                success: True if pose was detected successfully
                keypoints: Dictionary with normalized 2D coordinates for each joint
                          Format: {'joint_name': {'x': float, 'y': float, 'visibility': float}}
        """
        if frame is None:
            return False, None

        try:
            if not RTMLIB_AVAILABLE or self.wholebody is None:
                return False, None

            keypoints, scores = self.wholebody(frame)

            if keypoints is None or len(keypoints) == 0:
                return False, None

            # Take person 0
            person_kps = keypoints[0]   # shape (133, 2) pixel coords
            person_scores = scores[0]   # shape (133,)

            h, w = frame.shape[:2]

            # Extract gait-relevant keypoints, normalize to 0-1
            result = {}
            for joint_name, idx in self.keypoint_map.items():
                result[joint_name] = {
                    'x': float(person_kps[idx][0] / w),
                    'y': float(person_kps[idx][1] / h),
                    'visibility': float(person_scores[idx])
                }

            return True, result

        except Exception as e:
            print(f"Error in pose estimation: {e}")
            return False, None

    def is_pose_valid(self, keypoints: Dict, min_visibility: float = 0.3) -> bool:
        """
        Check if detected pose is valid for gait analysis.
        Requires hips, ankles, and at least one heel above threshold.
        """
        if keypoints is None:
            return False

        required_joints = ['left_hip', 'right_hip', 'left_ankle', 'right_ankle']
        for joint in required_joints:
            if joint not in keypoints:
                return False
            if keypoints[joint]['visibility'] < min_visibility:
                return False

        # At least one heel must be visible
        heel_visible = (
            keypoints.get('left_heel', {}).get('visibility', 0) >= min_visibility or
            keypoints.get('right_heel', {}).get('visibility', 0) >= min_visibility
        )
        if not heel_visible:
            return False

        return True

    def get_hip_center(self, keypoints: Dict) -> Optional[Tuple[float, float]]:
        """
        Calculate center point between left and right hips.

        Returns:
            Optional[Tuple[float, float]]: (x, y) normalized coordinates of hip center
        """
        if not self.is_pose_valid(keypoints):
            return None

        left_hip = keypoints['left_hip']
        right_hip = keypoints['right_hip']

        center_x = (left_hip['x'] + right_hip['x']) / 2
        center_y = (left_hip['y'] + right_hip['y']) / 2

        return (center_x, center_y)

    def get_ankle_positions(self, keypoints: Dict) -> Optional[Dict]:
        """
        Get ankle positions for stride detection.

        Returns:
            Optional[Dict]: Ankle positions with format:
                           {'left': (x, y), 'right': (x, y)}
        """
        if not self.is_pose_valid(keypoints):
            return None

        left_ankle = keypoints['left_ankle']
        right_ankle = keypoints['right_ankle']

        return {
            'left': (left_ankle['x'], left_ankle['y']),
            'right': (right_ankle['x'], right_ankle['y'])
        }

    def draw_skeleton(self, frame: np.ndarray, keypoints: Dict) -> np.ndarray:
        """
        Draw pose skeleton overlay with joint labels.

        Args:
            frame: Input BGR frame
            keypoints: Keypoints from estimate_pose()

        Returns:
            np.ndarray: Frame with skeleton overlay
        """
        if keypoints is None:
            return frame

        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Convert normalized coordinates to pixel coordinates
        pixel_kps = {}
        for joint_name, joint_data in keypoints.items():
            x = int(joint_data['x'] * w)
            y = int(joint_data['y'] * h)
            pixel_kps[joint_name] = (x, y)

        # Draw connections
        connections = [
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            # Foot connections
            ('left_ankle', 'left_heel'),
            ('left_ankle', 'left_big_toe'),
            ('left_big_toe', 'left_small_toe'),
            ('right_ankle', 'right_heel'),
            ('right_ankle', 'right_big_toe'),
            ('right_big_toe', 'right_small_toe'),
        ]

        for joint1, joint2 in connections:
            if joint1 in pixel_kps and joint2 in pixel_kps:
                pt1 = pixel_kps[joint1]
                pt2 = pixel_kps[joint2]
                cv2.line(overlay, pt1, pt2, (0, 255, 255), 4)

        # Joint colors by type
        joint_colors = {
            'left_hip': (255, 0, 0), 'right_hip': (255, 0, 0),
            'left_knee': (0, 255, 0), 'right_knee': (0, 255, 0),
            'left_ankle': (0, 0, 255), 'right_ankle': (0, 0, 255),
            'left_big_toe': (255, 0, 255), 'right_big_toe': (255, 0, 255),
            'left_small_toe': (255, 0, 255), 'right_small_toe': (255, 0, 255),
            'left_heel': (0, 255, 255), 'right_heel': (0, 255, 255),
        }

        joint_labels = {
            'left_hip': 'L Hip', 'right_hip': 'R Hip',
            'left_knee': 'L Knee', 'right_knee': 'R Knee',
            'left_ankle': 'L Ankle', 'right_ankle': 'R Ankle',
            'left_big_toe': 'L Toe', 'right_big_toe': 'R Toe',
            'left_small_toe': 'L SToe', 'right_small_toe': 'R SToe',
            'left_heel': 'L Heel', 'right_heel': 'R Heel',
        }

        for joint_name, (x, y) in pixel_kps.items():
            color = joint_colors.get(joint_name, (255, 255, 255))

            cv2.circle(overlay, (x, y), 8, color, -1)
            cv2.circle(overlay, (x, y), 8, (255, 255, 255), 2)

            label = joint_labels.get(joint_name, joint_name)
            label_pos = (x + 15, y - 10)
            cv2.putText(overlay, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 1, cv2.LINE_AA)

        return overlay

    def cleanup(self):
        """No-op for interface compatibility."""
        pass


def test_pose_estimator():
    """Test function to verify pose estimation functionality."""
    from camera.video_source import VideoSource

    print("Testing pose estimator...")

    pose_estimator = PoseEstimator()

    with VideoSource(0) as video:
        if not video.is_connected:
            print("No webcam found for testing")
            return

        print("Press 'q' to quit the test")

        while True:
            success, frame = video.read_frame()
            if not success:
                break

            pose_success, keypoints = pose_estimator.estimate_pose(frame)

            if pose_success:
                valid = pose_estimator.is_pose_valid(keypoints)
                print(f"Pose detected - Valid: {valid}, Joints: {len(keypoints)}")

                frame_with_pose = pose_estimator.draw_skeleton(frame, keypoints)

                hip_center = pose_estimator.get_hip_center(keypoints)
                if hip_center:
                    print(f"Hip center: ({hip_center[0]:.3f}, {hip_center[1]:.3f})")

                cv2.imshow('Pose Test', frame_with_pose)
            else:
                cv2.imshow('Pose Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    pose_estimator.cleanup()
    print("Pose estimator test completed")


if __name__ == "__main__":
    test_pose_estimator()
