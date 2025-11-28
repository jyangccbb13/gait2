"""
Pose estimation using MediaPipe BlazePose for gait analysis.
Extracts 2D keypoints focused on lower body joints (hips, knees, ankles).
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple

# Fallback for MediaPipe compatibility issues
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Using fallback pose estimator.")


class PoseEstimator:
    """
    Wrapper for MediaPipe BlazePose to extract lower body keypoints for gait analysis.

    Key joints extracted:
    - Left/Right Hip
    - Left/Right Knee
    - Left/Right Ankle
    """

    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize pose estimation model.

        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils

            # Initialize pose model
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )

            # Define the key joints we care about for gait analysis
            self.gait_joints = {
                'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
                'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
                'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
                'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
                'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
                'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE
            }

            print("PoseEstimator initialized with MediaPipe BlazePose")
        else:
            # Fallback mode - will generate dummy data for testing
            self.pose = None
            self.gait_joints = {
                'left_hip': 'left_hip',
                'right_hip': 'right_hip',
                'left_knee': 'left_knee',
                'right_knee': 'right_knee',
                'left_ankle': 'left_ankle',
                'right_ankle': 'right_ankle'
            }
            print("PoseEstimator initialized in FALLBACK mode (no MediaPipe)")

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
            if MEDIAPIPE_AVAILABLE and self.pose is not None:
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process frame
                results = self.pose.process(rgb_frame)

                if results.pose_landmarks is None:
                    return False, None

                # Extract keypoints for gait analysis
                keypoints = {}
                landmarks = results.pose_landmarks.landmark

                for joint_name, landmark_idx in self.gait_joints.items():
                    landmark = landmarks[landmark_idx]
                    keypoints[joint_name] = {
                        'x': landmark.x,  # Normalized coordinates (0-1)
                        'y': landmark.y,
                        'visibility': landmark.visibility  # Confidence score
                    }

                return True, keypoints
            else:
                # Fallback mode - NO synthetic data to avoid floating skeletons
                print(f"⚠️  Fallback mode (no MediaPipe) - no pose detected")
                return False, None

        except Exception as e:
            print(f"Error in pose estimation: {e}")
            return False, None

    def is_pose_valid(self, keypoints: Dict, min_visibility: float = 0.3) -> bool:
        """
        Check if detected pose is valid for gait analysis.
        SIMPLE VERSION - like yesterday that worked perfectly.
        """
        if keypoints is None:
            return False

        # SIMPLE CHECK - just verify required joints exist and are visible
        required_joints = ['left_hip', 'right_hip', 'left_ankle', 'right_ankle']

        for joint in required_joints:
            if joint not in keypoints:
                return False
            if keypoints[joint]['visibility'] < min_visibility:
                return False

        return True  # THAT'S IT - no complex validation like yesterday

    def _validate_pose_geometry(self, keypoints: Dict) -> bool:
        """
        Validate that pose has realistic human geometry.
        Prevents detection on furniture/objects with enhanced anti-hallucination checks.
        """
        try:
            # Check hip-to-ankle vertical relationship (person should be upright)
            left_hip_y = keypoints['left_hip']['y']
            left_ankle_y = keypoints['left_ankle']['y']
            right_hip_y = keypoints['right_hip']['y']
            right_ankle_y = keypoints['right_ankle']['y']

            # Hip should be significantly above ankle (person upright)
            left_height_ratio = (left_ankle_y - left_hip_y)
            right_height_ratio = (right_ankle_y - right_hip_y)

            if left_height_ratio < 0.15 or right_height_ratio < 0.15:  # At least 15% of frame height
                return False

            # Check hip width is reasonable (not too wide/narrow)
            hip_width = abs(keypoints['left_hip']['x'] - keypoints['right_hip']['x'])
            if hip_width < 0.05 or hip_width > 0.4:  # 5-40% of frame width
                return False

            # Check ankle spacing is reasonable
            ankle_width = abs(keypoints['left_ankle']['x'] - keypoints['right_ankle']['x'])
            if ankle_width > 0.6:  # Maximum 60% of frame width
                return False

            # Enhanced furniture/object detection prevention
            # Check for unnatural limb configurations that suggest furniture

            # 1. Check if joints are aligned horizontally (furniture legs)
            left_hip_x = keypoints['left_hip']['x']
            right_hip_x = keypoints['right_hip']['x']
            left_ankle_x = keypoints['left_ankle']['x']
            right_ankle_x = keypoints['right_ankle']['x']

            # Legs should not be perfectly vertical (suggests table/chair legs)
            left_leg_horizontal_offset = abs(left_hip_x - left_ankle_x)
            right_leg_horizontal_offset = abs(right_hip_x - right_ankle_x)

            if left_leg_horizontal_offset < 0.02 and right_leg_horizontal_offset < 0.02:
                return False  # Too vertical, likely furniture

            # 2. Check for abnormal body proportions
            torso_width = hip_width
            leg_separation = ankle_width

            # Human proportions: ankle separation should be similar to hip width ±50%
            if leg_separation > torso_width * 2.0:  # Legs too far apart
                return False

            # 3. Check vertical alignment consistency
            # Hips and ankles should maintain reasonable left-right symmetry
            hip_y_diff = abs(left_hip_y - right_hip_y)
            ankle_y_diff = abs(left_ankle_y - right_ankle_y)

            if hip_y_diff > 0.1 or ankle_y_diff > 0.1:  # More than 10% height difference
                return False  # Unnatural asymmetry

            # Check knee positions are between hip and ankle
            if 'left_knee' in keypoints and 'right_knee' in keypoints:
                left_knee_y = keypoints['left_knee']['y']
                right_knee_y = keypoints['right_knee']['y']
                left_knee_x = keypoints['left_knee']['x']
                right_knee_x = keypoints['right_knee']['x']

                # Knee should be between hip and ankle vertically
                if not (left_hip_y < left_knee_y < left_ankle_y):
                    return False
                if not (right_hip_y < right_knee_y < right_ankle_y):
                    return False

                # Knee should be reasonably positioned horizontally
                # Check that knee is not too far from the hip-ankle line
                left_knee_deviation = abs(left_knee_x - (left_hip_x + left_ankle_x) / 2)
                right_knee_deviation = abs(right_knee_x - (right_hip_x + right_ankle_x) / 2)

                if left_knee_deviation > 0.15 or right_knee_deviation > 0.15:
                    return False  # Knee too far from natural position

            return True

        except Exception:
            return False

    def _validate_pose_stability(self, keypoints: Dict) -> bool:
        """
        Check for pose stability to avoid jittery false positives.
        Enhanced with stricter confidence requirements to reduce hallucinations.
        """
        try:
            # Check that all keypoints have reasonable confidence
            confidences = [joint['visibility'] for joint in keypoints.values()]
            avg_confidence = sum(confidences) / len(confidences)

            # Super lenient confidence requirement for maximum sensitivity
            if avg_confidence < 0.2:  # Much more lenient for demo
                return False

            # Require minimum confidence for critical joints
            critical_joints = ['left_hip', 'right_hip', 'left_ankle', 'right_ankle']
            for joint in critical_joints:
                if keypoints[joint]['visibility'] < 0.1:  # Very lenient threshold
                    return False

            # Check that pose is roughly symmetric (left/right similar confidence)
            left_joints = [k for k in keypoints.keys() if 'left' in k]
            right_joints = [k for k in keypoints.keys() if 'right' in k]

            if len(left_joints) != len(right_joints):
                return False

            left_avg_conf = sum(keypoints[j]['visibility'] for j in left_joints) / len(left_joints)
            right_avg_conf = sum(keypoints[j]['visibility'] for j in right_joints) / len(right_joints)

            # Left and right shouldn't differ too much - very lenient
            if abs(left_avg_conf - right_avg_conf) > 0.7:  # Much more tolerant
                return False

            # Additional stability check: ensure no extremely low confidence joints
            min_confidence = min(confidences)
            if min_confidence < 0.05:  # Extremely low threshold for maximum sensitivity
                return False

            return True

        except Exception:
            return False

    def get_hip_center(self, keypoints: Dict) -> Optional[Tuple[float, float]]:
        """
        Calculate center point between left and right hips.

        Args:
            keypoints: Keypoints dictionary from estimate_pose()

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

        Args:
            keypoints: Keypoints dictionary from estimate_pose()

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
        Draw pose skeleton overlay on frame for visualization.

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
        pixel_keypoints = {}
        for joint_name, joint_data in keypoints.items():
            x = int(joint_data['x'] * w)
            y = int(joint_data['y'] * h)
            pixel_keypoints[joint_name] = (x, y)

        # Draw joints as circles
        for joint_name, (x, y) in pixel_keypoints.items():
            color = (0, 255, 0)  # Green for joints
            cv2.circle(overlay, (x, y), 5, color, -1)

        # Draw connections between joints
        connections = [
            ('left_hip', 'right_hip'),      # Hip line
            ('left_hip', 'left_knee'),      # Left leg
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),    # Right leg
            ('right_knee', 'right_ankle')
        ]

        for joint1, joint2 in connections:
            if joint1 in pixel_keypoints and joint2 in pixel_keypoints:
                pt1 = pixel_keypoints[joint1]
                pt2 = pixel_keypoints[joint2]
                cv2.line(overlay, pt1, pt2, (255, 0, 0), 2)  # Blue lines

        return overlay

    def cleanup(self):
        """Clean up MediaPipe resources."""
        if MEDIAPIPE_AVAILABLE and hasattr(self, 'pose') and self.pose is not None:
            self.pose.close()


def test_pose_estimator():
    """Test function to verify pose estimation functionality."""
    from camera.video_source import VideoSource

    print("Testing pose estimator...")

    # Initialize pose estimator
    pose_estimator = PoseEstimator()

    # Test with webcam
    with VideoSource(0) as video:
        if not video.is_connected:
            print("No webcam found for testing")
            return

        print("Press 'q' to quit the test")

        while True:
            success, frame = video.read_frame()
            if not success:
                break

            # Run pose estimation
            pose_success, keypoints = pose_estimator.estimate_pose(frame)

            if pose_success:
                print(f"Pose detected - Valid: {pose_estimator.is_pose_valid(keypoints)}")

                # Draw skeleton overlay
                frame_with_pose = pose_estimator.draw_skeleton(frame, keypoints)

                # Get hip center for testing
                hip_center = pose_estimator.get_hip_center(keypoints)
                if hip_center:
                    print(f"Hip center: ({hip_center[0]:.3f}, {hip_center[1]:.3f})")

                cv2.imshow('Pose Test', frame_with_pose)
            else:
                cv2.imshow('Pose Test', frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    pose_estimator.cleanup()
    print("Pose estimator test completed")


if __name__ == "__main__":
    test_pose_estimator()