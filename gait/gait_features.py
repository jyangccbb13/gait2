"""
Gait feature extraction from pose keypoint sequences.
Computes gait speed, stride detection, stride length, and variability metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class GaitAnalyzer:
    """
    Analyzes gait patterns from sequence of pose keypoints.

    Computes:
    - Gait speed from hip displacement
    - Stride events from ankle movement
    - Stride length estimates
    - Temporal and spatial variability
    """

    def __init__(self, fps: float = 30.0):
        """
        Initialize gait analyzer.

        Args:
            fps: Frames per second of the video source
        """
        self.fps = fps
        self.frame_duration = 1.0 / fps

    def compute_gait_speed(self,
                          keypoint_sequence: List[Dict],
                          pixel_to_meter_ratio: float = 0.001) -> Tuple[List[float], float]:
        """
        Compute gait speed over time from hip center displacement.

        Args:
            keypoint_sequence: List of keypoint dictionaries from pose estimation
            pixel_to_meter_ratio: Conversion factor from pixels to meters (approximate)

        Returns:
            Tuple[List[float], float]: (speed_over_time, average_speed)
                speed_over_time: Speed at each frame in m/s
                average_speed: Average walking speed in m/s
        """
        hip_centers = []
        timestamps = []

        # Extract hip centers over time
        for i, keypoints in enumerate(keypoint_sequence):
            if keypoints is None:
                continue

            if 'left_hip' in keypoints and 'right_hip' in keypoints:
                left_hip = keypoints['left_hip']
                right_hip = keypoints['right_hip']

                # Calculate hip center
                center_x = (left_hip['x'] + right_hip['x']) / 2
                center_y = (left_hip['y'] + right_hip['y']) / 2

                hip_centers.append((center_x, center_y))
                timestamps.append(i * self.frame_duration)

        if len(hip_centers) < 2:
            return [], 0.0

        # Calculate instantaneous speeds
        speeds = []
        for i in range(1, len(hip_centers)):
            # Calculate displacement
            dx = hip_centers[i][0] - hip_centers[i-1][0]
            dy = hip_centers[i][1] - hip_centers[i-1][1]

            # Convert to meters (very approximate)
            displacement = np.sqrt(dx**2 + dy**2) * pixel_to_meter_ratio

            # Calculate speed
            speed = displacement / self.frame_duration
            speeds.append(speed)

        # Smooth speeds to reduce noise
        if len(speeds) > 5:
            speeds = self._smooth_signal(speeds, window_size=5)

        average_speed = np.mean(speeds) if speeds else 0.0

        return speeds, average_speed

    def detect_stride_events(self,
                           keypoint_sequence: List[Dict],
                           min_stride_time: float = 0.3) -> Dict[str, List[int]]:
        """
        Detect stride events (heel strikes) from ankle movement patterns.

        Args:
            keypoint_sequence: List of keypoint dictionaries
            min_stride_time: Minimum time between strides in seconds

        Returns:
            Dict[str, List[int]]: Stride event frame indices for each foot
                                 {'left': [frame_idx, ...], 'right': [frame_idx, ...]}
        """
        left_ankle_y = []
        right_ankle_y = []
        valid_frames = []

        # Extract ankle y-coordinates over time
        for i, keypoints in enumerate(keypoint_sequence):
            if keypoints is None:
                continue

            if ('left_ankle' in keypoints and 'right_ankle' in keypoints):
                left_ankle_y.append(keypoints['left_ankle']['y'])
                right_ankle_y.append(keypoints['right_ankle']['y'])
                valid_frames.append(i)

        if len(left_ankle_y) < 10:
            return {'left': [], 'right': []}

        # Convert to numpy arrays
        left_ankle_y = np.array(left_ankle_y)
        right_ankle_y = np.array(right_ankle_y)

        # Find peaks (maximum y values indicate stance phase)
        min_distance = int(min_stride_time * self.fps)

        # Find stride events for left foot
        left_peaks, _ = find_peaks(left_ankle_y,
                                  distance=min_distance,
                                  prominence=0.01)

        # Find stride events for right foot
        right_peaks, _ = find_peaks(right_ankle_y,
                                   distance=min_distance,
                                   prominence=0.01)

        # Convert back to original frame indices
        left_stride_frames = [valid_frames[i] for i in left_peaks if i < len(valid_frames)]
        right_stride_frames = [valid_frames[i] for i in right_peaks if i < len(valid_frames)]

        return {
            'left': left_stride_frames,
            'right': right_stride_frames
        }

    def compute_stride_metrics(self,
                             keypoint_sequence: List[Dict],
                             stride_events: Dict[str, List[int]]) -> Dict[str, float]:
        """
        Compute stride length and temporal metrics.

        Args:
            keypoint_sequence: List of keypoint dictionaries
            stride_events: Stride event frame indices from detect_stride_events()

        Returns:
            Dict[str, float]: Stride metrics including:
                - stride_length_left/right: Average stride length (normalized units)
                - stride_time_left/right: Average stride time in seconds
                - stride_length_variability: Coefficient of variation for stride length
                - stride_time_variability: Coefficient of variation for stride time
        """
        metrics = {}

        for foot in ['left', 'right']:
            events = stride_events[foot]
            if len(events) < 2:
                metrics[f'stride_length_{foot}'] = 0.0
                metrics[f'stride_time_{foot}'] = 0.0
                continue

            # Calculate stride times
            stride_times = []
            stride_lengths = []

            for i in range(1, len(events)):
                # Stride time
                time_diff = (events[i] - events[i-1]) * self.frame_duration
                stride_times.append(time_diff)

                # Stride length (approximate from hip displacement)
                start_frame = events[i-1]
                end_frame = events[i]

                if (start_frame < len(keypoint_sequence) and
                    end_frame < len(keypoint_sequence) and
                    keypoint_sequence[start_frame] is not None and
                    keypoint_sequence[end_frame] is not None):

                    start_keypoints = keypoint_sequence[start_frame]
                    end_keypoints = keypoint_sequence[end_frame]

                    if ('left_hip' in start_keypoints and 'left_hip' in end_keypoints and
                        'right_hip' in start_keypoints and 'right_hip' in end_keypoints):

                        # Hip center at start
                        start_x = (start_keypoints['left_hip']['x'] + start_keypoints['right_hip']['x']) / 2
                        # Hip center at end
                        end_x = (end_keypoints['left_hip']['x'] + end_keypoints['right_hip']['x']) / 2

                        stride_length = abs(end_x - start_x)
                        stride_lengths.append(stride_length)

            # Store metrics
            metrics[f'stride_time_{foot}'] = np.mean(stride_times) if stride_times else 0.0
            metrics[f'stride_length_{foot}'] = np.mean(stride_lengths) if stride_lengths else 0.0

            # Store raw data for variability calculation
            metrics[f'_stride_times_{foot}'] = stride_times
            metrics[f'_stride_lengths_{foot}'] = stride_lengths

        # Calculate variability metrics (coefficient of variation)
        all_stride_times = (metrics.get('_stride_times_left', []) +
                           metrics.get('_stride_times_right', []))
        all_stride_lengths = (metrics.get('_stride_lengths_left', []) +
                            metrics.get('_stride_lengths_right', []))

        metrics['stride_time_variability'] = self._coefficient_of_variation(all_stride_times)
        metrics['stride_length_variability'] = self._coefficient_of_variation(all_stride_lengths)

        # Clean up temporary data
        keys_to_remove = [k for k in metrics.keys() if k.startswith('_')]
        for key in keys_to_remove:
            del metrics[key]

        return metrics

    def analyze_walk(self, keypoint_sequence: List[Dict]) -> Dict[str, any]:
        """
        Complete gait analysis pipeline for a walking sequence.

        Args:
            keypoint_sequence: List of keypoint dictionaries from pose estimation

        Returns:
            Dict[str, any]: Complete gait analysis results including:
                - speeds: Speed over time
                - average_speed: Mean walking speed
                - stride_events: Frame indices of stride events
                - stride_metrics: Stride length and timing metrics
                - duration: Total walking duration in seconds
        """
        # Compute gait speed
        speeds, average_speed = self.compute_gait_speed(keypoint_sequence)

        # Detect stride events
        stride_events = self.detect_stride_events(keypoint_sequence)

        # Compute stride metrics
        stride_metrics = self.compute_stride_metrics(keypoint_sequence, stride_events)

        # Calculate total duration
        duration = len(keypoint_sequence) * self.frame_duration

        return {
            'speeds': speeds,
            'average_speed': average_speed,
            'stride_events': stride_events,
            'stride_metrics': stride_metrics,
            'duration': duration,
            'frame_count': len(keypoint_sequence),
            'fps': self.fps
        }

    def _smooth_signal(self, signal: List[float], window_size: int = 5) -> List[float]:
        """Apply simple moving average smoothing to a signal."""
        if len(signal) <= window_size:
            return signal

        smoothed = []
        for i in range(len(signal)):
            start = max(0, i - window_size // 2)
            end = min(len(signal), i + window_size // 2 + 1)
            smoothed.append(np.mean(signal[start:end]))

        return smoothed

    def _coefficient_of_variation(self, values: List[float]) -> float:
        """Calculate coefficient of variation (std/mean) for a list of values."""
        if not values or len(values) < 2:
            return 0.0

        mean_val = np.mean(values)
        if mean_val == 0:
            return 0.0

        std_val = np.std(values)
        return std_val / mean_val


def test_gait_analyzer():
    """Test function for gait analysis with synthetic data."""
    print("Testing gait analyzer with synthetic data...")

    # Create synthetic walking data
    frames = 300  # 10 seconds at 30 FPS
    keypoint_sequence = []

    for i in range(frames):
        # Simulate walking motion
        t = i / 30.0  # Time in seconds

        # Simulate forward movement
        forward_progress = t * 0.1  # Moving forward

        # Simulate walking oscillation
        walking_cycle = np.sin(2 * np.pi * t * 1.2)  # 1.2 Hz walking

        keypoints = {
            'left_hip': {'x': 0.45 + forward_progress, 'y': 0.4, 'visibility': 0.9},
            'right_hip': {'x': 0.55 + forward_progress, 'y': 0.4, 'visibility': 0.9},
            'left_ankle': {'x': 0.45 + forward_progress + 0.1 * walking_cycle,
                          'y': 0.8 + 0.02 * abs(walking_cycle), 'visibility': 0.9},
            'right_ankle': {'x': 0.55 + forward_progress - 0.1 * walking_cycle,
                           'y': 0.8 + 0.02 * abs(-walking_cycle), 'visibility': 0.9}
        }
        keypoint_sequence.append(keypoints)

    # Analyze the synthetic walk
    analyzer = GaitAnalyzer(fps=30.0)
    results = analyzer.analyze_walk(keypoint_sequence)

    print(f"Analysis results:")
    print(f"  Duration: {results['duration']:.2f} seconds")
    print(f"  Average speed: {results['average_speed']:.4f} normalized units/s")
    print(f"  Left strides detected: {len(results['stride_events']['left'])}")
    print(f"  Right strides detected: {len(results['stride_events']['right'])}")
    print(f"  Stride time variability: {results['stride_metrics']['stride_time_variability']:.3f}")
    print(f"  Stride length variability: {results['stride_metrics']['stride_length_variability']:.3f}")

    print("Gait analyzer test completed successfully")


if __name__ == "__main__":
    test_gait_analyzer()