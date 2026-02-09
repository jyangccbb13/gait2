"""
Gait feature extraction from pose keypoint sequences.
Uses velocity-based gait event detection and clinically meaningful metrics.
"""

import numpy as np
from typing import List, Dict, Optional
from scipy.signal import butter, filtfilt

BUTTER_ORDER = 4
BUTTER_CUTOFF_HZ = 6.0
MIN_STRIDE_TIME = 0.3
MIN_FILTER_SAMPLES = 15
HEIGHT_TO_LEG_RATIO = 0.53

# Joint names expected from pose estimator
JOINT_NAMES = [
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_big_toe', 'left_small_toe', 'left_heel',
    'right_big_toe', 'right_small_toe', 'right_heel',
]


class GaitAnalyzer:
    """
    Analyzes gait patterns from sequence of pose keypoints.
    Uses velocity-based heel strike and toe off detection.
    """

    def __init__(self, fps: float = 30.0, person_height_m: float = None):
        self.fps = fps
        self.frame_duration = 1.0 / fps
        self.person_height_m = person_height_m

        # Pre-compute Butterworth filter coefficients
        self._b, self._a = butter(BUTTER_ORDER, BUTTER_CUTOFF_HZ, fs=fps, btype='low')

    def _butterworth_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply Butterworth low-pass filter. Returns raw if too few samples."""
        if len(signal) >= MIN_FILTER_SAMPLES:
            return filtfilt(self._b, self._a, signal)
        return signal

    def _extract_trajectories(self, keypoint_sequence: List[Optional[Dict]]):
        """
        Extract and filter joint trajectories from keypoint sequence.

        Returns:
            trajectories: {joint_name: {'x': array, 'y': array}}
            valid_mask: boolean array indicating which frames had valid data
        """
        n_frames = len(keypoint_sequence)
        valid_mask = np.array([kp is not None for kp in keypoint_sequence])

        trajectories = {}
        for joint in JOINT_NAMES:
            x_raw = np.full(n_frames, np.nan)
            y_raw = np.full(n_frames, np.nan)

            for i, kp in enumerate(keypoint_sequence):
                if kp is not None and joint in kp:
                    x_raw[i] = kp[joint]['x']
                    y_raw[i] = kp[joint]['y']

            # Interpolate gaps using valid indices
            valid_idx = np.where(~np.isnan(x_raw))[0]
            if len(valid_idx) >= 2:
                all_idx = np.arange(n_frames)
                x_interp = np.interp(all_idx, valid_idx, x_raw[valid_idx])
                y_interp = np.interp(all_idx, valid_idx, y_raw[valid_idx])

                x_filtered = self._butterworth_filter(x_interp)
                y_filtered = self._butterworth_filter(y_interp)
            elif len(valid_idx) == 1:
                x_filtered = np.full(n_frames, x_raw[valid_idx[0]])
                y_filtered = np.full(n_frames, y_raw[valid_idx[0]])
            else:
                x_filtered = np.zeros(n_frames)
                y_filtered = np.zeros(n_frames)

            trajectories[joint] = {'x': x_filtered, 'y': y_filtered}

        return trajectories, valid_mask

    def _compute_velocity(self, position_array: np.ndarray) -> np.ndarray:
        """Compute velocity from position using numpy gradient."""
        return np.gradient(position_array) * self.fps

    def _find_zero_crossings_neg(self, signal: np.ndarray, min_distance: int) -> List[int]:
        """
        Find positive-to-negative zero crossings in a signal.
        Enforces minimum distance between crossings.
        """
        crossings = []
        for i in range(1, len(signal)):
            if signal[i - 1] >= 0 and signal[i] < 0:
                if not crossings or (i - crossings[-1]) >= min_distance:
                    crossings.append(i)
        return crossings

    def detect_gait_events(self, keypoint_sequence: List[Optional[Dict]]) -> Dict[str, List[int]]:
        """
        Detect heel strikes and toe offs using velocity-based method.

        Heel strike: positive-to-negative zero crossing in heel Y velocity
        Toe off: positive-to-negative zero crossing in big toe Y velocity

        Returns:
            Dict with keys: left_heel_strikes, right_heel_strikes,
                           left_toe_offs, right_toe_offs
        """
        trajectories, valid_mask = self._extract_trajectories(keypoint_sequence)

        min_dist = int(MIN_STRIDE_TIME * self.fps)

        events = {
            'left_heel_strikes': [],
            'right_heel_strikes': [],
            'left_toe_offs': [],
            'right_toe_offs': [],
        }

        # Left heel strikes from left_heel Y velocity
        left_heel_vy = self._compute_velocity(trajectories['left_heel']['y'])
        events['left_heel_strikes'] = self._find_zero_crossings_neg(left_heel_vy, min_dist)

        # Right heel strikes from right_heel Y velocity
        right_heel_vy = self._compute_velocity(trajectories['right_heel']['y'])
        events['right_heel_strikes'] = self._find_zero_crossings_neg(right_heel_vy, min_dist)

        # Left toe off from left_big_toe Y velocity
        left_toe_vy = self._compute_velocity(trajectories['left_big_toe']['y'])
        events['left_toe_offs'] = self._find_zero_crossings_neg(left_toe_vy, min_dist)

        # Right toe off from right_big_toe Y velocity
        right_toe_vy = self._compute_velocity(trajectories['right_big_toe']['y'])
        events['right_toe_offs'] = self._find_zero_crossings_neg(right_toe_vy, min_dist)

        return events

    def compute_metrics(self, keypoint_sequence: List[Optional[Dict]],
                       gait_events: Dict[str, List[int]]) -> Dict[str, float]:
        """
        Compute temporal and spatial gait metrics.

        Returns dict with: cadence, stride_time_left/right, step_time_left/right,
        stride_time_cv, swing_phase_pct_left/right, stance_phase_pct_left/right,
        double_support_pct, step_asymmetry, and if person_height_m set:
        stride_length, gait_speed.
        """
        metrics = {}
        n_frames = len(keypoint_sequence)
        duration = n_frames * self.frame_duration

        lhs = gait_events.get('left_heel_strikes', [])
        rhs = gait_events.get('right_heel_strikes', [])
        lto = gait_events.get('left_toe_offs', [])
        rto = gait_events.get('right_toe_offs', [])

        total_heel_strikes = len(lhs) + len(rhs)

        # Cadence: total heel strikes / duration * 60
        metrics['cadence'] = (total_heel_strikes / duration * 60.0) if duration > 0 else 0.0

        # Stride times (consecutive same-foot heel strikes)
        left_stride_times = []
        for i in range(1, len(lhs)):
            left_stride_times.append((lhs[i] - lhs[i - 1]) * self.frame_duration)

        right_stride_times = []
        for i in range(1, len(rhs)):
            right_stride_times.append((rhs[i] - rhs[i - 1]) * self.frame_duration)

        metrics['stride_time_left'] = float(np.mean(left_stride_times)) if left_stride_times else 0.0
        metrics['stride_time_right'] = float(np.mean(right_stride_times)) if right_stride_times else 0.0

        # Step times (alternating-foot heel strikes)
        # Merge and sort all heel strikes with foot labels
        all_hs = sorted([(f, 'L') for f in lhs] + [(f, 'R') for f in rhs])
        left_step_times = []
        right_step_times = []
        for i in range(1, len(all_hs)):
            if all_hs[i][1] != all_hs[i - 1][1]:
                step_t = (all_hs[i][0] - all_hs[i - 1][0]) * self.frame_duration
                if all_hs[i][1] == 'L':
                    left_step_times.append(step_t)
                else:
                    right_step_times.append(step_t)

        metrics['step_time_left'] = float(np.mean(left_step_times)) if left_step_times else 0.0
        metrics['step_time_right'] = float(np.mean(right_step_times)) if right_step_times else 0.0

        # Stride time CV
        all_stride_times = left_stride_times + right_stride_times
        if len(all_stride_times) >= 2:
            mean_st = np.mean(all_stride_times)
            std_st = np.std(all_stride_times)
            metrics['stride_time_cv'] = float(std_st / mean_st) if mean_st > 0 else 0.0
        else:
            metrics['stride_time_cv'] = 0.0

        # Swing and stance phase percentages
        # Swing: time from toe off to next heel strike (same foot)
        # Stance: time from heel strike to toe off (same foot)
        for side, hs_list, to_list, label in [
            ('left', lhs, lto, 'left'),
            ('right', rhs, rto, 'right'),
        ]:
            swing_pcts = []
            stance_pcts = []
            stride_t = metrics[f'stride_time_{label}']

            if stride_t > 0:
                # For each heel strike, find the next toe off after it (stance end)
                for hs_frame in hs_list:
                    # Find toe off after this heel strike
                    to_after = [t for t in to_list if t > hs_frame]
                    if to_after:
                        stance_dur = (to_after[0] - hs_frame) * self.frame_duration
                        stance_pcts.append(stance_dur / stride_t * 100.0)

                # For each toe off, find the next heel strike after it (swing end)
                for to_frame in to_list:
                    hs_after = [h for h in hs_list if h > to_frame]
                    if hs_after:
                        swing_dur = (hs_after[0] - to_frame) * self.frame_duration
                        swing_pcts.append(swing_dur / stride_t * 100.0)

            metrics[f'swing_phase_pct_{label}'] = float(np.mean(swing_pcts)) if swing_pcts else 0.0
            metrics[f'stance_phase_pct_{label}'] = float(np.mean(stance_pcts)) if stance_pcts else 0.0

        # Double support percentage
        avg_stance = (metrics.get('stance_phase_pct_left', 0) + metrics.get('stance_phase_pct_right', 0)) / 2
        avg_swing = (metrics.get('swing_phase_pct_left', 0) + metrics.get('swing_phase_pct_right', 0)) / 2
        metrics['double_support_pct'] = max(0.0, avg_stance - avg_swing) if avg_stance > 0 else 0.0

        # Step asymmetry
        mean_step = (metrics['step_time_left'] + metrics['step_time_right']) / 2
        if mean_step > 0:
            metrics['step_asymmetry'] = abs(metrics['step_time_left'] - metrics['step_time_right']) / mean_step
        else:
            metrics['step_asymmetry'] = 0.0

        # Spatial metrics (only if person_height_m is set)
        if self.person_height_m is not None and self.person_height_m > 0:
            trajectories, _ = self._extract_trajectories(keypoint_sequence)

            # Estimate leg length in pixels from hip-ankle distance
            hip_y = (trajectories['left_hip']['y'] + trajectories['right_hip']['y']) / 2
            ankle_y = (trajectories['left_ankle']['y'] + trajectories['right_ankle']['y']) / 2
            leg_pixel = float(np.mean(np.abs(ankle_y - hip_y)))

            leg_m = self.person_height_m * HEIGHT_TO_LEG_RATIO
            pixel_to_m = leg_m / leg_pixel if leg_pixel > 0 else 0.0

            # Stride length from hip X displacement between heel strikes
            stride_lengths = []
            hip_x = (trajectories['left_hip']['x'] + trajectories['right_hip']['x']) / 2

            for hs_list in [lhs, rhs]:
                for i in range(1, len(hs_list)):
                    dx = abs(hip_x[hs_list[i]] - hip_x[hs_list[i - 1]])
                    stride_lengths.append(dx * pixel_to_m)

            metrics['stride_length'] = float(np.mean(stride_lengths)) if stride_lengths else 0.0

            # Gait speed
            mean_stride_time = np.mean(all_stride_times) if all_stride_times else 0.0
            if mean_stride_time > 0 and metrics['stride_length'] > 0:
                metrics['gait_speed'] = metrics['stride_length'] / mean_stride_time
            else:
                metrics['gait_speed'] = 0.0
        else:
            metrics['stride_length'] = 0.0
            metrics['gait_speed'] = 0.0

        return metrics

    def analyze_walk(self, keypoint_sequence: List[Optional[Dict]]) -> Dict:
        """
        Complete gait analysis pipeline for a walking sequence.

        Returns dict with:
            metrics, gait_events, duration, frame_count, fps,
            and backward-compat keys: average_speed, stride_metrics, stride_events
        """
        gait_events = self.detect_gait_events(keypoint_sequence)
        metrics = self.compute_metrics(keypoint_sequence, gait_events)

        duration = len(keypoint_sequence) * self.frame_duration

        return {
            'metrics': metrics,
            'gait_events': gait_events,
            'duration': duration,
            'frame_count': len(keypoint_sequence),
            'fps': self.fps,
            # Backward compatibility
            'average_speed': metrics.get('gait_speed', 0.0),
            'stride_metrics': dict(metrics),
            'stride_events': {
                'left': gait_events.get('left_heel_strikes', []),
                'right': gait_events.get('right_heel_strikes', []),
            },
            'speeds': [],
        }


def test_gait_analyzer():
    """Test function for gait analysis with synthetic data."""
    print("Testing gait analyzer with synthetic data...")

    frames = 300  # 10 seconds at 30 FPS
    fps = 30.0
    keypoint_sequence = []

    for i in range(frames):
        t = i / fps
        # Simulate ~1.1s stride cycle (~55 cadence per foot, ~110 total)
        walking_freq = 0.9  # Hz per foot (stride freq)
        phase = 2 * np.pi * walking_freq * t

        # Forward movement
        forward = t * 0.05

        # Heel Y: oscillates up and down (heel strike = peak descending)
        left_heel_y = 0.85 + 0.03 * np.sin(phase)
        right_heel_y = 0.85 + 0.03 * np.sin(phase + np.pi)

        # Big toe Y: similar but phase-shifted (toe off = peak descending)
        left_toe_y = 0.84 + 0.025 * np.sin(phase + 0.5)
        right_toe_y = 0.84 + 0.025 * np.sin(phase + np.pi + 0.5)

        keypoints = {
            'left_hip': {'x': 0.48 + forward, 'y': 0.4, 'visibility': 0.9},
            'right_hip': {'x': 0.52 + forward, 'y': 0.4, 'visibility': 0.9},
            'left_knee': {'x': 0.47 + forward, 'y': 0.6, 'visibility': 0.9},
            'right_knee': {'x': 0.53 + forward, 'y': 0.6, 'visibility': 0.9},
            'left_ankle': {'x': 0.46 + forward + 0.02 * np.sin(phase), 'y': 0.78, 'visibility': 0.9},
            'right_ankle': {'x': 0.54 + forward + 0.02 * np.sin(phase + np.pi), 'y': 0.78, 'visibility': 0.9},
            'left_big_toe': {'x': 0.45 + forward, 'y': left_toe_y, 'visibility': 0.8},
            'left_small_toe': {'x': 0.44 + forward, 'y': left_toe_y + 0.005, 'visibility': 0.7},
            'left_heel': {'x': 0.47 + forward, 'y': left_heel_y, 'visibility': 0.8},
            'right_big_toe': {'x': 0.55 + forward, 'y': right_toe_y, 'visibility': 0.8},
            'right_small_toe': {'x': 0.56 + forward, 'y': right_toe_y + 0.005, 'visibility': 0.7},
            'right_heel': {'x': 0.53 + forward, 'y': right_heel_y, 'visibility': 0.8},
        }
        keypoint_sequence.append(keypoints)

    analyzer = GaitAnalyzer(fps=fps, person_height_m=1.75)
    results = analyzer.analyze_walk(keypoint_sequence)

    metrics = results['metrics']
    print(f"Analysis results:")
    print(f"  Duration: {results['duration']:.2f} seconds")
    print(f"  Cadence: {metrics['cadence']:.1f} steps/min")
    print(f"  Stride time L: {metrics['stride_time_left']:.3f}s")
    print(f"  Stride time R: {metrics['stride_time_right']:.3f}s")
    print(f"  Stride time CV: {metrics['stride_time_cv']:.4f}")
    print(f"  Step asymmetry: {metrics['step_asymmetry']:.4f}")
    print(f"  Swing phase L: {metrics['swing_phase_pct_left']:.1f}%")
    print(f"  Swing phase R: {metrics['swing_phase_pct_right']:.1f}%")
    print(f"  Double support: {metrics['double_support_pct']:.1f}%")
    print(f"  Stride length: {metrics['stride_length']:.3f} m")
    print(f"  Gait speed: {metrics['gait_speed']:.3f} m/s")
    print(f"  Left HS events: {len(results['gait_events']['left_heel_strikes'])}")
    print(f"  Right HS events: {len(results['gait_events']['right_heel_strikes'])}")
    print("Gait analyzer test completed successfully")


if __name__ == "__main__":
    test_gait_analyzer()
