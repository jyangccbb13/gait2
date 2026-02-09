"""
Baseline modeling for gait analysis.
Stores historical walking data and compares new walks against baseline patterns.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os


class BaselineModel:
    """
    Manages baseline gait patterns for comparison with new walks.

    Stores historical walk data and computes statistical baselines
    for key gait metrics to identify deviations from normal patterns.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize baseline model.

        Args:
            data_dir: Directory to store baseline data files
        """
        self.data_dir = data_dir
        self.baseline_file = os.path.join(data_dir, "baseline_data.json")
        self.walks_file = os.path.join(data_dir, "walk_history.json")

        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Key metrics to track for baseline
        self.baseline_metrics = [
            'gait_speed',
            'cadence',
            'stride_time_left',
            'stride_time_right',
            'stride_time_cv',
            'step_asymmetry',
            'swing_phase_pct_left',
            'swing_phase_pct_right',
            'double_support_pct',
            'stride_length',
        ]

        # Load existing data
        self.walk_history = self._load_walk_history()
        self.baseline_stats = self._load_baseline_stats()

    def add_walk(self, walk_results: Dict, subject_id: str = "default") -> str:
        """
        Add a new walk to the history and update baseline.

        Args:
            walk_results: Results from GaitAnalyzer.analyze_walk()
            subject_id: Identifier for the subject (for future multi-subject support)

        Returns:
            str: Unique walk ID for this recording
        """
        # Generate unique walk ID
        walk_id = f"{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create walk record
        walk_record = {
            'walk_id': walk_id,
            'subject_id': subject_id,
            'timestamp': datetime.now().isoformat(),
            'duration': walk_results['duration'],
            'frame_count': walk_results['frame_count'],
            'fps': walk_results['fps'],
            'metrics': {}
        }

        # Extract key metrics for baseline tracking
        metrics_source = walk_results.get('metrics', {}) or walk_results.get('stride_metrics', {})
        walk_record['metrics']['average_speed'] = metrics_source.get('gait_speed', walk_results.get('average_speed', 0.0))

        for metric in self.baseline_metrics:
            if metric in metrics_source:
                walk_record['metrics'][metric] = metrics_source[metric]

        # Store full results for detailed analysis
        walk_record['full_results'] = walk_results

        # Add to history
        if subject_id not in self.walk_history:
            self.walk_history[subject_id] = []

        self.walk_history[subject_id].append(walk_record)

        # Update baseline statistics
        self._update_baseline_stats(subject_id)

        # Save to files
        self._save_walk_history()
        self._save_baseline_stats()

        return walk_id

    def compare_to_baseline(self,
                          walk_results: Dict,
                          subject_id: str = "default") -> Dict[str, float]:
        """
        Compare a walk to the current baseline for the subject.

        Args:
            walk_results: Results from GaitAnalyzer.analyze_walk()
            subject_id: Subject identifier

        Returns:
            Dict[str, float]: Z-scores for each baseline metric
                             Positive = above baseline, Negative = below baseline
        """
        if subject_id not in self.baseline_stats:
            return {metric: 0.0 for metric in self.baseline_metrics}

        baseline = self.baseline_stats[subject_id]
        z_scores = {}

        # Calculate z-scores for each metric
        metrics_source = walk_results.get('metrics', {}) or walk_results.get('stride_metrics', {})
        current_metrics = dict(metrics_source)
        current_metrics.setdefault('average_speed', walk_results.get('average_speed', 0.0))

        for metric in self.baseline_metrics:
            if metric in baseline and metric in current_metrics:
                mean = baseline[metric]['mean']
                std = baseline[metric]['std']

                if std > 0:
                    z_score = (current_metrics[metric] - mean) / std
                else:
                    z_score = 0.0

                z_scores[metric] = z_score
            else:
                z_scores[metric] = 0.0

        return z_scores

    def get_baseline_summary(self, subject_id: str = "default") -> Optional[Dict]:
        """
        Get baseline statistics summary for a subject.

        Args:
            subject_id: Subject identifier

        Returns:
            Optional[Dict]: Baseline statistics or None if no data
        """
        if subject_id not in self.baseline_stats:
            return None

        baseline = self.baseline_stats[subject_id]
        walk_count = len(self.walk_history.get(subject_id, []))

        summary = {
            'subject_id': subject_id,
            'walk_count': walk_count,
            'last_updated': baseline.get('last_updated'),
            'metrics': {}
        }

        # Format baseline metrics for display
        for metric in self.baseline_metrics:
            if metric in baseline:
                summary['metrics'][metric] = {
                    'mean': round(baseline[metric]['mean'], 4),
                    'std': round(baseline[metric]['std'], 4),
                    'min': round(baseline[metric]['min'], 4),
                    'max': round(baseline[metric]['max'], 4)
                }

        return summary

    def get_walk_history(self, subject_id: str = "default", limit: int = 20) -> List[Dict]:
        """
        Get recent walk history for a subject.

        Args:
            subject_id: Subject identifier
            limit: Maximum number of walks to return

        Returns:
            List[Dict]: Recent walks, most recent first
        """
        if subject_id not in self.walk_history:
            return []

        walks = self.walk_history[subject_id]
        # Return most recent walks first
        return walks[-limit:][::-1]

    def _update_baseline_stats(self, subject_id: str):
        """Update baseline statistics for a subject."""
        if subject_id not in self.walk_history or not self.walk_history[subject_id]:
            return

        walks = self.walk_history[subject_id]

        # Extract metrics from all walks
        metrics_data = {metric: [] for metric in self.baseline_metrics}

        for walk in walks:
            walk_metrics = walk['metrics']
            for metric in self.baseline_metrics:
                if metric in walk_metrics and walk_metrics[metric] is not None:
                    metrics_data[metric].append(walk_metrics[metric])

        # Calculate statistics for each metric
        if subject_id not in self.baseline_stats:
            self.baseline_stats[subject_id] = {}

        for metric in self.baseline_metrics:
            values = metrics_data[metric]
            if len(values) >= 2:  # Need at least 2 data points
                self.baseline_stats[subject_id][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }

        self.baseline_stats[subject_id]['last_updated'] = datetime.now().isoformat()

    def _load_walk_history(self) -> Dict:
        """Load walk history from file."""
        try:
            if os.path.exists(self.walks_file):
                with open(self.walks_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load walk history: {e}")

        return {}

    def _save_walk_history(self):
        """Save walk history to file."""
        try:
            with open(self.walks_file, 'w') as f:
                json.dump(self.walk_history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save walk history: {e}")

    def _load_baseline_stats(self) -> Dict:
        """Load baseline statistics from file."""
        try:
            if os.path.exists(self.baseline_file):
                with open(self.baseline_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load baseline stats: {e}")

        return {}

    def _save_baseline_stats(self):
        """Save baseline statistics to file."""
        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(self.baseline_stats, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save baseline stats: {e}")

    def reset_baseline(self, subject_id: str = "default"):
        """Reset baseline for a subject (useful for testing or new subjects)."""
        if subject_id in self.walk_history:
            del self.walk_history[subject_id]

        if subject_id in self.baseline_stats:
            del self.baseline_stats[subject_id]

        self._save_walk_history()
        self._save_baseline_stats()


def test_baseline_model():
    """Test function for baseline modeling."""
    from gait_features import GaitAnalyzer

    print("Testing baseline model...")

    # Create test data directory
    test_data_dir = "test_data"
    baseline = BaselineModel(test_data_dir)

    # Clear any existing test data
    baseline.reset_baseline("test_subject")

    # Create synthetic walk data for testing
    analyzer = GaitAnalyzer()

    # Simulate 3 baseline walks
    for walk_num in range(3):
        print(f"Adding baseline walk {walk_num + 1}")

        # Create synthetic keypoint sequence
        keypoint_sequence = []
        for i in range(150):  # 5 seconds
            t = i / 30.0
            keypoints = {
                'left_hip': {'x': 0.45 + t * 0.1, 'y': 0.4, 'visibility': 0.9},
                'right_hip': {'x': 0.55 + t * 0.1, 'y': 0.4, 'visibility': 0.9},
                'left_ankle': {'x': 0.45 + t * 0.1, 'y': 0.8, 'visibility': 0.9},
                'right_ankle': {'x': 0.55 + t * 0.1, 'y': 0.8, 'visibility': 0.9}
            }
            keypoint_sequence.append(keypoints)

        # Analyze walk
        walk_results = analyzer.analyze_walk(keypoint_sequence)

        # Add to baseline
        walk_id = baseline.add_walk(walk_results, "test_subject")
        print(f"  Added walk: {walk_id}")

    # Check baseline summary
    summary = baseline.get_baseline_summary("test_subject")
    if summary:
        print(f"\nBaseline Summary:")
        print(f"  Subject: {summary['subject_id']}")
        print(f"  Walk count: {summary['walk_count']}")
        print(f"  Average speed baseline: {summary['metrics']['average_speed']['mean']:.4f} ± {summary['metrics']['average_speed']['std']:.4f}")

    # Test comparison with a new walk
    print(f"\nTesting comparison with new walk...")

    # Create a slightly different walk
    new_keypoint_sequence = []
    for i in range(150):
        t = i / 30.0
        keypoints = {
            'left_hip': {'x': 0.45 + t * 0.12, 'y': 0.4, 'visibility': 0.9},  # Slightly faster
            'right_hip': {'x': 0.55 + t * 0.12, 'y': 0.4, 'visibility': 0.9},
            'left_ankle': {'x': 0.45 + t * 0.12, 'y': 0.8, 'visibility': 0.9},
            'right_ankle': {'x': 0.55 + t * 0.12, 'y': 0.8, 'visibility': 0.9}
        }
        new_keypoint_sequence.append(keypoints)

    new_walk_results = analyzer.analyze_walk(new_keypoint_sequence)
    z_scores = baseline.compare_to_baseline(new_walk_results, "test_subject")

    print(f"Z-scores for new walk:")
    for metric, z_score in z_scores.items():
        if abs(z_score) > 0.1:  # Only show significant deviations
            print(f"  {metric}: {z_score:.3f}")

    # Clean up test files
    import shutil
    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir)

    print("Baseline model test completed")


if __name__ == "__main__":
    test_baseline_model()