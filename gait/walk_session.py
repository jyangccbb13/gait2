"""
Walk session detection and analysis.
Automatically detects complete walking sessions and provides clinical insights.
"""

import numpy as np
import time
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class WalkSession:
    """Represents a complete walking session with clinical metrics."""
    session_id: str
    start_time: float
    end_time: float
    duration: float
    keypoints_data: List[Dict]
    gait_metrics: Dict
    clinical_assessment: Dict
    baseline_comparison: Optional[Dict] = None


class WalkDetector:
    """
    Automatically detects when a person completes a walking session.
    Uses person presence and movement patterns to segment walks.
    """

    def __init__(self,
                 min_walk_duration: float = 3.0,
                 max_gap_duration: float = 2.0,
                 min_movement_threshold: float = 0.01):
        """
        Initialize walk detector.

        Args:
            min_walk_duration: Minimum time for a valid walk session
            max_gap_duration: Maximum time gap before ending a session
            min_movement_threshold: Minimum movement to consider active walking
        """
        self.min_walk_duration = min_walk_duration
        self.max_gap_duration = max_gap_duration
        self.min_movement_threshold = min_movement_threshold

        # Session tracking
        self.current_session_start = None
        self.last_valid_pose_time = None
        self.session_keypoints = []
        self.is_recording_session = False

    def update(self, keypoints: Optional[Dict], timestamp: float) -> Optional[WalkSession]:
        """
        Update detector with new pose data.

        Args:
            keypoints: Pose keypoints or None if no pose detected
            timestamp: Current timestamp

        Returns:
            WalkSession if a complete session was detected, None otherwise
        """
        if keypoints is not None and self._is_valid_walking_pose(keypoints):
            # Valid pose detected
            if not self.is_recording_session:
                # Start new session
                self._start_session(timestamp)

            self.session_keypoints.append({
                'timestamp': timestamp,
                'keypoints': keypoints
            })
            self.last_valid_pose_time = timestamp

        else:
            # No valid pose or person not visible
            if self.is_recording_session:
                time_gap = timestamp - self.last_valid_pose_time

                if time_gap > self.max_gap_duration:
                    # Gap too long, end session
                    return self._end_session(timestamp)

        return None

    def force_end_session(self, timestamp: float) -> Optional[WalkSession]:
        """Force end current session (useful for manual control)."""
        if self.is_recording_session:
            return self._end_session(timestamp)
        return None

    def _start_session(self, timestamp: float):
        """Start a new walk session."""
        self.current_session_start = timestamp
        self.last_valid_pose_time = timestamp
        self.session_keypoints = []
        self.is_recording_session = True
        print(f"🚶‍♂️ Started walk session at {timestamp:.1f}")

    def _end_session(self, timestamp: float) -> Optional[WalkSession]:
        """End current session and return WalkSession if valid."""
        if not self.is_recording_session:
            return None

        duration = timestamp - self.current_session_start

        # Only return valid sessions
        if duration >= self.min_walk_duration and len(self.session_keypoints) > 10:
            session_id = f"walk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            print(f"✅ Completed walk session: {duration:.1f}s, {len(self.session_keypoints)} poses")

            # Reset for next session
            self.is_recording_session = False

            return WalkSession(
                session_id=session_id,
                start_time=self.current_session_start,
                end_time=timestamp,
                duration=duration,
                keypoints_data=self.session_keypoints.copy(),
                gait_metrics={},  # Will be filled by analyzer
                clinical_assessment={}  # Will be filled by clinical analyzer
            )
        else:
            print(f"❌ Session too short: {duration:.1f}s, discarded")

        # Reset for next session
        self.is_recording_session = False
        return None

    def _is_valid_walking_pose(self, keypoints: Dict) -> bool:
        """Check if pose represents valid walking posture."""
        # Check that key joints are visible
        required_joints = ['left_hip', 'right_hip', 'left_ankle', 'right_ankle']

        for joint in required_joints:
            if joint not in keypoints or keypoints[joint]['visibility'] < 0.5:
                return False

        # Check for reasonable posture (person upright)
        left_hip_y = keypoints['left_hip']['y']
        left_ankle_y = keypoints['left_ankle']['y']

        # Hip should be above ankle
        if left_ankle_y <= left_hip_y:
            return False

        return True


class ClinicalGaitAnalyzer:
    """
    Provides clinical interpretation of gait metrics based on research literature.
    """

    def __init__(self):
        """Initialize with clinical reference values."""
        # Reference values from gait research literature
        self.reference_values = {
            'speed': {
                'normal_range': (1.2, 1.4),  # m/s for healthy adults
                'concerning_below': 1.0,     # Speed indicating mobility issues
                'units': 'm/s'
            },
            'stride_time': {
                'normal_range': (1.0, 1.2),  # seconds
                'concerning_above': 1.4,     # Slower than normal
                'units': 'seconds'
            },
            'stride_length': {
                'normal_range': (1.3, 1.5),  # meters
                'concerning_below': 1.1,     # Shorter strides
                'units': 'meters'
            },
            'stride_time_variability': {
                'normal_range': (0.0, 0.03), # CV < 3% is normal
                'concerning_above': 0.05,    # CV > 5% indicates instability
                'units': 'coefficient of variation'
            },
            'stride_length_variability': {
                'normal_range': (0.0, 0.04), # CV < 4% is normal
                'concerning_above': 0.06,    # CV > 6% indicates problems
                'units': 'coefficient of variation'
            }
        }

    def analyze_session(self, session: WalkSession, gait_results: Dict) -> Dict:
        """
        Provide clinical interpretation of a walk session.

        Args:
            session: Walk session data
            gait_results: Results from GaitAnalyzer

        Returns:
            Clinical assessment with interpretations and recommendations
        """
        assessment = {
            'session_id': session.session_id,
            'timestamp': datetime.now().isoformat(),
            'raw_metrics': gait_results,
            'clinical_scores': {},
            'interpretations': {},
            'recommendations': [],
            'risk_factors': [],
            'overall_score': 0
        }

        # Analyze each metric
        metrics_to_analyze = [
            ('average_speed', 'speed'),
            ('stride_time_left', 'stride_time'),
            ('stride_length_left', 'stride_length'),
            ('stride_time_variability', 'stride_time_variability'),
            ('stride_length_variability', 'stride_length_variability')
        ]

        total_score = 0
        scored_metrics = 0

        for metric_key, reference_key in metrics_to_analyze:
            if metric_key in gait_results.get('stride_metrics', {}) or metric_key == 'average_speed':
                value = (gait_results.get(metric_key, 0) if metric_key == 'average_speed'
                        else gait_results.get('stride_metrics', {}).get(metric_key, 0))

                score, interpretation = self._score_metric(value, reference_key)
                assessment['clinical_scores'][metric_key] = score
                assessment['interpretations'][metric_key] = interpretation

                total_score += score
                scored_metrics += 1

        # Calculate overall score (0-100)
        if scored_metrics > 0:
            assessment['overall_score'] = (total_score / scored_metrics) * 100

        # Generate recommendations and risk factors
        assessment['recommendations'] = self._generate_recommendations(assessment)
        assessment['risk_factors'] = self._identify_risk_factors(assessment)

        return assessment

    def _score_metric(self, value: float, metric_type: str) -> Tuple[float, str]:
        """
        Score a single metric and provide interpretation.

        Returns:
            Tuple of (score 0-1, interpretation string)
        """
        if metric_type not in self.reference_values:
            return 0.5, "No reference data available"

        ref = self.reference_values[metric_type]

        if metric_type in ['stride_time_variability', 'stride_length_variability']:
            # Lower variability is better
            if value <= ref['normal_range'][1]:
                score = 1.0
                interpretation = f"Excellent stability (CV: {value:.3f})"
            elif value <= ref['concerning_above']:
                score = 0.7
                interpretation = f"Mild variability (CV: {value:.3f})"
            else:
                score = 0.3
                interpretation = f"High variability - instability concern (CV: {value:.3f})"

        elif metric_type == 'stride_time':
            # Moderate stride time is best
            if ref['normal_range'][0] <= value <= ref['normal_range'][1]:
                score = 1.0
                interpretation = f"Normal stride timing ({value:.2f}s)"
            elif value > ref['concerning_above']:
                score = 0.4
                interpretation = f"Slow stride timing ({value:.2f}s) - may indicate weakness"
            else:
                score = 0.7
                interpretation = f"Fast stride timing ({value:.2f}s)"

        else:
            # Higher values generally better (speed, stride length)
            if value >= ref['normal_range'][0]:
                score = 1.0
                interpretation = f"Normal {metric_type} ({value:.3f})"
            elif 'concerning_below' in ref and value < ref['concerning_below']:
                score = 0.3
                interpretation = f"Low {metric_type} ({value:.3f}) - mobility concern"
            else:
                score = 0.6
                interpretation = f"Below average {metric_type} ({value:.3f})"

        return score, interpretation

    def _generate_recommendations(self, assessment: Dict) -> List[str]:
        """Generate clinical recommendations based on assessment."""
        recommendations = []
        scores = assessment['clinical_scores']

        # Speed recommendations
        if 'average_speed' in scores and scores['average_speed'] < 0.5:
            recommendations.append("Consider strength training to improve walking speed")
            recommendations.append("Consult physical therapist for gait training")

        # Variability recommendations
        variability_metrics = ['stride_time_variability', 'stride_length_variability']
        high_variability = any(scores.get(metric, 1) < 0.5 for metric in variability_metrics)

        if high_variability:
            recommendations.append("Practice balance exercises to reduce gait variability")
            recommendations.append("Consider fall risk assessment")

        # Overall recommendations
        if assessment['overall_score'] < 60:
            recommendations.append("Schedule comprehensive gait analysis with healthcare provider")
        elif assessment['overall_score'] < 80:
            recommendations.append("Continue monitoring gait patterns")

        return recommendations

    def _identify_risk_factors(self, assessment: Dict) -> List[str]:
        """Identify potential risk factors from gait analysis."""
        risk_factors = []
        scores = assessment['clinical_scores']

        # Fall risk factors
        if scores.get('stride_time_variability', 1) < 0.5:
            risk_factors.append("Increased fall risk due to gait instability")

        if scores.get('average_speed', 1) < 0.4:
            risk_factors.append("Slow walking speed associated with mobility decline")

        if scores.get('stride_length_left', 1) < 0.5:
            risk_factors.append("Reduced stride length may indicate muscle weakness")

        # Neurological concerns
        high_variability_count = sum(1 for metric in ['stride_time_variability', 'stride_length_variability']
                                   if scores.get(metric, 1) < 0.5)

        if high_variability_count >= 2:
            risk_factors.append("Multiple variability issues may indicate neurological concerns")

        return risk_factors

    def compare_to_baseline(self, current_assessment: Dict, baseline_assessment: Dict) -> Dict:
        """Compare current assessment to baseline."""
        comparison = {
            'baseline_date': baseline_assessment.get('timestamp'),
            'current_date': current_assessment.get('timestamp'),
            'score_change': current_assessment['overall_score'] - baseline_assessment['overall_score'],
            'metric_changes': {},
            'interpretation': '',
            'recommendations': []
        }

        # Compare individual metrics
        for metric in current_assessment['clinical_scores']:
            if metric in baseline_assessment['clinical_scores']:
                current_score = current_assessment['clinical_scores'][metric]
                baseline_score = baseline_assessment['clinical_scores'][metric]
                change = (current_score - baseline_score) * 100

                comparison['metric_changes'][metric] = {
                    'change_percent': change,
                    'direction': 'improvement' if change > 5 else 'decline' if change < -5 else 'stable',
                    'current_score': current_score * 100,
                    'baseline_score': baseline_score * 100
                }

        # Overall interpretation
        score_change = comparison['score_change']
        if score_change > 5:
            comparison['interpretation'] = "Significant improvement in gait patterns"
        elif score_change < -5:
            comparison['interpretation'] = "Concerning decline in gait patterns"
        else:
            comparison['interpretation'] = "Gait patterns remain stable"

        # Baseline-specific recommendations
        if score_change < -10:
            comparison['recommendations'].append("Consult healthcare provider about gait changes")
        elif score_change > 10:
            comparison['recommendations'].append("Continue current activity/therapy program")

        return comparison


def test_walk_detection():
    """Test walk detection and clinical analysis."""
    print("Testing walk detection and clinical analysis...")

    detector = WalkDetector(min_walk_duration=2.0)
    clinical_analyzer = ClinicalGaitAnalyzer()

    # Simulate walking session
    import time
    start_time = time.time()

    # Simulate poses during walking
    for i in range(100):  # 100 frames ~ 3 seconds
        current_time = start_time + (i * 0.033)  # 30 FPS

        if 10 <= i <= 90:  # Person visible for most of the time
            # Simulate walking keypoints
            keypoints = {
                'left_hip': {'x': 0.45, 'y': 0.4, 'visibility': 0.9},
                'right_hip': {'x': 0.55, 'y': 0.4, 'visibility': 0.9},
                'left_ankle': {'x': 0.45, 'y': 0.8, 'visibility': 0.8},
                'right_ankle': {'x': 0.55, 'y': 0.8, 'visibility': 0.8}
            }
        else:
            keypoints = None  # Person not visible

        session = detector.update(keypoints, current_time)

        if session:
            print(f"Walk session detected: {session.session_id}")
            print(f"Duration: {session.duration:.1f}s")

            # Mock gait analysis results for testing
            mock_gait_results = {
                'average_speed': 1.3,
                'stride_metrics': {
                    'stride_time_left': 1.1,
                    'stride_length_left': 1.4,
                    'stride_time_variability': 0.02,
                    'stride_length_variability': 0.03
                }
            }

            # Clinical analysis
            clinical_assessment = clinical_analyzer.analyze_session(session, mock_gait_results)
            print(f"Overall clinical score: {clinical_assessment['overall_score']:.1f}/100")
            print("Recommendations:", clinical_assessment['recommendations'])

            break

    print("Walk detection test completed")


if __name__ == "__main__":
    test_walk_detection()