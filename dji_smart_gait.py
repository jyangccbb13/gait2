"""
Smart DJI Gait Analysis with automatic walk detection, clinical insights, and baseline comparison.
"""

import cv2
import numpy as np
import time
import json
import os
from camera.video_source import VideoSource
from pose.pose_estimator import PoseEstimator
from gait.gait_features import GaitAnalyzer
from gait.baseline import BaselineModel
from gait.walk_session import WalkDetector, ClinicalGaitAnalyzer, WalkSession


class SmartGaitAnalyzer:
    """
    Advanced gait analysis with automatic session detection and clinical insights.
    """

    def __init__(self):
        """Initialize smart gait analyzer."""
        # Core components
        self.video_source = VideoSource(0)  # DJI camera
        self.pose_estimator = PoseEstimator()
        self.gait_analyzer = GaitAnalyzer(fps=30.0)
        self.baseline_model = BaselineModel()

        # Smart features
        self.walk_detector = WalkDetector(min_walk_duration=3.0, max_gap_duration=2.0)
        self.clinical_analyzer = ClinicalGaitAnalyzer()

        # Session management
        self.completed_sessions = []
        self.current_session_display = None
        self.show_clinical_analysis = True
        self.baseline_mode = False

        # Display settings
        self.show_skeleton = True
        self.show_metrics = True

        # Data persistence
        self.sessions_file = "data/walk_sessions.json"
        self.baseline_file = "data/baseline_session.json"
        os.makedirs("data", exist_ok=True)

    def run(self):
        """Start the smart gait analysis system."""
        print("🧠 Smart DJI Gait Analysis System")
        print("=" * 60)
        print("🎯 Features:")
        print("  • Automatic walk detection")
        print("  • Clinical interpretation")
        print("  • Baseline comparison")
        print("  • Research-based insights")
        print("\n🎮 Controls:")
        print("  • 'q' - Quit")
        print("  • 's' - Toggle skeleton")
        print("  • 'm' - Toggle metrics")
        print("  • 'c' - Toggle clinical analysis")
        print("  • 'b' - Baseline mode (next walk becomes baseline)")
        print("  • 'r' - Reset session data")
        print("  • 'SPACE' - Force end current walk session")

        if not self.video_source.connect():
            print("❌ Could not connect to DJI camera")
            return

        print(f"\n✅ DJI connected: {self.video_source.get_properties()}")
        print("\n🚶‍♂️ Walk in front of camera for automatic detection...")

        try:
            while True:
                frame = self._process_frame()
                if frame is None:
                    break

                cv2.imshow('🧠 Smart Gait Analysis - DJI Osmo 5', frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keypress(key):
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self._cleanup()

    def _process_frame(self):
        """Process a single frame with smart analysis."""
        success, frame = self.video_source.read_frame()
        if not success:
            return None

        current_time = time.time()

        # Run pose estimation
        pose_success, keypoints = self.pose_estimator.estimate_pose(frame)
        valid_keypoints = keypoints if (pose_success and
                                      self.pose_estimator.is_pose_valid(keypoints)) else None

        # Update walk detector
        completed_session = self.walk_detector.update(valid_keypoints, current_time)

        if completed_session:
            self._process_completed_session(completed_session)

        # Create display frame
        display_frame = self._create_display_frame(frame, valid_keypoints, current_time)

        return display_frame

    def _process_completed_session(self, session: WalkSession):
        """Process a completed walk session."""
        print(f"\n🎉 Walk completed: {session.session_id}")

        # Extract keypoints for analysis
        keypoints_only = [frame_data['keypoints'] for frame_data in session.keypoints_data]

        # Run gait analysis
        gait_results = self.gait_analyzer.analyze_walk(keypoints_only)
        session.gait_metrics = gait_results

        # Clinical analysis
        clinical_assessment = self.clinical_analyzer.analyze_session(session, gait_results)
        session.clinical_assessment = clinical_assessment

        # Baseline handling
        if self.baseline_mode:
            self._save_baseline(session)
            self.baseline_mode = False
            print("✅ Baseline recorded!")
        else:
            # Compare to baseline if available
            baseline_assessment = self._load_baseline()
            if baseline_assessment:
                comparison = self.clinical_analyzer.compare_to_baseline(
                    clinical_assessment, baseline_assessment
                )
                session.baseline_comparison = comparison

        # Store session
        self.completed_sessions.append(session)
        self.current_session_display = session
        self._save_session(session)

        # Print summary
        self._print_session_summary(session)

    def _create_display_frame(self, frame, keypoints, current_time):
        """Create the main display frame with all overlays."""
        display_frame = frame.copy()
        height, width = frame.shape[:2]

        # Draw pose overlay
        if self.show_skeleton and keypoints:
            display_frame = self._draw_skeleton(display_frame, keypoints)
            self._draw_confidence_indicator(display_frame, keypoints)

        # Status indicators
        self._draw_status_bar(display_frame, current_time)

        # Main metrics panel
        if self.show_metrics:
            self._draw_main_metrics_panel(display_frame)

        # Clinical analysis panel
        if self.show_clinical_analysis and self.current_session_display:
            self._draw_clinical_panel(display_frame)

        return display_frame

    def _draw_skeleton(self, frame, keypoints):
        """Draw skeleton with enhanced visibility."""
        height, width = frame.shape[:2]

        pixel_joints = {}
        for joint_name, joint_data in keypoints.items():
            if joint_data['visibility'] > 0.5:
                x = int(joint_data['x'] * width)
                y = int(joint_data['y'] * height)
                pixel_joints[joint_name] = (x, y)

        # Enhanced skeleton colors
        connections = [
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle')
        ]

        # Draw connections (bright cyan)
        for joint1, joint2 in connections:
            if joint1 in pixel_joints and joint2 in pixel_joints:
                cv2.line(frame, pixel_joints[joint1], pixel_joints[joint2], (255, 255, 0), 5)

        # Draw joints (bright green with labels)
        for joint_name, (x, y) in pixel_joints.items():
            cv2.circle(frame, (x, y), 12, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 12, (255, 255, 255), 3)

            # Joint label
            label = joint_name.split('_')[1][:4].upper()
            cv2.putText(frame, label, (x + 15, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    def _draw_confidence_indicator(self, frame, keypoints):
        """Draw pose confidence indicator."""
        confidences = [joint['visibility'] for joint in keypoints.values()]
        avg_confidence = sum(confidences) / len(confidences)

        color = ((0, 255, 0) if avg_confidence > 0.8 else
                (0, 255, 255) if avg_confidence > 0.5 else (0, 0, 255))

        cv2.circle(frame, (60, 60), 30, color, -1)
        cv2.circle(frame, (60, 60), 30, (255, 255, 255), 3)

        status = "HIGH" if avg_confidence > 0.8 else "MED" if avg_confidence > 0.5 else "LOW"
        cv2.putText(frame, status, (105, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _draw_status_bar(self, frame, current_time):
        """Draw system status bar."""
        height, width = frame.shape[:2]

        # Title and mode indicators
        title = "🧠 SMART GAIT ANALYSIS - DJI OSMO 5"
        cv2.putText(frame, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Mode indicators
        y_pos = 70
        if self.baseline_mode:
            cv2.putText(frame, "🎯 BASELINE MODE - Next walk will be baseline", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        # Session status
        session_status = "🔍 Detecting walks..." if not self.walk_detector.is_recording_session else "📹 Recording walk..."
        cv2.putText(frame, session_status, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_main_metrics_panel(self, frame):
        """Draw main metrics panel."""
        height, width = frame.shape[:2]

        panel_x = width - 350
        panel_y = 100
        panel_width = 340

        # Session count and current status
        metrics_text = [
            "=== SMART GAIT SYSTEM ===",
            f"Completed Walks: {len(self.completed_sessions)}",
            f"Recording: {'YES' if self.walk_detector.is_recording_session else 'NO'}",
            ""
        ]

        if self.walk_detector.is_recording_session:
            session_duration = time.time() - self.walk_detector.current_session_start
            pose_count = len([kp for kp in self.walk_detector.session_keypoints if kp])
            metrics_text.extend([
                "=== CURRENT WALK ===",
                f"Duration: {session_duration:.1f}s",
                f"Valid Poses: {pose_count}",
                ""
            ])

        # Latest session summary
        if self.current_session_display:
            session = self.current_session_display
            clinical = session.clinical_assessment

            metrics_text.extend([
                "=== LAST COMPLETED WALK ===",
                f"ID: {session.session_id[-8:]}",
                f"Duration: {session.duration:.1f}s",
                f"Clinical Score: {clinical['overall_score']:.0f}/100",
                ""
            ])

            # Show key metrics
            if clinical['clinical_scores']:
                metrics_text.append("=== KEY METRICS ===")
                for metric, score in clinical['clinical_scores'].items():
                    display_name = metric.replace('_', ' ').title()
                    score_text = f"{score*100:.0f}%"
                    status = "✅" if score > 0.8 else "⚠️" if score > 0.5 else "❌"
                    metrics_text.append(f"{status} {display_name}: {score_text}")

        self._draw_panel(frame, panel_x, panel_y, panel_width, metrics_text)

    def _draw_clinical_panel(self, frame):
        """Draw clinical analysis panel."""
        if not self.current_session_display:
            return

        session = self.current_session_display
        clinical = session.clinical_assessment
        height, width = frame.shape[:2]

        panel_x = 10
        panel_y = height - 300
        panel_width = 600

        clinical_text = [
            "=== CLINICAL ANALYSIS ===",
            f"Overall Score: {clinical['overall_score']:.0f}/100",
            ""
        ]

        # Risk factors
        if clinical['risk_factors']:
            clinical_text.append("⚠️  RISK FACTORS:")
            for risk in clinical['risk_factors'][:2]:  # Show first 2
                clinical_text.append(f"  • {risk}")
            clinical_text.append("")

        # Recommendations
        if clinical['recommendations']:
            clinical_text.append("💡 RECOMMENDATIONS:")
            for rec in clinical['recommendations'][:2]:  # Show first 2
                clinical_text.append(f"  • {rec}")
            clinical_text.append("")

        # Baseline comparison
        if session.baseline_comparison:
            comparison = session.baseline_comparison
            clinical_text.extend([
                "📊 BASELINE COMPARISON:",
                f"Score Change: {comparison['score_change']:+.1f}",
                f"Status: {comparison['interpretation']}",
                ""
            ])

        self._draw_panel(frame, panel_x, panel_y, panel_width, clinical_text, alpha=0.9)

    def _draw_panel(self, frame, x, y, width, text_lines, alpha=0.8):
        """Draw a semi-transparent panel with text."""
        height_needed = len(text_lines) * 25 + 30

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height_needed), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + height_needed), (0, 255, 255), 2)

        # Text
        for i, text in enumerate(text_lines):
            text_y = y + 25 + (i * 22)
            color = (0, 255, 255) if "===" in text else (255, 255, 255)
            size = 0.6 if "===" in text else 0.5
            cv2.putText(frame, text, (x + 10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

    def _handle_keypress(self, key):
        """Handle keyboard input."""
        if key == ord('q'):
            return False
        elif key == ord('s'):
            self.show_skeleton = not self.show_skeleton
            print(f"Skeleton: {'ON' if self.show_skeleton else 'OFF'}")
        elif key == ord('m'):
            self.show_metrics = not self.show_metrics
            print(f"Metrics: {'ON' if self.show_metrics else 'OFF'}")
        elif key == ord('c'):
            self.show_clinical_analysis = not self.show_clinical_analysis
            print(f"Clinical analysis: {'ON' if self.show_clinical_analysis else 'OFF'}")
        elif key == ord('b'):
            self.baseline_mode = not self.baseline_mode
            mode_text = "BASELINE MODE ON - Next walk will be baseline" if self.baseline_mode else "Baseline mode OFF"
            print(f"🎯 {mode_text}")
        elif key == ord('r'):
            self.completed_sessions = []
            self.current_session_display = None
            print("🔄 Reset session data")
        elif key == ord(' '):  # Space bar
            session = self.walk_detector.force_end_session(time.time())
            if session:
                self._process_completed_session(session)
            else:
                print("No active session to end")

        return True

    def _save_baseline(self, session: WalkSession):
        """Save session as baseline."""
        baseline_data = {
            'session_id': session.session_id,
            'timestamp': time.time(),
            'clinical_assessment': session.clinical_assessment
        }

        with open(self.baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)

    def _load_baseline(self):
        """Load baseline assessment."""
        try:
            if os.path.exists(self.baseline_file):
                with open(self.baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                return baseline_data.get('clinical_assessment')
        except Exception as e:
            print(f"Could not load baseline: {e}")
        return None

    def _save_session(self, session: WalkSession):
        """Save session to file."""
        try:
            # Convert session to dict (handle dataclass)
            session_data = {
                'session_id': session.session_id,
                'start_time': session.start_time,
                'end_time': session.end_time,
                'duration': session.duration,
                'gait_metrics': session.gait_metrics,
                'clinical_assessment': session.clinical_assessment,
                'baseline_comparison': session.baseline_comparison
            }

            # Load existing sessions
            sessions = []
            if os.path.exists(self.sessions_file):
                with open(self.sessions_file, 'r') as f:
                    sessions = json.load(f)

            # Add new session
            sessions.append(session_data)

            # Keep only last 50 sessions
            sessions = sessions[-50:]

            # Save back
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f, indent=2)

        except Exception as e:
            print(f"Could not save session: {e}")

    def _print_session_summary(self, session: WalkSession):
        """Print session summary to console."""
        clinical = session.clinical_assessment

        print("\n" + "="*60)
        print(f"🚶‍♂️ WALK SESSION COMPLETE: {session.session_id}")
        print("="*60)
        print(f"Duration: {session.duration:.1f} seconds")
        print(f"Clinical Score: {clinical['overall_score']:.0f}/100")

        if clinical['risk_factors']:
            print("\n⚠️  Risk Factors:")
            for risk in clinical['risk_factors']:
                print(f"  • {risk}")

        if clinical['recommendations']:
            print("\n💡 Recommendations:")
            for rec in clinical['recommendations']:
                print(f"  • {rec}")

        if session.baseline_comparison:
            comparison = session.baseline_comparison
            print(f"\n📊 Baseline Comparison:")
            print(f"Score Change: {comparison['score_change']:+.1f}")
            print(f"Status: {comparison['interpretation']}")

        print("="*60)

    def _cleanup(self):
        """Clean up resources."""
        cv2.destroyAllWindows()
        self.video_source.disconnect()
        self.pose_estimator.cleanup()
        print("✅ Smart gait analysis ended")


if __name__ == "__main__":
    smart_analyzer = SmartGaitAnalyzer()
    smart_analyzer.run()