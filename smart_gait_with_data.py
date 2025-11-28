"""
Enhanced Smart DJI Gait Analysis with Real Data Persistence
- Stores actual walking sessions with timestamps
- Direction detection (only records walking towards camera)
- Real trend analysis and replay capabilities
- Comprehensive flagging system
"""

import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
from camera.video_source import VideoSource
from pose.pose_estimator import PoseEstimator
from gait.gait_features import GaitAnalyzer
from gait.walk_session import WalkDetector, ClinicalGaitAnalyzer, WalkSession
from data_persistence import GaitDataManager, create_session_from_walk_data

class SmartGaitWithData:
    """
    Advanced gait analysis with real data persistence and replay.
    Perfect for pitch day with actual user data trends.
    """

    def __init__(self, subject_id: str = "user"):
        """Initialize smart gait analyzer with data persistence."""
        # Core components
        self.video_source = VideoSource(0)  # DJI USB camera (OsmoAction5pro)
        self.pose_estimator = PoseEstimator()
        self.gait_analyzer = GaitAnalyzer(fps=30.0)

        # Enhanced features with direction detection
        self.walk_detector = WalkDetector(min_walk_duration=3.0, max_gap_duration=2.0)
        self.clinical_analyzer = ClinicalGaitAnalyzer()

        # Data management
        self.data_manager = GaitDataManager("gait_sessions")
        self.subject_id = subject_id

        # Session tracking
        self.current_session_display = None
        self.show_clinical_analysis = True

        # Display settings
        self.show_skeleton = True
        self.show_metrics = True
        self.show_trends = True

        print(f"🧠 Smart Gait Analysis initialized for subject: {subject_id}")

    def run(self):
        """Start the smart gait analysis system with data collection."""
        print("\n🧠 Smart DJI Gait Analysis with Real Data Collection")
        print("=" * 60)
        print("🎯 Features:")
        print("  • Automatic walk detection (towards camera only)")
        print("  • Real data persistence and trends")
        print("  • Clinical analysis with flags")
        print("  • Session replay and comparison")
        print("")
        print("🎮 Controls:")
        print("  • 'q' - Quit")
        print("  • 's' - Toggle skeleton")
        print("  • 'm' - Toggle metrics")
        print("  • 'c' - Toggle clinical analysis")
        print("  • 't' - Toggle trends display")
        print("  • 'r' - Show recent sessions")
        print("  • 'h' - Show statistics")
        print("  • 'SPACE' - Force end current walk")
        print("")

        if not self.video_source.connect():
            print("❌ Error: Could not connect to camera")
            return

        properties = self.video_source.get_properties()
        print(f"✅ Camera connected: {properties}")

        # Update FPS based on camera
        if 'fps' in properties:
            self.gait_analyzer.fps = properties['fps']

        # Show existing data summary
        self._show_data_summary()

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n🛑 Analysis stopped by user")
        finally:
            self._cleanup()

    def _show_data_summary(self):
        """Show existing session data."""
        stats = self.data_manager.get_statistics(self.subject_id)
        print(f"\n📊 Existing Data for {self.subject_id}:")
        print(f"  • Total sessions: {stats['total_sessions']}")
        print(f"  • Average score: {stats['average_score']}/100")
        print(f"  • Average duration: {stats['average_duration']:.1f}s")
        print(f"  • High-risk sessions: {stats['high_risk_sessions']}")

        if stats['total_sessions'] > 0:
            baseline_comparison = self.data_manager.compare_to_baseline(self.subject_id)
            if baseline_comparison['status'] == 'available':
                trend = baseline_comparison['trend']
                change = baseline_comparison['score_change_percent']
                print(f"  • Recent trend: {trend} ({change:+.1f}%)")

    def _main_loop(self):
        """Main analysis loop with data persistence."""
        print(f"\n🚶‍♂️ Walk towards the camera for automatic detection...")

        frame_count = 0
        last_stats_time = time.time()

        while True:
            success, frame = self.video_source.read_frame()
            if not success:
                continue

            frame_count += 1
            current_time = time.time()

            # Run pose estimation
            pose_success, keypoints = self.pose_estimator.estimate_pose(frame)
            valid_pose = False

            if pose_success and self.pose_estimator.is_pose_valid(keypoints):
                valid_pose = True

            # Update walk detector
            completed_session = self.walk_detector.update(
                keypoints if valid_pose else None,
                current_time
            )

            # Process completed session
            if completed_session:
                self._process_completed_session(completed_session)

            # Draw visualization
            display_frame = self._draw_visualization(
                frame, keypoints if valid_pose else None, current_time
            )

            # Show frame
            cv2.imshow('🧠 Smart Gait Analysis - Real Data Collection', display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.show_skeleton = not self.show_skeleton
                print(f"Skeleton: {'ON' if self.show_skeleton else 'OFF'}")
            elif key == ord('m'):
                self.show_metrics = not self.show_metrics
                print(f"Metrics: {'ON' if self.show_metrics else 'OFF'}")
            elif key == ord('c'):
                self.show_clinical_analysis = not self.show_clinical_analysis
                print(f"Clinical Analysis: {'ON' if self.show_clinical_analysis else 'OFF'}")
            elif key == ord('t'):
                self.show_trends = not self.show_trends
                print(f"Trends: {'ON' if self.show_trends else 'OFF'}")
            elif key == ord('r'):
                self._show_recent_sessions()
            elif key == ord('h'):
                self._show_detailed_stats()
            elif key == ord(' '):
                # Force end session
                forced_session = self.walk_detector.force_end_session(current_time)
                if forced_session:
                    self._process_completed_session(forced_session)

    def _process_completed_session(self, session: WalkSession):
        """Process and save a completed walking session."""
        print(f"\n🎉 Walk completed: {session.session_id}")

        try:
            # Analyze gait metrics
            valid_keypoints = [kp for kp in session.keypoints_data if kp['keypoints'] is not None]
            keypoints_sequence = [kp['keypoints'] for kp in valid_keypoints]

            if len(keypoints_sequence) < 10:
                print("❌ Not enough valid poses for analysis")
                return

            # Run gait analysis
            gait_results = self.gait_analyzer.analyze_walk(keypoints_sequence)

            # Add keypoints sequence to results for storage
            gait_results['keypoints_sequence'] = keypoints_sequence

            # Clinical assessment
            clinical_assessment = self.clinical_analyzer.analyze_session(session, gait_results)

            # Create persistent session
            persistent_session = create_session_from_walk_data(
                gait_results, clinical_assessment, self.subject_id
            )

            # Save to database
            self.data_manager.save_session(persistent_session)

            # Store for current display
            self.current_session_display = {
                'session': persistent_session,
                'gait_results': gait_results,
                'clinical_assessment': clinical_assessment
            }

            self._print_session_summary(persistent_session, clinical_assessment)

        except Exception as e:
            print(f"❌ Error processing session: {e}")

    def _print_session_summary(self, session, clinical_assessment):
        """Print formatted session summary."""
        print("=" * 60)
        print(f"🚶‍♂️ WALK SESSION: {session.session_id}")
        print("=" * 60)
        print(f"Duration: {session.duration:.1f} seconds")
        print(f"Risk Level: {session.risk_level.upper()}")
        print(f"Clinical Score: {clinical_assessment['overall_score']:.0f}/100")

        if session.flags:
            print(f"\n⚠️  Flags Detected:")
            for flag in session.flags:
                print(f"  • {flag}")

        if clinical_assessment.get('recommendations'):
            print(f"\n💡 Recommendations:")
            for rec in clinical_assessment['recommendations']:
                print(f"  • {rec}")

        # Compare to baseline if available
        baseline_comparison = self.data_manager.compare_to_baseline(self.subject_id)
        if baseline_comparison['status'] == 'available':
            print(f"\n📈 Trend Analysis:")
            print(f"  • Baseline score: {baseline_comparison['baseline_score']}/100")
            print(f"  • Recent average: {baseline_comparison['recent_score']}/100")
            print(f"  • Change: {baseline_comparison['score_change_percent']:+.1f}%")
            print(f"  • Trend: {baseline_comparison['trend'].upper()}")

    def _show_recent_sessions(self):
        """Display recent sessions summary."""
        sessions = self.data_manager.get_recent_sessions(self.subject_id, days=7, limit=10)

        print(f"\n📅 Recent Sessions for {self.subject_id}:")
        print("-" * 50)

        if not sessions:
            print("No recent sessions found.")
            return

        for session in sessions:
            date_str = session['timestamp'].split('T')[0]
            time_str = session['timestamp'].split('T')[1][:8]
            risk_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}.get(session['risk_level'], '⚪')

            print(f"{risk_emoji} {date_str} {time_str} | "
                  f"Score: {session['clinical_score']:.0f}/100 | "
                  f"Duration: {session['duration']:.1f}s | "
                  f"Flags: {session['flags_count']}")

    def _show_detailed_stats(self):
        """Show detailed statistics."""
        stats = self.data_manager.get_statistics(self.subject_id)

        print(f"\n📊 Detailed Statistics for {self.subject_id}:")
        print("-" * 50)
        print(f"Total sessions: {stats['total_sessions']}")
        print(f"Average clinical score: {stats['average_score']}/100")
        print(f"Average walk duration: {stats['average_duration']:.1f} seconds")
        print(f"High-risk sessions: {stats['high_risk_sessions']}")
        print(f"Total flags raised: {stats['total_flags']}")

        if stats['total_sessions'] >= 3:
            trend_data = self.data_manager.get_trend_data(self.subject_id, days=30)
            if trend_data['clinical_scores']:
                recent_scores = trend_data['clinical_scores'][-5:]  # Last 5 sessions
                trend_direction = "improving" if recent_scores[-1] > recent_scores[0] else "declining"
                print(f"Recent trend: {trend_direction}")

    def _draw_visualization(self, frame, keypoints, current_time):
        """Draw comprehensive visualization overlay."""
        display_frame = frame.copy()
        h, w = frame.shape[:2]

        # Draw skeleton if valid pose
        if keypoints and self.show_skeleton:
            display_frame = self._draw_enhanced_skeleton(display_frame, keypoints)

        # Draw metrics panel
        if self.show_metrics:
            self._draw_metrics_panel(display_frame, current_time)

        # Draw session info if available
        if self.current_session_display and self.show_clinical_analysis:
            self._draw_clinical_panel(display_frame)

        # Draw trends if available
        if self.show_trends:
            self._draw_trends_panel(display_frame)

        return display_frame

    def _draw_enhanced_skeleton(self, frame, keypoints):
        """Draw enhanced skeleton with direction indication."""
        h, w = frame.shape[:2]

        # Convert to pixel coordinates
        pixel_joints = {}
        for joint_name, joint_data in keypoints.items():
            if joint_data['visibility'] > 0.5:
                x = int(joint_data['x'] * w)
                y = int(joint_data['y'] * h)
                pixel_joints[joint_name] = (x, y)

        # Draw connections
        connections = [
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle')
        ]

        # Draw lines (cyan for good direction detection)
        for joint1, joint2 in connections:
            if joint1 in pixel_joints and joint2 in pixel_joints:
                cv2.line(frame, pixel_joints[joint1], pixel_joints[joint2], (255, 255, 0), 3)

        # Draw joints with confidence indication
        for joint_name, (x, y) in pixel_joints.items():
            confidence = keypoints[joint_name]['visibility']
            color_intensity = int(confidence * 255)
            cv2.circle(frame, (x, y), 8, (0, color_intensity, 0), -1)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)

        # Direction indicator
        if 'left_hip' in pixel_joints and 'right_hip' in pixel_joints:
            hip_center_x = (pixel_joints['left_hip'][0] + pixel_joints['right_hip'][0]) // 2
            hip_center_y = (pixel_joints['left_hip'][1] + pixel_joints['right_hip'][1]) // 2

            # Green arrow indicating "walking towards camera detected"
            cv2.arrowedLine(frame, (hip_center_x, hip_center_y - 30),
                          (hip_center_x, hip_center_y - 60), (0, 255, 0), 3)
            cv2.putText(frame, "TOWARDS CAMERA", (hip_center_x - 80, hip_center_y - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def _draw_metrics_panel(self, frame, current_time):
        """Draw real-time metrics panel."""
        h, w = frame.shape[:2]

        # Panel background
        panel_w, panel_h = 300, 200
        panel_x, panel_y = w - panel_w - 10, 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h), (255, 255, 0), 2)

        # Content
        y_offset = panel_y + 25
        cv2.putText(frame, "=== REAL DATA COLLECTION ===",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        y_offset += 25
        if self.walk_detector.is_recording_session:
            duration = current_time - self.walk_detector.current_session_start
            cv2.putText(frame, f"🔴 Recording: {duration:.1f}s",
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "⚪ Ready to detect walk",
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        y_offset += 25
        stats = self.data_manager.get_statistics(self.subject_id)
        cv2.putText(frame, f"Sessions today: {stats['total_sessions']}",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"Avg score: {stats['average_score']:.1f}/100",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_clinical_panel(self, frame):
        """Draw clinical analysis panel."""
        if not self.current_session_display:
            return

        h, w = frame.shape[:2]
        session_data = self.current_session_display

        # Panel position (bottom left)
        panel_w, panel_h = 400, 150
        panel_x, panel_y = 10, h - panel_h - 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        risk_color = {'low': (0, 255, 0), 'medium': (0, 255, 255), 'high': (0, 0, 255)}
        border_color = risk_color.get(session_data['session'].risk_level, (255, 255, 255))
        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h), border_color, 2)

        # Content
        y_offset = panel_y + 25
        cv2.putText(frame, f"Last Session: {session_data['session'].session_id}",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        y_offset += 25
        score = session_data['clinical_assessment']['overall_score']
        cv2.putText(frame, f"Clinical Score: {score:.0f}/100",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2)

        y_offset += 25
        risk = session_data['session'].risk_level.upper()
        cv2.putText(frame, f"Risk Level: {risk}",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2)

        if session_data['session'].flags:
            y_offset += 25
            flags_text = f"Flags: {len(session_data['session'].flags)}"
            cv2.putText(frame, flags_text,
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def _draw_trends_panel(self, frame):
        """Draw trends panel."""
        stats = self.data_manager.get_statistics(self.subject_id)
        if stats['total_sessions'] < 3:
            return

        baseline_comparison = self.data_manager.compare_to_baseline(self.subject_id)
        if baseline_comparison['status'] != 'available':
            return

        h, w = frame.shape[:2]

        # Panel position (top left)
        panel_w, panel_h = 250, 100
        panel_x, panel_y = 10, 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        trend = baseline_comparison['trend']
        trend_color = (0, 255, 0) if trend == 'improving' else (0, 0, 255) if trend == 'declining' else (255, 255, 255)
        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h), trend_color, 2)

        # Content
        y_offset = panel_y + 25
        cv2.putText(frame, "=== TREND ANALYSIS ===",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        y_offset += 25
        change = baseline_comparison['score_change_percent']
        cv2.putText(frame, f"vs Baseline: {change:+.1f}%",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, trend_color, 2)

        y_offset += 25
        cv2.putText(frame, f"Status: {trend.upper()}",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, trend_color, 2)

    def _cleanup(self):
        """Clean up resources."""
        self.video_source.disconnect()
        self.pose_estimator.cleanup()
        cv2.destroyAllWindows()
        print("✅ Cleanup completed")

def main():
    """Run the enhanced smart gait analysis."""
    analyzer = SmartGaitWithData(subject_id="user")
    analyzer.run()

if __name__ == "__main__":
    main()