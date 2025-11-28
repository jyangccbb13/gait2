"""
Video-based gait analysis tool.
Upload a walking video and get output with pose overlay and gait metrics.
"""

import cv2
import numpy as np
import os
import json
from pose.pose_estimator import PoseEstimator
from gait.gait_features import GaitAnalyzer
from gait.baseline import BaselineModel


class VideoGaitAnalyzer:
    """
    Analyzes walking videos and outputs results with pose overlay.
    """

    def __init__(self):
        """Initialize video analyzer."""
        self.pose_estimator = PoseEstimator()
        self.gait_analyzer = GaitAnalyzer()
        self.baseline_model = BaselineModel()

        # Output settings
        self.output_dir = "output_videos"
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_video(self, video_path: str, output_name: str = None, show_preview: bool = True):
        """
        Analyze a walking video and create output with pose overlay.

        Args:
            video_path: Path to input video file
            output_name: Name for output video (optional)
            show_preview: Whether to show real-time preview
        """
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file not found: {video_path}")
            return None

        # Generate output filename
        if output_name is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_name = f"{base_name}_gait_analysis.mp4"

        output_path = os.path.join(self.output_dir, output_name)

        print(f"🎥 Analyzing video: {video_path}")
        print(f"📊 Output will be saved to: {output_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error: Could not open video file")
            return None

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📹 Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")

        # Update gait analyzer with actual FPS
        self.gait_analyzer.fps = fps

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print("❌ Error: Could not create output video file")
            cap.release()
            return None

        # Analysis variables
        keypoint_sequence = []
        frame_count = 0
        valid_poses = 0

        print("🔄 Processing frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run pose estimation
            pose_success, keypoints = self.pose_estimator.estimate_pose(frame)

            if pose_success and self.pose_estimator.is_pose_valid(keypoints):
                keypoint_sequence.append(keypoints)
                valid_poses += 1
            else:
                keypoint_sequence.append(None)

            # Create output frame with overlay
            output_frame = self._create_analysis_frame(
                frame, keypoints if pose_success else None,
                frame_count, total_frames, keypoint_sequence
            )

            # Write to output video
            out.write(output_frame)

            # Show preview if requested
            if show_preview and frame_count % 10 == 0:  # Show every 10th frame
                preview = cv2.resize(output_frame, (640, 360))
                cv2.imshow('Processing Preview (ESC to skip preview)', preview)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    show_preview = False
                    cv2.destroyAllWindows()

            # Progress update
            if frame_count % 30 == 0:  # Every second
                progress = (frame_count / total_frames) * 100
                print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"✅ Video processing complete!")
        print(f"📊 Analysis results:")
        print(f"  - Total frames: {frame_count}")
        print(f"  - Valid poses: {valid_poses} ({valid_poses/frame_count*100:.1f}%)")
        print(f"  - Output video: {output_path}")

        # Analyze gait metrics
        if valid_poses > 30:  # Need enough data
            self._analyze_and_save_metrics(keypoint_sequence, output_name)

        return output_path

    def _create_analysis_frame(self, frame, keypoints, frame_idx, total_frames, keypoint_history):
        """Create frame with pose overlay and metrics."""
        output_frame = frame.copy()

        # Draw skeleton overlay
        if keypoints is not None:
            output_frame = self._draw_skeleton(output_frame, keypoints)
            self._draw_confidence_indicator(output_frame, keypoints)

        # Draw progress and frame info
        self._draw_progress_bar(output_frame, frame_idx, total_frames)

        # Draw gait metrics (if enough data)
        if len([k for k in keypoint_history if k is not None]) > 30:
            self._draw_gait_metrics(output_frame, keypoint_history)

        return output_frame

    def _draw_skeleton(self, frame, keypoints):
        """Draw skeleton overlay on frame."""
        height, width = frame.shape[:2]

        # Convert to pixel coordinates
        pixel_joints = {}
        for joint_name, joint_data in keypoints.items():
            if joint_data['visibility'] > 0.5:
                x = int(joint_data['x'] * width)
                y = int(joint_data['y'] * height)
                pixel_joints[joint_name] = (x, y, joint_data['visibility'])

        # Draw connections
        connections = [
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle')
        ]

        # Draw bones (blue lines)
        for joint1, joint2 in connections:
            if joint1 in pixel_joints and joint2 in pixel_joints:
                pt1 = pixel_joints[joint1][:2]
                pt2 = pixel_joints[joint2][:2]

                # Line thickness based on confidence
                avg_conf = (pixel_joints[joint1][2] + pixel_joints[joint2][2]) / 2
                thickness = max(2, int(avg_conf * 6))

                cv2.line(frame, pt1, pt2, (255, 100, 0), thickness)

        # Draw joints (green circles)
        for joint_name, (x, y, visibility) in pixel_joints.items():
            radius = max(4, int(visibility * 12))

            cv2.circle(frame, (x, y), radius, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)

            # Joint labels
            if visibility > 0.7:
                label = joint_name.split('_')[1][:4].upper()  # "LEFT" -> "LEFT", "ankle" -> "ANKL"
                cv2.putText(frame, label, (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def _draw_confidence_indicator(self, frame, keypoints):
        """Draw confidence indicator."""
        confidences = [joint['visibility'] for joint in keypoints.values()]
        avg_confidence = sum(confidences) / len(confidences)

        # Choose color based on confidence
        if avg_confidence > 0.8:
            color = (0, 255, 0)  # Green
            status = "HIGH"
        elif avg_confidence > 0.5:
            color = (0, 255, 255)  # Yellow
            status = "MED"
        else:
            color = (0, 0, 255)  # Red
            status = "LOW"

        # Draw indicator
        cv2.circle(frame, (50, 50), 25, color, -1)
        cv2.circle(frame, (50, 50), 25, (255, 255, 255), 2)

        cv2.putText(frame, status, (85, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{avg_confidence:.2f}", (85, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _draw_progress_bar(self, frame, current_frame, total_frames):
        """Draw progress bar at bottom."""
        height, width = frame.shape[:2]

        progress = current_frame / total_frames
        bar_width = width - 40
        bar_height = 10
        bar_x = 20
        bar_y = height - 30

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

        # Progress
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)

        # Text
        progress_text = f"Frame {current_frame}/{total_frames} ({progress:.1%})"
        cv2.putText(frame, progress_text, (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_gait_metrics(self, frame, keypoint_history):
        """Draw current gait metrics."""
        height, width = frame.shape[:2]

        try:
            # Analyze recent keypoints
            valid_keypoints = [k for k in keypoint_history[-90:] if k is not None]  # Last 3 seconds

            if len(valid_keypoints) < 10:
                return

            # Quick analysis
            speeds, avg_speed = self.gait_analyzer.compute_gait_speed(valid_keypoints)
            stride_events = self.gait_analyzer.detect_stride_events(valid_keypoints)

            metrics_text = [
                "=== GAIT ANALYSIS ===",
                f"Poses: {len(valid_keypoints)}/90",
                f"Avg Speed: {avg_speed:.4f}",
                f"Left Strides: {len(stride_events['left'])}",
                f"Right Strides: {len(stride_events['right'])}"
            ]

            # Recent speed trend
            if len(speeds) > 5:
                recent_speed = sum(speeds[-5:]) / 5
                trend = "↑" if recent_speed > avg_speed else "↓"
                metrics_text.append(f"Recent: {recent_speed:.4f} {trend}")

        except Exception as e:
            metrics_text = [
                "=== GAIT ANALYSIS ===",
                f"Analysis error:",
                f"{str(e)[:20]}..."
            ]

        # Draw metrics panel
        panel_width = 250
        panel_height = len(metrics_text) * 25 + 20
        panel_x = width - panel_width - 20
        panel_y = 20

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 255, 255), 2)

        # Text
        for i, text in enumerate(metrics_text):
            y_pos = panel_y + 20 + (i * 20)
            color = (0, 255, 255) if "===" in text else (255, 255, 255)
            cv2.putText(frame, text, (panel_x + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def _analyze_and_save_metrics(self, keypoint_sequence, output_name):
        """Analyze full video and save metrics to JSON."""
        print("📊 Analyzing complete gait pattern...")

        try:
            # Full gait analysis
            gait_results = self.gait_analyzer.analyze_walk(keypoint_sequence)

            # Save results to JSON
            results_path = os.path.join(self.output_dir, f"{os.path.splitext(output_name)[0]}_metrics.json")

            analysis_summary = {
                'video_analysis': {
                    'duration_seconds': gait_results['duration'],
                    'total_frames': gait_results['frame_count'],
                    'fps': gait_results['fps'],
                    'valid_poses': len([k for k in keypoint_sequence if k is not None])
                },
                'gait_metrics': {
                    'average_speed': gait_results['average_speed'],
                    'stride_metrics': gait_results['stride_metrics'],
                    'stride_events': {
                        'left_count': len(gait_results['stride_events']['left']),
                        'right_count': len(gait_results['stride_events']['right']),
                        'total_strides': len(gait_results['stride_events']['left']) + len(gait_results['stride_events']['right'])
                    }
                }
            }

            with open(results_path, 'w') as f:
                json.dump(analysis_summary, f, indent=2)

            print(f"📄 Metrics saved to: {results_path}")
            print("\n📊 GAIT ANALYSIS SUMMARY:")
            print(f"  Duration: {gait_results['duration']:.1f} seconds")
            print(f"  Average Speed: {gait_results['average_speed']:.4f} units/sec")
            print(f"  Total Strides: {analysis_summary['gait_metrics']['stride_events']['total_strides']}")
            print(f"  Stride Variability: {gait_results['stride_metrics'].get('stride_time_variability', 'N/A')}")

        except Exception as e:
            print(f"❌ Error in gait analysis: {e}")


def main():
    """Main function for video analysis."""
    print("🎥 Video Gait Analysis Tool")
    print("=" * 50)

    # Create analyzer
    analyzer = VideoGaitAnalyzer()

    # Ask for video file
    while True:
        video_path = input("\n📁 Enter path to walking video (or 'demo' for sample): ").strip()

        if video_path.lower() == 'demo':
            # Create a demo instruction
            print("\n📋 Demo Instructions:")
            print("1. Download any walking video from YouTube or record one")
            print("2. Save it in the current directory")
            print("3. Run this script again with the video filename")
            print("\nSuggested search: 'person walking side view' or 'gait analysis sample video'")
            return

        if video_path.lower() == 'quit':
            return

        if os.path.exists(video_path):
            break
        else:
            print(f"❌ File not found: {video_path}")
            print("💡 Try entering the full path or 'demo' for instructions")

    # Analyze video
    output_path = analyzer.analyze_video(video_path, show_preview=True)

    if output_path:
        print(f"\n🎉 Success! Check the output:")
        print(f"  📺 Video: {output_path}")
        print(f"  📊 Metrics: {output_path.replace('.mp4', '_metrics.json')}")


if __name__ == "__main__":
    main()