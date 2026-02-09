"""
Flask backend for gait analysis system.
Provides API endpoints for recording walks, analyzing gait, and serving dashboard.
"""

import os
import sys
import json
import threading
import time
from flask import Flask, render_template, jsonify, request, Response
from datetime import datetime

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.video_source import VideoSource
from pose.pose_estimator import PoseEstimator
from gait.gait_features import GaitAnalyzer
from gait.baseline import BaselineModel


app = Flask(__name__)

# User settings
user_settings = {'person_height_m': None}

# Global objects for video processing
video_source = None
pose_estimator = None
gait_analyzer = None
baseline_model = None

# Recording state
recording_state = {
    'is_recording': False,
    'current_frames': [],
    'current_keypoints': [],
    'start_time': None,
    'recording_thread': None
}


def initialize_components():
    """Initialize all gait analysis components."""
    global video_source, pose_estimator, gait_analyzer, baseline_model

    try:
        # Initialize video source (MacBook camera for testing)
        video_source = VideoSource(0)

        # Initialize pose estimator
        pose_estimator = PoseEstimator()

        # Initialize gait analyzer (30 FPS default)
        gait_analyzer = GaitAnalyzer(fps=30.0)

        # Initialize baseline model
        baseline_model = BaselineModel(data_dir="data")

        print("All components initialized successfully")
        return True

    except Exception as e:
        print(f"Error initializing components: {e}")
        return False


def recording_worker(duration_seconds=15):
    """Worker function for recording video in a separate thread."""
    global recording_state, video_source, pose_estimator

    recording_state['current_frames'] = []
    recording_state['current_keypoints'] = []
    recording_state['start_time'] = time.time()

    try:
        # Connect to video source
        if not video_source.connect():
            recording_state['is_recording'] = False
            return

        # Update FPS from actual video source
        actual_fps = video_source.get_properties()['fps']
        gait_analyzer.fps = actual_fps

        total_frames = int(duration_seconds * actual_fps)
        frames_recorded = 0

        print(f"Starting recording: {duration_seconds}s at {actual_fps} FPS ({total_frames} frames)")

        while (recording_state['is_recording'] and
               frames_recorded < total_frames):

            # Read frame
            success, frame = video_source.read_frame()
            if not success:
                print("Failed to read frame, stopping recording")
                break

            # Store frame
            recording_state['current_frames'].append(frame)

            # Run pose estimation
            pose_success, keypoints = pose_estimator.estimate_pose(frame)

            if pose_success and pose_estimator.is_pose_valid(keypoints):
                recording_state['current_keypoints'].append(keypoints)
            else:
                recording_state['current_keypoints'].append(None)

            frames_recorded += 1

            # Small delay to maintain frame rate
            time.sleep(1.0 / actual_fps)

        # Disconnect video source
        video_source.disconnect()

        print(f"Recording completed: {frames_recorded} frames")

    except Exception as e:
        print(f"Error during recording: {e}")

    finally:
        recording_state['is_recording'] = False


@app.route('/')
def dashboard():
    """Serve the main dashboard."""
    return render_template('dashboard.html')


@app.route('/enterprise')
def enterprise_dashboard():
    """Serve the enterprise SaaS dashboard."""
    with open('enterprise_dashboard.html', 'r') as f:
        return f.read()


@app.route('/api/status')
def get_status():
    """Get current system status."""
    return jsonify({
        'recording': recording_state['is_recording'],
        'video_connected': video_source.is_connected if video_source else False,
        'components_initialized': all([
            video_source is not None,
            pose_estimator is not None,
            gait_analyzer is not None,
            baseline_model is not None
        ]),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/record/start', methods=['POST'])
def start_recording():
    """Start recording a new walk."""
    global recording_state

    if recording_state['is_recording']:
        return jsonify({'error': 'Recording already in progress'}), 400

    # Get duration from request (default 15 seconds)
    data = request.get_json() or {}
    duration = data.get('duration', 15)

    # Validate duration
    if duration < 1 or duration > 60:
        return jsonify({'error': 'Duration must be between 1 and 60 seconds'}), 400

    # Start recording in background thread
    recording_state['is_recording'] = True
    recording_state['recording_thread'] = threading.Thread(
        target=recording_worker,
        args=(duration,)
    )
    recording_state['recording_thread'].start()

    return jsonify({
        'message': 'Recording started',
        'duration': duration,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/record/stop', methods=['POST'])
def stop_recording():
    """Stop current recording."""
    global recording_state

    if not recording_state['is_recording']:
        return jsonify({'error': 'No recording in progress'}), 400

    recording_state['is_recording'] = False

    # Wait for recording thread to finish
    if recording_state['recording_thread']:
        recording_state['recording_thread'].join(timeout=5)

    return jsonify({
        'message': 'Recording stopped',
        'frames_recorded': len(recording_state['current_frames']),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update user settings (e.g. height)."""
    global user_settings, gait_analyzer

    data = request.get_json() or {}
    height_cm = data.get('height_cm')

    if height_cm is not None:
        try:
            height_cm = float(height_cm)
        except (ValueError, TypeError):
            return jsonify({'error': 'height_cm must be a number'}), 400

        if height_cm < 100 or height_cm > 250:
            return jsonify({'error': 'height_cm must be between 100 and 250'}), 400

        user_settings['person_height_m'] = height_cm / 100.0
        if gait_analyzer:
            gait_analyzer.person_height_m = user_settings['person_height_m']
    else:
        user_settings['person_height_m'] = None
        if gait_analyzer:
            gait_analyzer.person_height_m = None

    return jsonify({
        'message': 'Settings updated',
        'settings': {
            'person_height_m': user_settings['person_height_m'],
        }
    })


@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current user settings."""
    return jsonify({
        'settings': {
            'person_height_m': user_settings['person_height_m'],
        }
    })


@app.route('/api/record/analyze', methods=['POST'])
def analyze_recording():
    """Analyze the most recent recording."""
    global recording_state, gait_analyzer, baseline_model

    if recording_state['is_recording']:
        return jsonify({'error': 'Recording in progress, cannot analyze'}), 400

    if not recording_state['current_keypoints']:
        return jsonify({'error': 'No recording data available'}), 400

    try:
        # Analyze the gait
        gait_results = gait_analyzer.analyze_walk(recording_state['current_keypoints'])

        # Get subject ID from request
        data = request.get_json() or {}
        subject_id = data.get('subject_id', 'default')

        # Compare to baseline
        z_scores = baseline_model.compare_to_baseline(gait_results, subject_id)

        # Add to baseline history
        walk_id = baseline_model.add_walk(gait_results, subject_id)

        # Prepare response
        analysis_result = {
            'walk_id': walk_id,
            'subject_id': subject_id,
            'timestamp': datetime.now().isoformat(),
            'gait_metrics': {
                'average_speed': gait_results.get('average_speed', 0.0),
                'duration': gait_results['duration'],
                'frame_count': gait_results['frame_count'],
                'stride_metrics': gait_results.get('stride_metrics', {}),
                'metrics': gait_results.get('metrics', {}),
            },
            'baseline_comparison': z_scores,
            'speeds_over_time': gait_results.get('speeds', []),
            'stride_events': gait_results.get('stride_events', {}),
            'gait_events': gait_results.get('gait_events', {}),
        }

        # Clear recording data
        recording_state['current_frames'] = []
        recording_state['current_keypoints'] = []

        return jsonify(analysis_result)

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/api/walks')
def get_walks():
    """Get list of recorded walks."""
    subject_id = request.args.get('subject_id', 'default')
    limit = int(request.args.get('limit', 20))

    walks = baseline_model.get_walk_history(subject_id, limit)

    return jsonify({
        'walks': walks,
        'subject_id': subject_id,
        'count': len(walks)
    })


@app.route('/api/walks/<walk_id>')
def get_walk_details(walk_id):
    """Get detailed information for a specific walk."""
    # This would require storing walk details by ID
    # For now, return error as this is not implemented in BaselineModel
    return jsonify({'error': 'Walk details by ID not implemented yet'}), 501


@app.route('/api/baseline')
def get_baseline():
    """Get baseline statistics."""
    subject_id = request.args.get('subject_id', 'default')

    baseline_summary = baseline_model.get_baseline_summary(subject_id)

    if baseline_summary is None:
        return jsonify({'error': 'No baseline data available'}), 404

    return jsonify(baseline_summary)


@app.route('/api/baseline/reset', methods=['POST'])
def reset_baseline():
    """Reset baseline for a subject."""
    data = request.get_json() or {}
    subject_id = data.get('subject_id', 'default')

    baseline_model.reset_baseline(subject_id)

    return jsonify({
        'message': f'Baseline reset for subject {subject_id}',
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


def main():
    """Main function to start the Flask server."""
    print("Initializing Gait Analysis System...")

    # Initialize components
    if not initialize_components():
        print("Failed to initialize components. Exiting.")
        return

    print("Starting Flask server...")
    print("Dashboard will be available at: http://localhost:8000")
    print("API endpoints:")
    print("  POST /api/record/start - Start recording")
    print("  POST /api/record/stop - Stop recording")
    print("  POST /api/record/analyze - Analyze recording")
    print("  GET  /api/walks - Get walk history")
    print("  GET  /api/baseline - Get baseline statistics")
    print("  POST /api/baseline/reset - Reset baseline")
    print("  GET  /api/settings - Get current settings")
    print("  POST /api/settings - Update settings (height_cm)")

    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=False,  # Set to True for development
        threaded=True
    )


if __name__ == '__main__':
    main()