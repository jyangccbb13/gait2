# Gait Analysis System

A passive gait monitoring system for neurological and elderly patients using computer vision and pose estimation.

## Overview

This system captures walking patterns using a single camera, extracts gait metrics through pose estimation, and provides a web-based dashboard for analysis and baseline comparison.

### Key Features

- **Video Input**: Supports webcam and RTSP/RTMP streams (DJI Osmo Action 5 compatible)
- **Pose Estimation**: MediaPipe BlazePose for robust keypoint detection
- **Gait Metrics**: Speed, stride length, temporal variability analysis
- **Baseline Modeling**: Compare walks against historical patterns
- **Web Dashboard**: Real-time visualization with Chart.js
- **Skeleton Overlay**: Visual validation of pose estimation

## System Architecture

```
gait/
├── camera/           # Video input handling
├── pose/            # Pose estimation with MediaPipe
├── gait/            # Gait analysis and baseline modeling
├── app/             # Flask web application
├── data/            # Stored walks and baseline data
└── skeleton_overlay.py  # Visualization tools
```

## Requirements

### Hardware
- Camera: Webcam or IP camera (DJI Osmo Action 5 supported)
- CPU: Modern multi-core processor (GPU not required)
- Memory: 4GB RAM minimum, 8GB recommended

### Software
- Python 3.8+
- OpenCV
- MediaPipe
- Flask
- NumPy, SciPy

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd gait
```

2. **Create a virtual environment:**
```bash
python -m venv gait_env
source gait_env/bin/activate  # On Windows: gait_env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify camera access:**
```bash
python camera/video_source.py
```

## Quick Start

1. **Start the web application:**
```bash
cd app
python server.py
```

2. **Open dashboard:**
Navigate to `http://localhost:5000` in your browser

3. **Record a walk:**
   - Set duration (5-60 seconds)
   - Click "Start Recording"
   - Walk naturally in front of camera
   - Click "Analyze Walk" when recording completes

4. **View results:**
   - Gait metrics displayed in dashboard
   - Speed and variability charts updated
   - Walk added to history and baseline

## Usage Guide

### Recording Walks

1. **Camera Setup:**
   - Position camera at hip height, 2-3 meters from walking path
   - Ensure good lighting and minimal background clutter
   - Test camera view with "Start Recording" (short duration)

2. **Recording Process:**
   - Walk naturally at normal pace
   - Stay within camera view
   - Avoid loose clothing that might obscure joints
   - 10-15 seconds typically provides sufficient data

3. **Analysis:**
   - System automatically detects pose keypoints
   - Extracts gait speed, stride timing, and variability
   - Compares against baseline (after 3+ walks recorded)

### Understanding Metrics

**Gait Speed**: Average forward velocity during walk
- Units: Normalized coordinate changes per second
- Higher values indicate faster walking

**Stride Metrics**:
- *Length*: Distance covered per stride
- *Timing*: Duration between stride events
- *Variability*: Consistency of stride patterns (lower = more consistent)

**Baseline Comparison**: Z-scores showing deviation from personal baseline
- **Green bars**: Normal variation (|z| < 1)
- **Orange bars**: Mild deviation (1 < |z| < 2)
- **Red bars**: Significant deviation (|z| > 2)

### Dashboard Features

**Recording Controls:**
- Duration: Set recording length (5-60 seconds)
- Subject ID: Track multiple patients/subjects
- Real-time recording status and frame counter

**Analysis Display:**
- Metric cards with key measurements
- Speed-over-time chart showing gait pattern
- Baseline comparison highlighting deviations
- Walk history with trend analysis

**Data Management:**
- View historical walks
- Reset baseline for new analysis period
- Export data (JSON format in `data/` directory)

## Advanced Features

### Video Source Configuration

**Webcam Setup:**
```python
# Default webcam (index 0)
video_source = VideoSource(0)

# Specific camera device
video_source = VideoSource(1)  # Second camera
```

**RTSP Stream Setup:**
```python
# DJI Osmo Action 5 or IP camera
rtsp_url = "rtsp://192.168.1.100:8554/stream"
video_source = VideoSource(rtsp_url)
```

### Skeleton Overlay Visualization

Create videos with pose overlay for visual validation:

```bash
python skeleton_overlay.py
```

This generates test videos showing:
- Real-time pose keypoint detection
- Skeleton connections overlay
- Confidence indicators
- Recording progress

### Custom Gait Analysis

```python
from gait import GaitAnalyzer, BaselineModel

# Initialize analyzer
analyzer = GaitAnalyzer(fps=30.0)

# Analyze keypoint sequence
results = analyzer.analyze_walk(keypoint_sequence)

# Compare to baseline
baseline = BaselineModel()
z_scores = baseline.compare_to_baseline(results)
```

## API Reference

### REST Endpoints

**Recording Control:**
- `POST /api/record/start` - Start recording
- `POST /api/record/stop` - Stop recording
- `POST /api/record/analyze` - Analyze recorded walk

**Data Access:**
- `GET /api/walks` - Get walk history
- `GET /api/baseline` - Get baseline statistics
- `POST /api/baseline/reset` - Reset baseline

**System Status:**
- `GET /api/status` - Check system health

### Response Formats

**Walk Analysis Result:**
```json
{
  "walk_id": "default_20231124_143022",
  "gait_metrics": {
    "average_speed": 0.0423,
    "duration": 15.2,
    "stride_metrics": {
      "stride_time_variability": 0.12,
      "stride_length_variability": 0.08
    }
  },
  "baseline_comparison": {
    "average_speed": 0.5,
    "stride_time_variability": -1.2
  }
}
```

## Troubleshooting

### Camera Issues
- **No video feed**: Check camera permissions and USB connection
- **Poor pose detection**: Improve lighting, reduce background clutter
- **Frame drops**: Lower resolution or close other camera applications

### Analysis Issues
- **No strides detected**: Ensure full body visible, normal walking pace
- **Inconsistent metrics**: Check for camera shake, consistent lighting
- **High variability**: May indicate actual gait irregularities

### Performance Issues
- **Slow analysis**: Close unnecessary applications, use CPU optimization
- **Memory errors**: Reduce recording duration, restart application
- **Network issues**: Check firewall settings for Flask server

## Development

### Project Structure

```python
# Core modules
from camera import VideoSource          # Video input handling
from pose import PoseEstimator          # MediaPipe pose estimation
from gait import GaitAnalyzer, BaselineModel  # Analysis and modeling

# Web application
from app.server import app              # Flask backend
```

### Testing

Run individual module tests:
```bash
python camera/video_source.py          # Camera functionality
python pose/pose_estimator.py          # Pose estimation
python gait/gait_features.py          # Gait analysis
python skeleton_overlay.py            # Visualization
```

### Extending the System

**Adding New Metrics:**
1. Implement in `gait/gait_features.py`
2. Update baseline tracking in `gait/baseline.py`
3. Add visualization in dashboard

**New Video Sources:**
1. Extend `VideoSource` class in `camera/video_source.py`
2. Add configuration options in Flask app

**Custom Pose Models:**
1. Create new estimator in `pose/` directory
2. Implement same interface as `PoseEstimator`

## Clinical Considerations

- This system provides research-grade metrics, not medical diagnosis
- Baseline establishment requires 3+ consistent walks
- Environmental factors (lighting, surface, clothing) affect measurements
- Regular recalibration recommended for longitudinal monitoring
- Consider privacy and data security in clinical deployments

## License

[Add appropriate license information]

## Support

For technical issues and feature requests, please create an issue in the project repository.