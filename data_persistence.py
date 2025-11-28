"""
Data persistence system for real gait analysis sessions.
Stores actual walking data with timestamps, enables replay and trend analysis.
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import sqlite3
import hashlib

@dataclass
class GaitSession:
    """Complete gait session with all data needed for analysis and replay."""
    session_id: str
    timestamp: str
    subject_id: str
    duration: float
    raw_keypoints: List[Dict]
    gait_metrics: Dict[str, Any]
    clinical_assessment: Dict[str, Any]
    flags: List[str]
    risk_level: str  # 'low', 'medium', 'high'
    notes: Optional[str] = None

class GaitDataManager:
    """
    Manages storage and retrieval of gait analysis sessions.
    Enables real data collection, trends, and replay functionality.
    """

    def __init__(self, data_dir: str = "gait_sessions"):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "sessions.db")
        self._ensure_data_directory()
        self._init_database()

    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist."""
        os.makedirs(self.data_dir, exist_ok=True)

    def _init_database(self):
        """Initialize SQLite database for session metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    duration REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    clinical_score REAL,
                    flags_count INTEGER,
                    notes TEXT,
                    data_file TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON sessions(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_subject ON sessions(subject_id)
            """)

    def save_session(self, session: GaitSession) -> bool:
        """Save a complete gait session."""
        try:
            # Save detailed data to JSON file
            data_filename = f"{session.session_id}.json"
            data_path = os.path.join(self.data_dir, data_filename)

            session_data = asdict(session)
            with open(data_path, 'w') as f:
                json.dump(session_data, f, indent=2)

            # Save metadata to database
            clinical_score = session.clinical_assessment.get('overall_score', 0)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO sessions
                    (session_id, timestamp, subject_id, duration, risk_level,
                     clinical_score, flags_count, notes, data_file)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.timestamp,
                    session.subject_id,
                    session.duration,
                    session.risk_level,
                    clinical_score,
                    len(session.flags),
                    session.notes,
                    data_filename
                ))

            print(f"✅ Session saved: {session.session_id}")
            return True

        except Exception as e:
            print(f"❌ Error saving session: {e}")
            return False

    def get_session(self, session_id: str) -> Optional[GaitSession]:
        """Load a specific session by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT data_file FROM sessions WHERE session_id = ?
                """, (session_id,))
                result = cursor.fetchone()

                if not result:
                    return None

                data_path = os.path.join(self.data_dir, result[0])

                with open(data_path, 'r') as f:
                    session_data = json.load(f)

                return GaitSession(**session_data)

        except Exception as e:
            print(f"❌ Error loading session {session_id}: {e}")
            return None

    def get_recent_sessions(self, subject_id: str = None, days: int = 30, limit: int = 50) -> List[Dict]:
        """Get recent sessions for dashboard display."""
        try:
            since_date = (datetime.now() - timedelta(days=days)).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                if subject_id:
                    cursor = conn.execute("""
                        SELECT * FROM sessions
                        WHERE subject_id = ? AND timestamp >= ?
                        ORDER BY timestamp DESC LIMIT ?
                    """, (subject_id, since_date, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM sessions
                        WHERE timestamp >= ?
                        ORDER BY timestamp DESC LIMIT ?
                    """, (since_date, limit))

                columns = [desc[0] for desc in cursor.description]
                sessions = [dict(zip(columns, row)) for row in cursor.fetchall()]

                return sessions

        except Exception as e:
            print(f"❌ Error getting recent sessions: {e}")
            return []

    def get_trend_data(self, subject_id: str, days: int = 30) -> Dict[str, List]:
        """Get trend data for charts."""
        sessions = self.get_recent_sessions(subject_id, days)

        trend_data = {
            'dates': [],
            'clinical_scores': [],
            'durations': [],
            'risk_levels': [],
            'flags_counts': []
        }

        for session in reversed(sessions):  # Chronological order
            # Parse timestamp for date
            date_str = session['timestamp'].split('T')[0]  # Get date part
            trend_data['dates'].append(date_str)
            trend_data['clinical_scores'].append(session['clinical_score'] or 0)
            trend_data['durations'].append(session['duration'])
            trend_data['risk_levels'].append(session['risk_level'])
            trend_data['flags_counts'].append(session['flags_count'])

        return trend_data

    def get_statistics(self, subject_id: str = None) -> Dict[str, Any]:
        """Get summary statistics for dashboard."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if subject_id:
                    # Subject-specific stats
                    cursor = conn.execute("""
                        SELECT
                            COUNT(*) as total_sessions,
                            AVG(clinical_score) as avg_score,
                            AVG(duration) as avg_duration,
                            SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) as high_risk_count,
                            SUM(flags_count) as total_flags
                        FROM sessions WHERE subject_id = ?
                    """, (subject_id,))
                else:
                    # Overall stats
                    cursor = conn.execute("""
                        SELECT
                            COUNT(*) as total_sessions,
                            AVG(clinical_score) as avg_score,
                            AVG(duration) as avg_duration,
                            SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) as high_risk_count,
                            SUM(flags_count) as total_flags
                        FROM sessions
                    """)

                result = cursor.fetchone()

                return {
                    'total_sessions': result[0] or 0,
                    'average_score': round(result[1] or 0, 1),
                    'average_duration': round(result[2] or 0, 1),
                    'high_risk_sessions': result[3] or 0,
                    'total_flags': result[4] or 0
                }

        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            return {
                'total_sessions': 0,
                'average_score': 0,
                'average_duration': 0,
                'high_risk_sessions': 0,
                'total_flags': 0
            }

    def compare_to_baseline(self, subject_id: str, recent_days: int = 7) -> Dict[str, Any]:
        """Compare recent performance to baseline (first few sessions)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get baseline (first 3-5 sessions)
                baseline_cursor = conn.execute("""
                    SELECT AVG(clinical_score) as baseline_score,
                           AVG(duration) as baseline_duration
                    FROM (
                        SELECT clinical_score, duration FROM sessions
                        WHERE subject_id = ?
                        ORDER BY timestamp ASC LIMIT 5
                    )
                """, (subject_id,))
                baseline = baseline_cursor.fetchone()

                # Get recent performance
                recent_date = (datetime.now() - timedelta(days=recent_days)).isoformat()
                recent_cursor = conn.execute("""
                    SELECT AVG(clinical_score) as recent_score,
                           AVG(duration) as recent_duration
                    FROM sessions
                    WHERE subject_id = ? AND timestamp >= ?
                """, (subject_id, recent_date))
                recent = recent_cursor.fetchone()

                if not baseline[0] or not recent[0]:
                    return {'status': 'insufficient_data'}

                score_change = ((recent[0] - baseline[0]) / baseline[0]) * 100
                duration_change = ((recent[1] - baseline[1]) / baseline[1]) * 100

                return {
                    'status': 'available',
                    'baseline_score': round(baseline[0], 1),
                    'recent_score': round(recent[0], 1),
                    'score_change_percent': round(score_change, 1),
                    'duration_change_percent': round(duration_change, 1),
                    'trend': 'improving' if score_change > 5 else 'declining' if score_change < -5 else 'stable'
                }

        except Exception as e:
            print(f"❌ Error comparing to baseline: {e}")
            return {'status': 'error'}

def create_session_from_walk_data(walk_results: Dict, clinical_assessment: Dict, subject_id: str = "user") -> GaitSession:
    """Create a GaitSession from analysis results."""
    session_id = f"walk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Determine risk level based on clinical assessment
    overall_score = clinical_assessment.get('overall_score', 100)
    if overall_score < 60:
        risk_level = 'high'
    elif overall_score < 80:
        risk_level = 'medium'
    else:
        risk_level = 'low'

    # Extract flags from risk factors
    flags = clinical_assessment.get('risk_factors', [])

    return GaitSession(
        session_id=session_id,
        timestamp=datetime.now().isoformat(),
        subject_id=subject_id,
        duration=walk_results.get('duration', 0),
        raw_keypoints=walk_results.get('keypoints_sequence', []),
        gait_metrics=walk_results,
        clinical_assessment=clinical_assessment,
        flags=flags,
        risk_level=risk_level
    )

def test_data_manager():
    """Test the data management system."""
    print("Testing GaitDataManager...")

    manager = GaitDataManager("test_data")

    # Create test session
    test_walk_data = {
        'duration': 8.5,
        'average_speed': 1.2,
        'stride_metrics': {
            'stride_time_left': 1.1,
            'stride_length_left': 1.4,
            'stride_time_variability': 0.03
        }
    }

    test_clinical = {
        'overall_score': 85,
        'risk_factors': ['Minor gait asymmetry'],
        'recommendations': ['Continue regular activity']
    }

    session = create_session_from_walk_data(test_walk_data, test_clinical)

    # Save session
    success = manager.save_session(session)
    print(f"Save session: {success}")

    # Retrieve session
    retrieved = manager.get_session(session.session_id)
    print(f"Retrieved session: {retrieved.session_id if retrieved else 'None'}")

    # Get statistics
    stats = manager.get_statistics("user")
    print(f"Statistics: {stats}")

if __name__ == "__main__":
    test_data_manager()