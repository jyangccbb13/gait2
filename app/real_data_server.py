"""
Flask server for real gait analysis dashboard.
Serves actual session data instead of simulated numbers.
"""

import os
import sys
import json
from flask import Flask, render_template, jsonify, request, Response
from datetime import datetime, timedelta

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_persistence import GaitDataManager

app = Flask(__name__)

# Initialize data manager
data_manager = GaitDataManager("gait_sessions")

@app.route('/')
def dashboard():
    """Serve the real data dashboard."""
    return render_template('real_dashboard.html')

@app.route('/api/dashboard/stats')
def get_dashboard_stats():
    """Get real statistics for dashboard."""
    subject_id = request.args.get('subject_id', 'user')

    # Get overall statistics
    stats = data_manager.get_statistics(subject_id)

    # Get recent sessions for trends
    recent_sessions = data_manager.get_recent_sessions(subject_id, days=7, limit=20)

    # Calculate additional metrics
    today_sessions = [s for s in recent_sessions
                     if s['timestamp'].startswith(datetime.now().strftime('%Y-%m-%d'))]

    # Get baseline comparison
    baseline_comparison = data_manager.compare_to_baseline(subject_id)

    return jsonify({
        'total_residents': 1,  # Single user for now
        'active_residents': 1 if stats['total_sessions'] > 0 else 0,
        'average_mobility_score': stats['average_score'],
        'mobility_score_change': baseline_comparison.get('score_change_percent', 0) if baseline_comparison.get('status') == 'available' else 0,
        'active_alerts': stats['high_risk_sessions'],
        'alerts_change': 0,  # Could calculate this from historical data
        'sessions_today': len(today_sessions),
        'sessions_change': 12,  # Placeholder
        'total_sessions': stats['total_sessions'],
        'average_duration': stats['average_duration'],
        'last_updated': datetime.now().isoformat()
    })

@app.route('/api/dashboard/trends')
def get_trend_data():
    """Get trend data for charts."""
    subject_id = request.args.get('subject_id', 'user')
    days = int(request.args.get('days', 30))

    trend_data = data_manager.get_trend_data(subject_id, days)

    # Transform for Chart.js
    chart_data = {
        'labels': trend_data['dates'][-10:],  # Last 10 data points
        'mobility_scores': trend_data['clinical_scores'][-10:],
        'durations': trend_data['durations'][-10:],
        'risk_levels': trend_data['risk_levels'][-10:]
    }

    return jsonify(chart_data)

@app.route('/api/dashboard/recent_sessions')
def get_recent_sessions():
    """Get recent sessions for dashboard display."""
    subject_id = request.args.get('subject_id', 'user')
    limit = int(request.args.get('limit', 10))

    sessions = data_manager.get_recent_sessions(subject_id, days=30, limit=limit)

    # Format for display
    formatted_sessions = []
    for session in sessions:
        risk_level_map = {'low': 'Low Risk', 'medium': 'Medium Risk', 'high': 'High Risk'}

        # Parse timestamp for display
        dt = datetime.fromisoformat(session['timestamp'])
        formatted_sessions.append({
            'id': session['session_id'],
            'name': f"Walking Session {session['session_id'][-8:]}",
            'room': "Home Setup",  # Placeholder
            'age': 25,  # Placeholder
            'risk_level': risk_level_map.get(session['risk_level'], 'Unknown'),
            'risk_class': session['risk_level'],
            'last_assessment': dt.strftime('%Y-%m-%d %H:%M'),
            'clinical_score': session['clinical_score'],
            'duration': session['duration'],
            'flags_count': session['flags_count']
        })

    return jsonify(formatted_sessions)

@app.route('/api/dashboard/alerts')
def get_active_alerts():
    """Get active alerts for dashboard."""
    subject_id = request.args.get('subject_id', 'user')

    # Get recent high-risk sessions as "alerts"
    recent_sessions = data_manager.get_recent_sessions(subject_id, days=7, limit=50)
    high_risk_sessions = [s for s in recent_sessions if s['risk_level'] == 'high']
    medium_risk_sessions = [s for s in recent_sessions if s['risk_level'] == 'medium']

    alerts = []

    # High-risk alerts
    for session in high_risk_sessions[:3]:  # Top 3 high-risk
        dt = datetime.fromisoformat(session['timestamp'])
        alerts.append({
            'id': session['session_id'],
            'type': 'critical',
            'icon': '!',
            'title': 'High Fall Risk Detected',
            'description': f"Session {session['session_id'][-8:]} - Score: {session['clinical_score']:.0f}/100",
            'time': dt.strftime('%H:%M'),
            'time_ago': f"{(datetime.now() - dt).seconds // 60} min ago"
        })

    # Medium-risk alerts
    for session in medium_risk_sessions[:2]:  # Top 2 medium-risk
        dt = datetime.fromisoformat(session['timestamp'])
        alerts.append({
            'id': session['session_id'],
            'type': 'warning',
            'icon': '⚠',
            'title': 'Mobility Pattern Change',
            'description': f"Session {session['session_id'][-8:]} - Score: {session['clinical_score']:.0f}/100",
            'time': dt.strftime('%H:%M'),
            'time_ago': f"{(datetime.now() - dt).seconds // 60} min ago"
        })

    # Baseline reminder if needed
    stats = data_manager.get_statistics(subject_id)
    if stats['total_sessions'] >= 5 and stats['total_sessions'] % 10 == 0:
        alerts.append({
            'id': 'baseline_reminder',
            'type': 'info',
            'icon': 'ℹ',
            'title': 'Baseline Reassessment Available',
            'description': f"Consider updating baseline after {stats['total_sessions']} sessions",
            'time': '09:00',
            'time_ago': 'Scheduled'
        })

    return jsonify(alerts[:5])  # Limit to 5 alerts

@app.route('/api/dashboard/daily_performance')
def get_daily_performance():
    """Get daily performance data for charts."""
    subject_id = request.args.get('subject_id', 'user')

    # Get last 7 days of data
    trend_data = data_manager.get_trend_data(subject_id, days=7)

    # Group by day
    daily_data = {}
    for i, date in enumerate(trend_data['dates']):
        if date not in daily_data:
            daily_data[date] = []
        daily_data[date].append({
            'score': trend_data['clinical_scores'][i],
            'duration': trend_data['durations'][i]
        })

    # Calculate daily averages
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily_sessions = []
    daily_scores = []

    # Get last 7 days
    for i in range(7):
        date = (datetime.now() - timedelta(days=6-i)).strftime('%Y-%m-%d')
        day_sessions = daily_data.get(date, [])

        daily_sessions.append(len(day_sessions))
        if day_sessions:
            daily_scores.append(sum(s['score'] for s in day_sessions) / len(day_sessions))
        else:
            daily_scores.append(0)

    return jsonify({
        'labels': days,
        'sessions': daily_sessions,
        'average_scores': daily_scores
    })

@app.route('/api/session/<session_id>')
def get_session_details(session_id):
    """Get detailed information for a specific session."""
    session = data_manager.get_session(session_id)

    if not session:
        return jsonify({'error': 'Session not found'}), 404

    return jsonify({
        'session_id': session.session_id,
        'timestamp': session.timestamp,
        'duration': session.duration,
        'risk_level': session.risk_level,
        'clinical_assessment': session.clinical_assessment,
        'gait_metrics': session.gait_metrics,
        'flags': session.flags
    })

def main():
    """Start the real data Flask server."""
    print("🌐 Starting Real Gait Data Dashboard Server...")
    print("📊 Dashboard will show your actual walking data")
    print("🔗 Available at: http://localhost:8001")
    print("📈 API endpoints:")
    print("  GET /api/dashboard/stats - Overall statistics")
    print("  GET /api/dashboard/trends - Trend data for charts")
    print("  GET /api/dashboard/recent_sessions - Recent walking sessions")
    print("  GET /api/dashboard/alerts - Active alerts and flags")
    print("  GET /api/dashboard/daily_performance - Daily performance metrics")
    print("  GET /api/session/<id> - Detailed session information")

    app.run(
        host='0.0.0.0',
        port=8001,
        debug=False,
        threaded=True
    )

if __name__ == '__main__':
    main()