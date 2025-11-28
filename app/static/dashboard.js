// Gait Analysis Dashboard JavaScript

class GaitDashboard {
    constructor() {
        this.charts = {};
        this.currentSubjectId = 'default';
        this.isRecording = false;

        this.initializeEventListeners();
        this.initializeCharts();
        this.updateStatus();
        this.loadWalkHistory();

        // Update time and status periodically
        setInterval(() => this.updateTime(), 1000);
        setInterval(() => this.updateStatus(), 5000);
    }

    initializeEventListeners() {
        // Recording controls
        document.getElementById('start-recording').addEventListener('click', () => this.startRecording());
        document.getElementById('stop-recording').addEventListener('click', () => this.stopRecording());
        document.getElementById('analyze-recording').addEventListener('click', () => this.analyzeRecording());

        // History controls
        document.getElementById('refresh-history').addEventListener('click', () => this.loadWalkHistory());
        document.getElementById('reset-baseline').addEventListener('click', () => this.resetBaseline());

        // Subject ID change
        document.getElementById('subject-id').addEventListener('change', (e) => {
            this.currentSubjectId = e.target.value || 'default';
            this.loadWalkHistory();
        });
    }

    initializeCharts() {
        // Speed over time chart
        const speedCtx = document.getElementById('speed-chart').getContext('2d');
        this.charts.speed = new Chart(speedCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Gait Speed',
                    data: [],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: 'Time (seconds)' }
                    },
                    y: {
                        title: { display: true, text: 'Speed (units/s)' },
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });

        // Baseline comparison chart (z-scores)
        const baselineCtx = document.getElementById('baseline-chart').getContext('2d');
        this.charts.baseline = new Chart(baselineCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Z-Score',
                    data: [],
                    backgroundColor: [],
                    borderColor: '#2c3e50',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: 'Metrics' }
                    },
                    y: {
                        title: { display: true, text: 'Z-Score (Standard Deviations)' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    updateTime() {
        const now = new Date();
        document.getElementById('current-time').textContent = now.toLocaleTimeString();
    }

    async updateStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();

            const indicator = document.getElementById('status-indicator');
            const text = document.getElementById('status-text');

            if (status.recording) {
                indicator.className = 'status-indicator recording';
                text.textContent = 'Recording in progress...';
                this.isRecording = true;
                this.updateRecordingButtons();
            } else if (status.components_initialized) {
                indicator.className = 'status-indicator connected';
                text.textContent = 'System ready';
                this.isRecording = false;
                this.updateRecordingButtons();
            } else {
                indicator.className = 'status-indicator disconnected';
                text.textContent = 'System not ready';
            }

        } catch (error) {
            console.error('Failed to update status:', error);
            document.getElementById('status-indicator').className = 'status-indicator disconnected';
            document.getElementById('status-text').textContent = 'Connection error';
        }
    }

    updateRecordingButtons() {
        const startBtn = document.getElementById('start-recording');
        const stopBtn = document.getElementById('stop-recording');
        const analyzeBtn = document.getElementById('analyze-recording');

        if (this.isRecording) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            analyzeBtn.disabled = true;
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            analyzeBtn.disabled = false;
        }
    }

    showStatus(message, type = 'info') {
        const statusDiv = document.getElementById('recording-status');
        statusDiv.textContent = message;
        statusDiv.className = `recording-status ${type}`;
        statusDiv.style.display = 'block';

        // Auto-hide after 5 seconds
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 5000);
    }

    showLoading(message = 'Processing...') {
        document.getElementById('loading-text').textContent = message;
        document.getElementById('loading-modal').style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loading-modal').style.display = 'none';
    }

    async startRecording() {
        const duration = parseInt(document.getElementById('duration').value);

        if (duration < 5 || duration > 60) {
            this.showStatus('Duration must be between 5 and 60 seconds', 'error');
            return;
        }

        try {
            this.showLoading('Starting recording...');

            const response = await fetch('/api/record/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ duration })
            });

            const result = await response.json();

            if (response.ok) {
                this.showStatus(`Recording started for ${duration} seconds`, 'info');
                this.isRecording = true;
                this.updateRecordingButtons();

                // Auto-stop after duration
                setTimeout(() => {
                    this.stopRecording();
                }, duration * 1000 + 1000); // Add 1 second buffer

            } else {
                this.showStatus(`Error: ${result.error}`, 'error');
            }

        } catch (error) {
            console.error('Failed to start recording:', error);
            this.showStatus('Failed to start recording', 'error');
        } finally {
            this.hideLoading();
        }
    }

    async stopRecording() {
        try {
            this.showLoading('Stopping recording...');

            const response = await fetch('/api/record/stop', {
                method: 'POST'
            });

            const result = await response.json();

            if (response.ok) {
                this.showStatus(`Recording stopped. ${result.frames_recorded} frames recorded`, 'info');
                this.isRecording = false;
                this.updateRecordingButtons();
            } else {
                this.showStatus(`Error: ${result.error}`, 'error');
            }

        } catch (error) {
            console.error('Failed to stop recording:', error);
            this.showStatus('Failed to stop recording', 'error');
        } finally {
            this.hideLoading();
        }
    }

    async analyzeRecording() {
        try {
            this.showLoading('Analyzing gait pattern...');

            const response = await fetch('/api/record/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    subject_id: this.currentSubjectId
                })
            });

            const result = await response.json();

            if (response.ok) {
                this.displayAnalysisResults(result);
                this.showStatus('Analysis completed successfully', 'info');
                this.loadWalkHistory(); // Refresh history
            } else {
                this.showStatus(`Analysis error: ${result.error}`, 'error');
            }

        } catch (error) {
            console.error('Failed to analyze recording:', error);
            this.showStatus('Analysis failed', 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayAnalysisResults(result) {
        // Update metric cards
        document.getElementById('avg-speed').textContent = result.gait_metrics.average_speed.toFixed(4);
        document.getElementById('duration-value').textContent = result.gait_metrics.duration.toFixed(1);
        document.getElementById('stride-variability').textContent =
            (result.gait_metrics.stride_metrics.stride_length_variability || 0).toFixed(3);

        // Calculate speed variability
        const speeds = result.speeds_over_time;
        let speedVariability = 0;
        if (speeds.length > 1) {
            const mean = speeds.reduce((a, b) => a + b, 0) / speeds.length;
            const variance = speeds.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / speeds.length;
            speedVariability = mean > 0 ? Math.sqrt(variance) / mean : 0;
        }
        document.getElementById('speed-variability').textContent = speedVariability.toFixed(3);

        // Update speed chart
        const timeLabels = speeds.map((_, i) => (i / 30).toFixed(1)); // Assuming 30 FPS
        this.charts.speed.data.labels = timeLabels;
        this.charts.speed.data.datasets[0].data = speeds;
        this.charts.speed.update();

        // Update baseline comparison chart
        const baselineData = result.baseline_comparison;
        const metrics = Object.keys(baselineData);
        const zScores = Object.values(baselineData);

        // Color bars based on z-score magnitude
        const colors = zScores.map(z => {
            const abs_z = Math.abs(z);
            if (abs_z < 1) return '#27ae60'; // Green - normal
            if (abs_z < 2) return '#f39c12'; // Orange - mild deviation
            return '#e74c3c'; // Red - significant deviation
        });

        this.charts.baseline.data.labels = metrics.map(m => m.replace(/_/g, ' '));
        this.charts.baseline.data.datasets[0].data = zScores;
        this.charts.baseline.data.datasets[0].backgroundColor = colors;
        this.charts.baseline.update();
    }

    async loadWalkHistory() {
        try {
            const response = await fetch(`/api/walks?subject_id=${this.currentSubjectId}&limit=10`);
            const result = await response.json();

            const historyDiv = document.getElementById('walk-history');

            if (result.walks && result.walks.length > 0) {
                historyDiv.innerHTML = result.walks.map(walk => `
                    <div class="walk-item">
                        <h4>Walk ${walk.walk_id}</h4>
                        <p>Date: ${new Date(walk.timestamp).toLocaleString()}</p>
                        <p>Duration: ${walk.duration.toFixed(1)}s</p>
                        <p>Average Speed: ${walk.metrics.average_speed.toFixed(4)} units/s</p>
                        <p>Stride Variability: ${(walk.metrics.stride_length_variability || 0).toFixed(3)}</p>
                    </div>
                `).join('');
            } else {
                historyDiv.innerHTML = '<p style="text-align: center; color: #666;">No recorded walks yet</p>';
            }

        } catch (error) {
            console.error('Failed to load walk history:', error);
            document.getElementById('walk-history').innerHTML =
                '<p style="text-align: center; color: #e74c3c;">Failed to load walk history</p>';
        }
    }

    async resetBaseline() {
        if (!confirm('Are you sure you want to reset the baseline? This will remove all historical data.')) {
            return;
        }

        try {
            this.showLoading('Resetting baseline...');

            const response = await fetch('/api/baseline/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    subject_id: this.currentSubjectId
                })
            });

            const result = await response.json();

            if (response.ok) {
                this.showStatus('Baseline reset successfully', 'info');
                this.loadWalkHistory();

                // Clear charts
                this.charts.speed.data.labels = [];
                this.charts.speed.data.datasets[0].data = [];
                this.charts.speed.update();

                this.charts.baseline.data.labels = [];
                this.charts.baseline.data.datasets[0].data = [];
                this.charts.baseline.update();

                // Clear metric cards
                document.getElementById('avg-speed').textContent = '-';
                document.getElementById('duration-value').textContent = '-';
                document.getElementById('stride-variability').textContent = '-';
                document.getElementById('speed-variability').textContent = '-';

            } else {
                this.showStatus(`Error: ${result.error}`, 'error');
            }

        } catch (error) {
            console.error('Failed to reset baseline:', error);
            this.showStatus('Failed to reset baseline', 'error');
        } finally {
            this.hideLoading();
        }
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new GaitDashboard();
});