"""
Decision Visualizer Module for UCUP Framework.

Provides visualization capabilities for decision processes, probabilistic reasoning, and system behavior.
"""

import time
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import base64
import io
from ..core.manager import UCUPManager


@dataclass
class VisualizationData:
    """Container for visualization data."""
    title: str
    data_type: str  # "probabilistic", "coordination", "multimodal", "performance"
    data: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


class DecisionVisualizer:
    """Visualization system for UCUP framework decision processes."""

    def __init__(self, manager: UCUPManager):
        self.manager = manager
        self.config = manager.config
        self._visualizations = []
        self._output_dir = Path("visualizations")
        self._output_dir.mkdir(exist_ok=True)

    def visualize_probabilistic_decision(self, decision_data: Dict[str, Any],
                                       title: str = "Probabilistic Decision") -> str:
        """
        Visualize a probabilistic decision process.

        Args:
            decision_data: Decision data with probabilities, confidence intervals, etc.
            title: Visualization title

        Returns:
            Path to generated visualization file
        """
        viz_data = VisualizationData(
            title=title,
            data_type="probabilistic",
            data=decision_data,
            metadata={"visualization_type": "decision_tree"}
        )

        return self._generate_probabilistic_visualization(viz_data)

    def visualize_coordination_flow(self, coordination_data: List[Dict[str, Any]],
                                  title: str = "Coordination Flow") -> str:
        """
        Visualize agent coordination flow.

        Args:
            coordination_data: List of coordination steps/events
            title: Visualization title

        Returns:
            Path to generated visualization file
        """
        viz_data = VisualizationData(
            title=title,
            data_type="coordination",
            data={"steps": coordination_data},
            metadata={"visualization_type": "flow_diagram"}
        )

        return self._generate_coordination_visualization(viz_data)

    def visualize_multimodal_fusion(self, fusion_data: Dict[str, Any],
                                  title: str = "Multimodal Fusion") -> str:
        """
        Visualize multimodal data fusion process.

        Args:
            fusion_data: Fusion results with modality contributions
            title: Visualization title

        Returns:
            Path to generated visualization file
        """
        viz_data = VisualizationData(
            title=title,
            data_type="multimodal",
            data=fusion_data,
            metadata={"visualization_type": "fusion_heatmap"}
        )

        return self._generate_multimodal_visualization(viz_data)

    def visualize_performance_metrics(self, metrics_data: Dict[str, Any],
                                    title: str = "Performance Metrics") -> str:
        """
        Visualize performance metrics over time.

        Args:
            metrics_data: Performance metrics data
            title: Visualization title

        Returns:
            Path to generated visualization file
        """
        viz_data = VisualizationData(
            title=title,
            data_type="performance",
            data=metrics_data,
            metadata={"visualization_type": "time_series"}
        )

        return self._generate_performance_visualization(viz_data)

    def visualize_decision_comparison(self, decisions: List[Dict[str, Any]],
                                    title: str = "Decision Comparison") -> str:
        """
        Visualize comparison of multiple decision outcomes.

        Args:
            decisions: List of decision results to compare
            title: Visualization title

        Returns:
            Path to generated visualization file
        """
        viz_data = VisualizationData(
            title=title,
            data_type="comparison",
            data={"decisions": decisions},
            metadata={"visualization_type": "radar_chart"}
        )

        return self._generate_comparison_visualization(viz_data)

    def export_visualization_data(self, filepath: str, format: str = "json"):
        """Export all visualization data."""
        data = {
            "visualizations": [self._serialize_viz(viz) for viz in self._visualizations],
            "export_timestamp": time.time(),
            "framework_version": getattr(self.manager, 'version', 'unknown')
        }

        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "html":
            self._export_as_html(filepath, data)

    def get_visualization_history(self) -> List[Dict[str, Any]]:
        """Get history of all visualizations."""
        return [self._serialize_viz(viz) for viz in self._visualizations]

    def clear_visualizations(self):
        """Clear stored visualizations."""
        self._visualizations.clear()

    def _generate_probabilistic_visualization(self, viz_data: VisualizationData) -> str:
        """Generate probabilistic decision visualization."""
        # Create HTML visualization
        html_content = self._create_probabilistic_html(viz_data)

        filename = f"probabilistic_decision_{int(time.time())}.html"
        filepath = self._output_dir / filename

        with open(filepath, 'w') as f:
            f.write(html_content)

        self._visualizations.append(viz_data)
        return str(filepath)

    def _generate_coordination_visualization(self, viz_data: VisualizationData) -> str:
        """Generate coordination flow visualization."""
        html_content = self._create_coordination_html(viz_data)

        filename = f"coordination_flow_{int(time.time())}.html"
        filepath = self._output_dir / filename

        with open(filepath, 'w') as f:
            f.write(html_content)

        self._visualizations.append(viz_data)
        return str(filepath)

    def _generate_multimodal_visualization(self, viz_data: VisualizationData) -> str:
        """Generate multimodal fusion visualization."""
        html_content = self._create_multimodal_html(viz_data)

        filename = f"multimodal_fusion_{int(time.time())}.html"
        filepath = self._output_dir / filename

        with open(filepath, 'w') as f:
            f.write(html_content)

        self._visualizations.append(viz_data)
        return str(filepath)

    def _generate_performance_visualization(self, viz_data: VisualizationData) -> str:
        """Generate performance metrics visualization."""
        html_content = self._create_performance_html(viz_data)

        filename = f"performance_metrics_{int(time.time())}.html"
        filepath = self._output_dir / filename

        with open(filepath, 'w') as f:
            f.write(html_content)

        self._visualizations.append(viz_data)
        return str(filepath)

    def _generate_comparison_visualization(self, viz_data: VisualizationData) -> str:
        """Generate decision comparison visualization."""
        html_content = self._create_comparison_html(viz_data)

        filename = f"decision_comparison_{int(time.time())}.html"
        filepath = self._output_dir / filename

        with open(filepath, 'w') as f:
            f.write(html_content)

        self._visualizations.append(viz_data)
        return str(filepath)

    def _create_probabilistic_html(self, viz_data: VisualizationData) -> str:
        """Create HTML for probabilistic decision visualization."""
        data = viz_data.data

        # Extract decision data
        probability = data.get('probability', 0)
        confidence_interval = data.get('confidence_interval', [0, 1])
        factors = data.get('contributing_factors', [])

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{viz_data.title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{viz_data.title}</h1>
        <div class="metric">
            <strong>Decision Probability:</strong> {probability:.3f}<br>
            <strong>Confidence Interval:</strong> [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]
        </div>

        <div id="probabilityGauge" style="width:100%;height:400px;"></div>

        <h2>Contributing Factors</h2>
        <ul>
"""

        for factor in factors:
            html += f"<li>{factor.get('name', 'Unknown')}: {factor.get('weight', 0):.3f}</li>"

        html += """
        </ul>
    </div>

    <script>
        var data = [{
            type: "indicator",
            mode: "gauge+number",
            value: """ + str(probability) + """,
            title: { text: "Decision Confidence" },
            gauge: {
                axis: { range: [0, 1] },
                bar: { color: "#1f77b4" },
                steps: [
                    { range: [0, 0.5], color: "#ff7f7f" },
                    { range: [0.5, 0.8], color: "#ffff7f" },
                    { range: [0.8, 1], color: "#7fff7f" }
                ]
            }
        }];

        var layout = { width: 600, height: 400 };
        Plotly.newPlot('probabilityGauge', data, layout);
    </script>
</body>
</html>
"""

        return html

    def _create_coordination_html(self, viz_data: VisualizationData) -> str:
        """Create HTML for coordination flow visualization."""
        steps = viz_data.data.get('steps', [])

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{viz_data.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .step {{ background: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #1f77b4; }}
        .timeline {{ position: relative; padding-left: 30px; }}
        .timeline::before {{ content: ''; position: absolute; left: 15px; top: 0; bottom: 0; width: 2px; background: #1f77b4; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{viz_data.title}</h1>
        <div class="timeline">
"""

        for i, step in enumerate(steps):
            agent = step.get('agent', 'Unknown')
            action = step.get('action', 'Unknown')
            timestamp = step.get('timestamp', 0)
            status = step.get('status', 'completed')

            html += f"""
            <div class="step">
                <strong>Step {i+1}:</strong> Agent {agent} - {action}<br>
                <small>Time: {timestamp:.2f}s | Status: {status}</small>
            </div>
"""

        html += """
        </div>
    </div>
</body>
</html>
"""

        return html

    def _create_multimodal_html(self, viz_data: VisualizationData) -> str:
        """Create HTML for multimodal fusion visualization."""
        data = viz_data.data
        modalities = data.get('modalities', {})
        fused_result = data.get('fused_analysis', {})

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{viz_data.title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .modality {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{viz_data.title}</h1>

        <h2>Modality Contributions</h2>
"""

        for modality_name, modality_data in modalities.items():
            confidence = modality_data.get('confidence', 0)
            html += f"""
            <div class="modality">
                <strong>{modality_name.upper()}</strong><br>
                Confidence: {confidence:.3f}
            </div>
"""

        html += """
        <h2>Fusion Result</h2>
        <div id="fusionChart" style="width:100%;height:400px;"></div>
    </div>

    <script>
        var fusionData = """ + json.dumps(fused_result) + """;

        // Simple bar chart for fusion results
        var data = [{
            type: 'bar',
            x: Object.keys(fusionData),
            y: Object.values(fusionData),
            marker: { color: '#1f77b4' }
        }];

        var layout = {
            title: 'Fusion Analysis Results',
            xaxis: { title: 'Metrics' },
            yaxis: { title: 'Values' }
        };

        Plotly.newPlot('fusionChart', data, layout);
    </script>
</body>
</html>
"""

        return html

    def _create_performance_html(self, viz_data: VisualizationData) -> str:
        """Create HTML for performance metrics visualization."""
        data = viz_data.data
        metrics = data.get('metrics', {})

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{viz_data.title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{viz_data.title}</h1>

        <div id="performanceChart" style="width:100%;height:400px;"></div>

        <h2>Key Metrics</h2>
"""

        for metric_name, metric_data in metrics.items():
            mean_val = metric_data.get('mean', 0)
            p95 = metric_data.get('p95', 0)
            html += f"""
            <div class="metric">
                <strong>{metric_name}</strong><br>
                Mean: {mean_val:.3f} | P95: {p95:.3f}
            </div>
"""

        html += """
    </div>

    <script>
        var metricsData = """ + json.dumps(metrics) + """;

        // Create traces for each metric
        var traces = [];
        var colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'];

        var traceIndex = 0;
        for (var metricName in metricsData) {
            if (metricsData.hasOwnProperty(metricName)) {
                var metric = metricsData[metricName];
                traces.push({
                    x: [metric.p50, metric.p95, metric.p99],
                    y: [metricName, metricName, metricName],
                    name: metricName,
                    type: 'scatter',
                    mode: 'markers',
                    marker: { color: colors[traceIndex % colors.length], size: 10 }
                });
                traceIndex++;
            }
        }

        var layout = {
            title: 'Performance Percentiles',
            xaxis: { title: 'Value' },
            yaxis: { title: 'Metric' }
        };

        Plotly.newPlot('performanceChart', traces, layout);
    </script>
</body>
</html>
"""

        return html

    def _create_comparison_html(self, viz_data: VisualizationData) -> str:
        """Create HTML for decision comparison visualization."""
        decisions = viz_data.data.get('decisions', [])

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{viz_data.title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .decision {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{viz_data.title}</h1>

        <div id="comparisonChart" style="width:100%;height:400px;"></div>

        <h2>Decision Details</h2>
"""

        for i, decision in enumerate(decisions):
            probability = decision.get('probability', 0)
            confidence = decision.get('confidence', 0)
            html += f"""
            <div class="decision">
                <strong>Decision {i+1}</strong><br>
                Probability: {probability:.3f} | Confidence: {confidence:.3f}
            </div>
"""

        html += """
    </div>

    <script>
        var decisions = """ + json.dumps(decisions) + """;

        // Radar chart for decision comparison
        var data = [];
        var colors = ['#1f77b4', '#ff7f0e', '#2ca02c'];

        decisions.forEach(function(decision, index) {
            data.push({
                type: 'scatterpolar',
                r: [decision.probability || 0, decision.confidence || 0, (decision.probability || 0) * (decision.confidence || 0)],
                theta: ['Probability', 'Confidence', 'Combined Score'],
                fill: 'toself',
                name: 'Decision ' + (index + 1),
                line: { color: colors[index % colors.length] }
            });
        });

        var layout = {
            polar: {
                radialaxis: { visible: true, range: [0, 1] }
            },
            showlegend: true,
            title: 'Decision Comparison'
        };

        Plotly.newPlot('comparisonChart', data, layout);
    </script>
</body>
</html>
"""

        return html

    def _serialize_viz(self, viz: VisualizationData) -> Dict[str, Any]:
        """Serialize visualization data for export."""
        return {
            "title": viz.title,
            "data_type": viz.data_type,
            "timestamp": viz.timestamp,
            "metadata": viz.metadata,
            "data": viz.data
        }

    def _export_as_html(self, filepath: str, data: Dict[str, Any]):
        """Export visualization data as HTML dashboard."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>UCUP Visualization Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .viz-item {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>UCUP Visualization Dashboard</h1>
        <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="viz-grid">
"""

        for viz in data.get("visualizations", []):
            html_content += f"""
            <div class="viz-item">
                <h3>{viz['title']}</h3>
                <p><strong>Type:</strong> {viz['data_type']}</p>
                <p><strong>Created:</strong> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(viz['timestamp']))}</p>
            </div>
"""

        html_content += """
        </div>
    </div>
</body>
</html>
"""

        with open(filepath, 'w') as f:
            f.write(html_content)

    def __repr__(self) -> str:
        return f"DecisionVisualizer(visualizations={len(self._visualizations)})"
