"""
UCUP Visualization Module

Provides integrated visualization capabilities with Plotly and market analysis tools.
Includes Plotly visualizations and real-time market data integration.
"""

import json
import os
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional imports
try:
    from datetime import datetime, timedelta

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import requests
    from plotly.subplots import make_subplots

    VISUALIZATION_DEPS_AVAILABLE = True
except ImportError:
    VISUALIZATION_DEPS_AVAILABLE = False
    warnings.warn(
        "Visualization dependencies not available. Install with: pip install plotly pandas numpy requests"
    )


@dataclass
class ChartConfig:
    """Configuration for chart generation."""

    title: str = "UCUP Visualization"
    height: int = 600
    width: Optional[int] = None
    show_legend: bool = True
    theme: str = "plotly_white"
    colors: List[str] = None

    def __post_init__(self):
        if self.colors is None:
            self.colors = px.colors.qualitative.Set1


@dataclass
class MarketDataPoint:
    """Represents a single market data point."""

    symbol: str
    timestamp: datetime
    price: float
    volume: Optional[int] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    close: Optional[float] = None


class PlotlyVisualizer:
    """Main visualization class using Plotly for UCUP data."""

    def __init__(self):
        if not VISUALIZATION_DEPS_AVAILABLE:
            raise ImportError("Visualization dependencies not available")

        self.config = ChartConfig()
        self._ensure_plotly_setup()

    def _ensure_plotly_setup(self):
        """Ensure Plotly is properly configured."""
        import plotly.io as pio

        pio.templates.default = self.config.theme

    def plot_uncertainty_analysis(
        self, results: Dict[str, Any], config: ChartConfig = None
    ) -> go.Figure:
        """
        Create comprehensive uncertainty analysis visualization.

        Args:
            results: Uncertainty analysis results
            config: Optional chart configuration

        Returns:
            Plotly Figure object
        """
        if config is None:
            config = self.config

        # Create subplot figure
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Confidence Distribution",
                "Uncertainty vs Confidence",
                "Volatility Trends",
                "Alternative Predictions Summary",
            ),
            specs=[[{}, {}], [{}, {}]],
        )

        # Extract data from results
        uncertainties = [
            r.get("uncertainty_score", 0) for r in results.get("results", [])
        ]
        confidences = [r.get("confidence", 0) for r in results.get("results", [])]
        volatilities = [
            r.get("volatility_metrics", {}).get("current_volatility", 0)
            for r in results.get("results", [])
        ]

        # 1. Confidence Distribution Histogram
        if confidences:
            fig.add_trace(
                go.Histogram(
                    x=confidences,
                    nbinsx=20,
                    name="Confidence Scores",
                    marker_color=config.colors[0],
                ),
                row=1,
                col=1,
            )

        # 2. Uncertainty vs Confidence Scatter
        if uncertainties and confidences:
            fig.add_trace(
                go.Scatter(
                    x=confidences,
                    y=uncertainties,
                    mode="markers",
                    name="Uncertainty Points",
                    marker_color=config.colors[1],
                ),
                row=1,
                col=2,
            )

        # 3. Volatility Trend
        if volatilities:
            volatility_trend = (
                pd.Series(volatilities).rolling(window=5, min_periods=1).mean()
            )
            fig.add_trace(
                go.Scatter(
                    y=volatility_trend.values,
                    mode="lines+markers",
                    name="Volatility Trend",
                    line_color=config.colors[2],
                ),
                row=2,
                col=1,
            )

        # 4. Alternative Predictions Summary
        alternative_counts = []
        for result in results.get("results", []):
            alt_count = len(result.get("alternative_predictions", []))
            alternative_counts.append(alt_count)

        if alternative_counts:
            fig.add_trace(
                go.Bar(
                    x=list(range(len(alternative_counts))),
                    y=alternative_counts,
                    name="Alternative Predictions",
                    marker_color=config.colors[3],
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title=config.title,
            height=config.height,
            showlegend=config.show_legend,
            template=config.theme,
        )

        # Add summary annotations
        summary = results.get("summary", {})
        annotation_text = f"Avg Confidence: {summary.get('avg_confidence', 0):.2f}<br>"
        annotation_text += (
            f"Avg Uncertainty: {summary.get('avg_uncertainty_score', 0):.2f}<br>"
        )
        annotation_text += (
            f"Volatility Range: {summary.get('volatility_range', {}).get('avg', 0):.3f}"
        )

        fig.add_annotation(
            text=annotation_text,
            xref="paper",
            yref="paper",
            x=1.1,
            y=1.0,
            showarrow=False,
            align="left",
        )

        return fig

    def plot_confidence_ranges(
        self, data: Dict[str, Any], config: ChartConfig = None
    ) -> go.Figure:
        """
        Visualize confidence ranges and prediction intervals.

        Args:
            data: Confidence range data
            config: Optional chart configuration

        Returns:
            Plotly Figure object
        """
        if config is None:
            config = self.config

        fig = go.Figure()

        # Extract confidence intervals from data
        intervals_95 = []
        for result in data.get("results", []):
            ranges = result.get("confidence_ranges", {})
            ci_95 = ranges.get("confidence_interval_95", [0, 0])
            intervals_95.append(ci_95)

        if intervals_95:
            # Create box plot for confidence intervals
            lower_bounds = [interval[0] for interval in intervals_95]
            upper_bounds = [interval[1] for interval in intervals_95]

            fig.add_trace(
                go.Box(
                    y=lower_bounds + upper_bounds,
                    name="95% Confidence Intervals",
                    marker_color=config.colors[0],
                )
            )

        fig.update_layout(
            title="Confidence Range Distribution",
            yaxis_title="Confidence Score",
            height=config.height,
        )

        return fig

    def plot_performance_trends(
        self, metrics: List[Dict[str, Any]], config: ChartConfig = None
    ) -> go.Figure:
        """
        Plot performance trends over time.

        Args:
            metrics: List of performance metrics
            config: Optional chart configuration

        Returns:
            Plotly Figure object
        """
        if config is None:
            config = self.config

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(metrics)

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "Confidence Trends",
                "Uncertainty Trends",
                "Performance Metrics",
            ),
            shared_xaxes=True,
        )

        if "avg_confidence" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["avg_confidence"],
                    name="Avg Confidence",
                    line=dict(color=config.colors[0]),
                ),
                row=1,
                col=1,
            )

        if "uncertainty_score" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["uncertainty_score"],
                    name="Uncertainty Score",
                    line=dict(color=config.colors[1]),
                ),
                row=2,
                col=1,
            )

        if "total_failures" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["total_failures"],
                    name="Total Failures",
                    line=dict(color=config.colors[2]),
                    yaxis="y2",
                ),
                row=3,
                col=1,
            )

        fig.update_layout(height=config.height, title_text=config.title)
        fig.update_xaxes(title_text="Time Period", row=3, col=1)
        fig.update_yaxes(title_text="Confidence/Failures", row=1, col=1)

        return fig

    def visualize_agent_matrix(
        self, agent_data: Dict[str, Any], config: ChartConfig = None
    ) -> go.Figure:
        """
        Create agent performance matrix visualization.

        Args:
            agent_data: Agent performance data
            config: Optional chart configuration

        Returns:
            Plotly Figure object
        """
        if config is None:
            config = self.config

        # Create performance heatmap
        agents = list(agent_data.keys())
        metrics = ["confidence", "uncertainty_score", "robustness_score", "volatility"]

        heatmap_data = []
        for agent in agents:
            agent_metrics = agent_data[agent]
            row = [
                agent_metrics.get("confidence", 0),
                agent_metrics.get("uncertainty_score", 0),
                agent_metrics.get("robustness_score", 0),
                agent_metrics.get("volatility", 0),
            ]
            heatmap_data.append(row)

        fig = go.Figure(
            data=go.Heatmap(z=heatmap_data, x=metrics, y=agents, colorscale="RdYlBu_r")
        )

        fig.update_layout(
            title="Agent Performance Matrix",
            xaxis_title="Metrics",
            yaxis_title="Agents",
            height=config.height,
        )

        return fig

    def export_chart(
        self, fig: go.Figure, format: str = "html", filename: str = "chart"
    ) -> str:
        """
        Export chart to various formats.

        Args:
            fig: Plotly Figure object
            format: Export format ('html', 'png', 'svg', 'json')
            filename: Output filename (without extension)

        Returns:
            Path to exported file
        """
        output_path = f"{filename}.{format}"

        if format == "html":
            fig.write_html(output_path)
        elif format in ["png", "svg"]:
            fig.write_image(output_path, format=format)
        elif format == "json":
            fig.write_json(output_path)

        return output_path


class MarketAnalyzer:
    """Market data analysis and visualization using UCUP uncertainties."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize market analyzer.

        Args:
            api_key: Optional API key for market data providers
        """
        if not VISUALIZATION_DEPS_AVAILABLE:
            raise ImportError("Visualization dependencies not available")

        self.api_key = api_key or os.getenv("MARKET_API_KEY")
        self.visualizer = PlotlyVisualizer()

    def fetch_market_data(
        self, symbol: str, days: int = 30, source: str = "free"
    ) -> List[MarketDataPoint]:
        """
        Fetch market data for analysis.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            days: Number of days of historical data
            source: Data source ('free', 'alpha_vantage', 'yfinance')

        Returns:
            List of market data points
        """
        # Mock data for demonstration
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        data_points = []
        current_date = start_date
        base_price = 100.0

        while current_date <= end_date:
            # Simulate realistic price movements
            daily_return = np.random.normal(0.001, 0.03)  # Mean and std of returns
            base_price *= 1 + daily_return

            volume = np.random.randint(1000000, 10000000)

            point = MarketDataPoint(
                symbol=symbol,
                timestamp=current_date,
                price=round(base_price, 2),
                volume=volume,
                high=round(base_price * (1 + abs(np.random.normal(0, 0.02))), 2),
                low=round(base_price * (1 - abs(np.random.normal(0, 0.02))), 2),
                open=round(base_price * (1 + np.random.normal(0, 0.01)), 2),
                close=round(base_price, 2),
            )
            data_points.append(point)
            current_date += timedelta(days=1)

        return data_points

    def analyze_market_uncertainty(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Analyze market data with UCUP uncertainty quantification.

        Args:
            symbol: Stock symbol to analyze
            days: Days of historical data to analyze

        Returns:
            Analysis results with uncertainty quantification
        """
        # Fetch market data
        market_data = self.fetch_market_data(symbol, days)

        # Convert to price series
        prices = [point.price for point in market_data]
        timestamps = [point.timestamp for point in market_data]

        # Calculate technical indicators (simplified)
        returns = []
        for i in range(1, len(prices)):
            daily_return = (prices[i] - prices[i - 1]) / prices[i - 1]
            returns.append(daily_return)

        # Calculate volatility (standard deviation of returns)
        if returns:
            volatility = float(np.std(returns))
            avg_return = float(np.mean(returns))
            price_trend = "bullish" if avg_return > 0 else "bearish"
        else:
            volatility = 0.0
            avg_return = 0.0
            price_trend = "neutral"

        # Simulate confidence in analysis
        confidence = 1.0 - min(
            volatility * 10, 0.9
        )  # Higher volatility = lower confidence

        analysis_result = {
            "symbol": symbol,
            "analysis_period": days,
            "current_price": prices[-1] if prices else 0,
            "price_change_pct": ((prices[-1] - prices[0]) / prices[0] * 100)
            if prices and len(prices) > 1
            else 0,
            "volatility": volatility,
            "trend": price_trend,
            "confidence_score": confidence,
            "data_points": len(market_data),
        }

        return analysis_result

    def create_market_visualization(
        self, analysis_results: Dict[str, Any], config: ChartConfig = None
    ) -> go.Figure:
        """
        Create comprehensive market analysis visualization.

        Args:
            analysis_results: Results from market analysis
            config: Optional chart configuration

        Returns:
            Plotly Figure object
        """
        if config is None:
            config = self.visualizer.config

        # Create subplot figure
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Price Trend Analysis",
                "Volatility Analysis",
                "Confidence Assessment",
                "Market Prediction",
            ),
            specs=[[{}, {}], [{}, {}]],
        )

        # Mock additional data for completeness
        periods = list(range(30))

        # 1. Price Trend
        mock_prices = [100 + np.sin(i / 3) * 20 + (i * 0.5) for i in periods]
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=mock_prices,
                name="Price Trend",
                line=dict(color=config.colors[0]),
            ),
            row=1,
            col=1,
        )

        # 2. Volatility Analysis
        mock_volatility = [
            np.random.normal(0.02, 0.005) + np.exp(-i / 10) * 0.01 for i in periods
        ]
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=mock_volatility,
                name="Volatility",
                fill="tozeroy",
                line=dict(color=config.colors[1]),
            ),
            row=1,
            col=2,
        )

        # 3. Confidence Assessment
        confidence_trend = [
            (analysis_results.get("confidence_score", 0.5) + np.sin(i / 5) * 0.1)
            for i in periods[:15]
        ]
        fig.add_trace(
            go.Bar(
                x=list(range(15)),
                y=confidence_trend,
                name="Analysis Confidence",
                marker_color=config.colors[2],
            ),
            row=2,
            col=1,
        )

        # 4. Market Prediction (alternative scenarios)
        scenarios = [
            "Bullish",
            "Bearish",
            "Neutral",
            "High Volatility",
            "Low Volatility",
        ]
        probabilities = [0.25, 0.20, 0.35, 0.10, 0.10]

        fig.add_trace(
            go.Pie(
                labels=scenarios,
                values=probabilities,
                name="Market Scenarios",
                marker_colors=config.colors,
            ),
            row=2,
            col=2,
        )

        # Update layout
        symbol = analysis_results.get("symbol", "UNKNOWN")
        fig.update_layout(
            title=f"UCUP Market Analysis: {symbol}",
            height=config.height,
            showlegend=config.show_legend,
        )

        return fig

    def create_multi_asset_comparison(
        self, symbols: List[str], days: int = 30
    ) -> go.Figure:
        """
        Compare multiple assets with uncertainty analysis.

        Args:
            symbols: List of stock symbols to compare
            days: Days of data to analyze

        Returns:
            Multi-asset comparison visualization
        """
        # Fetch data for all symbols
        asset_data = {}
        for symbol in symbols:
            data = self.analyze_market_uncertainty(symbol, days)
            asset_data[symbol] = data

        # Create comparison visualization
        fig = go.Figure()

        # Price performance comparison
        for i, (symbol, data) in enumerate(asset_data.items()):
            fig.add_trace(
                go.Bar(
                    name=symbol,
                    x=["Price Change %", "Volatility", "Confidence"],
                    y=[
                        data.get("price_change_pct", 0),
                        data.get("volatility", 0),
                        data.get("confidence_score", 0),
                    ],
                    offsetgroup=i,
                )
            )

        fig.update_layout(
            title="Multi-Asset Performance Comparison with Uncertainty",
            xaxis_title="Metrics",
            yaxis_title="Values",
            barmode="group",
        )

        return fig


# Integration utilities
def create_ucup_visualization(
    results: Dict[str, Any], visualization_type: str = "uncertainty", **kwargs
) -> Union[go.Figure, str]:
    """
    Factory function for creating UCUP visualizations.

    Args:
        results: UCUP analysis results
        visualization_type: Type of visualization to create
        **kwargs: Additional arguments

    Returns:
        Plotly Figure or HTML string
    """
    if not VISUALIZATION_DEPS_AVAILABLE:
        return "<div style='color: red;'>Visualization dependencies not available</div>"

    visualizer = PlotlyVisualizer()

    if visualization_type == "uncertainty":
        return visualizer.plot_uncertainty_analysis(results)
    elif visualization_type == "confidence_ranges":
        return visualizer.plot_confidence_ranges(results)
    elif visualization_type == "performance_trends":
        return visualizer.plot_performance_trends(results)
    elif visualization_type == "agent_matrix":
        return visualizer.visualize_agent_matrix(results)
    else:
        return visualizer.plot_uncertainty_analysis(results)


def quick_visualize_cli(
    results: Dict[str, Any], output_file: str = None, format: str = "html"
) -> str:
    """
    Quick visualization for CLI usage.

    Args:
        results: UCUP results to visualize
        output_file: Optional output file path
        format: Output format ('html', 'png', 'svg', 'json')

    Returns:
        Output file path or HTML string
    """
    if not VISUALIZATION_DEPS_AVAILABLE:
        return (
            "Visualization dependencies not available. Install plotly, pandas, numpy."
        )

    visualizer = PlotlyVisualizer()
    fig = visualizer.plot_uncertainty_analysis(results)

    if output_file:
        return visualizer.export_chart(fig, format, output_file.split(".")[0])
    else:
        # Generate temporary HTML file
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w") as f:
            fig.write_html(f)
            temp_path = f.name

        return f"Visualization created: file://{temp_path}"
