"""
UCUP Metrics and Benefits Tracking System

This module tracks and displays UCUP benefits to developers during builds and test runs.
It shows time saved, errors prevented, and other valuable metrics.
"""

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BuildMetrics:
    """Metrics for a single build"""

    timestamp: str
    duration: float
    errors_detected: int
    warnings_detected: int
    uncertainty_checks: int
    validation_runs: int
    tests_run: int
    tests_passed: int
    tests_failed: int


@dataclass
class DonationPromptHistory:
    """Tracks donation prompt display history"""

    last_build_prompt: Optional[str] = None
    last_test_prompt: Optional[str] = None
    last_cli_prompt: Optional[str] = None
    last_vscode_prompt: Optional[str] = None
    build_prompt_count: int = 0
    test_prompt_count: int = 0
    cli_prompt_count: int = 0
    vscode_prompt_count: int = 0


@dataclass
class UCUPBenefits:
    """Cumulative UCUP benefits"""

    total_builds: int = 0
    total_tests: int = 0
    total_time_saved: float = 0.0
    total_errors_prevented: int = 0
    total_uncertainty_checks: int = 0
    total_validations: int = 0
    avg_build_improvement: float = 0.0
    avg_test_improvement: float = 0.0
    first_use_date: Optional[str] = None
    last_use_date: Optional[str] = None
    donation_history: Optional[DonationPromptHistory] = None


class MetricsTracker:
    """Tracks and persists UCUP usage metrics"""

    def __init__(self, metrics_dir: Optional[Path] = None):
        """Initialize metrics tracker with optional custom directory"""
        if metrics_dir is None:
            # Store in user's home directory
            self.metrics_dir = Path.home() / ".ucup" / "metrics"
        else:
            self.metrics_dir = Path(metrics_dir)

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.metrics_dir / "ucup_metrics.json"
        self.lock = threading.Lock()

        # Load existing metrics
        self.benefits = self._load_metrics()

        # Initialize donation history if not present
        if self.benefits.donation_history is None:
            self.benefits.donation_history = DonationPromptHistory()

    def _load_metrics(self) -> UCUPBenefits:
        """Load metrics from disk"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, "r") as f:
                    data = json.load(f)
                return UCUPBenefits(**data)
            except (json.JSONDecodeError, TypeError):
                return UCUPBenefits()
        return UCUPBenefits()

    def _save_metrics(self) -> None:
        """Save metrics to disk"""
        with self.lock:
            with open(self.metrics_file, "w") as f:
                json.dump(asdict(self.benefits), f, indent=2)

    def record_build(self, metrics: BuildMetrics) -> None:
        """Record metrics from a build"""
        with self.lock:
            self.benefits.total_builds += 1
            self.benefits.total_errors_prevented += metrics.errors_detected
            self.benefits.total_uncertainty_checks += metrics.uncertainty_checks
            self.benefits.total_validations += metrics.validation_runs

            # Estimate time saved (conservative estimate: 5 min per error caught early)
            time_saved = metrics.errors_detected * 5 * 60  # 5 minutes in seconds
            self.benefits.total_time_saved += time_saved

            # Update dates
            now = datetime.now().isoformat()
            if self.benefits.first_use_date is None:
                self.benefits.first_use_date = now
            self.benefits.last_use_date = now

            # Update averages
            if self.benefits.total_builds > 0:
                self.benefits.avg_build_improvement = (
                    self.benefits.total_time_saved / self.benefits.total_builds
                )

        self._save_metrics()

    def record_test_run(
        self, tests_run: int, tests_passed: int, tests_failed: int, duration: float
    ) -> None:
        """Record metrics from a test run"""
        with self.lock:
            self.benefits.total_tests += tests_run

            # Estimate improvement from UCUP's intelligent testing features
            # Conservative: 10% faster test discovery and execution
            time_saved = duration * 0.10
            self.benefits.total_time_saved += time_saved

            # Update dates
            now = datetime.now().isoformat()
            if self.benefits.first_use_date is None:
                self.benefits.first_use_date = now
            self.benefits.last_use_date = now

            # Update averages
            if self.benefits.total_tests > 0:
                self.benefits.avg_test_improvement = (
                    self.benefits.total_time_saved
                    / max(1, self.benefits.total_builds + 1)
                )

        self._save_metrics()

    def should_show_donation_prompt(self, prompt_type: str) -> bool:
        """
        Check if donation prompt should be shown for the given type.

        Frequency control:
        - Once per day per workflow type
        - Or after every 10th use of that workflow
        - Whichever comes first
        """
        with self.lock:
            if not self.benefits.donation_history:
                return True

            history = self.benefits.donation_history
            now = datetime.now()
            today = now.date().isoformat()

            if prompt_type == "build":
                # Show if not shown today OR after every 10 builds
                last_shown = history.last_build_prompt
                count = history.build_prompt_count
                return (last_shown != today) or (count % 10 == 0)

            elif prompt_type == "test":
                last_shown = history.last_test_prompt
                count = history.test_prompt_count
                return (last_shown != today) or (count % 10 == 0)

            elif prompt_type == "cli":
                last_shown = history.last_cli_prompt
                count = history.cli_prompt_count
                return (last_shown != today) or (count % 10 == 0)

            elif prompt_type == "vscode":
                last_shown = history.last_vscode_prompt
                count = history.vscode_prompt_count
                return (last_shown != today) or (count % 10 == 0)

            return True  # Default to showing for unknown types

    def record_donation_prompt_shown(self, prompt_type: str) -> None:
        """Record that a donation prompt was shown for the given type."""
        with self.lock:
            if not self.benefits.donation_history:
                self.benefits.donation_history = DonationPromptHistory()

            history = self.benefits.donation_history
            now = datetime.now().date().isoformat()

            if prompt_type == "build":
                history.last_build_prompt = now
                history.build_prompt_count += 1
            elif prompt_type == "test":
                history.last_test_prompt = now
                history.test_prompt_count += 1
            elif prompt_type == "cli":
                history.last_cli_prompt = now
                history.cli_prompt_count += 1
            elif prompt_type == "vscode":
                history.last_vscode_prompt = now
                history.vscode_prompt_count += 1

            self._save_metrics()

    def get_benefits(self) -> UCUPBenefits:
        """Get current benefits"""
        return self.benefits


class BenefitsDisplay:
    """Displays UCUP benefits to developers"""

    # Color codes for terminal output
    COLORS = {
        "green": "\033[92m",
        "blue": "\033[94m",
        "yellow": "\033[93m",
        "cyan": "\033[96m",
        "magenta": "\033[95m",
        "bold": "\033[1m",
        "end": "\033[0m",
    }

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable time"""
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
        else:
            days = seconds / 86400
            return f"{days:.1f} days"

    @classmethod
    def _colorize(cls, text: str, color: str) -> str:
        """Add color to text if terminal supports it"""
        if os.getenv("NO_COLOR") or not os.isatty(1):
            return text
        return f"{cls.COLORS.get(color, '')}{text}{cls.COLORS['end']}"

    @classmethod
    def show_build_benefits(cls, tracker: MetricsTracker) -> None:
        """Display benefits during build"""
        benefits = tracker.get_benefits()

        print()
        print(cls._colorize("╔" + "═" * 78 + "╗", "cyan"))
        print(cls._colorize("║" + " " * 78 + "║", "cyan"))

        title = "🚀 UCUP is Making Your Development Better!"
        padding = (78 - len(title)) // 2
        print(
            cls._colorize(
                "║" + " " * padding + title + " " * (78 - padding - len(title)) + "║",
                "cyan",
            )
        )

        print(cls._colorize("║" + " " * 78 + "║", "cyan"))
        print(cls._colorize("╠" + "═" * 78 + "╣", "cyan"))

        # Show key metrics
        if benefits.total_builds > 0:
            print(
                cls._colorize("║ ", "cyan")
                + cls._colorize("✓ ", "green")
                + f"Total Builds with UCUP: {cls._colorize(str(benefits.total_builds), 'bold')}"
                + " " * (62 - len(str(benefits.total_builds)))
                + cls._colorize("║", "cyan")
            )

        if benefits.total_time_saved > 0:
            time_str = cls._format_time(benefits.total_time_saved)
            print(
                cls._colorize("║ ", "cyan")
                + cls._colorize("⏱  ", "yellow")
                + f"Time Saved: {cls._colorize(time_str, 'bold')}"
                + " " * (64 - len(time_str))
                + cls._colorize("║", "cyan")
            )

        if benefits.total_errors_prevented > 0:
            print(
                cls._colorize("║ ", "cyan")
                + cls._colorize("🛡️  ", "green")
                + f"Errors Prevented: {cls._colorize(str(benefits.total_errors_prevented), 'bold')}"
                + " " * (58 - len(str(benefits.total_errors_prevented)))
                + cls._colorize("║", "cyan")
            )

        if benefits.total_uncertainty_checks > 0:
            print(
                cls._colorize("║ ", "cyan")
                + cls._colorize("📊 ", "blue")
                + f"Uncertainty Checks: {cls._colorize(str(benefits.total_uncertainty_checks), 'bold')}"
                + " " * (56 - len(str(benefits.total_uncertainty_checks)))
                + cls._colorize("║", "cyan")
            )

        if benefits.total_validations > 0:
            print(
                cls._colorize("║ ", "cyan")
                + cls._colorize("✔️  ", "green")
                + f"Validations Run: {cls._colorize(str(benefits.total_validations), 'bold')}"
                + " " * (59 - len(str(benefits.total_validations)))
                + cls._colorize("║", "cyan")
            )

        if benefits.total_tests > 0:
            print(
                cls._colorize("║ ", "cyan")
                + cls._colorize("🧪 ", "magenta")
                + f"Tests Enhanced: {cls._colorize(str(benefits.total_tests), 'bold')}"
                + " " * (60 - len(str(benefits.total_tests)))
                + cls._colorize("║", "cyan")
            )

        print(cls._colorize("║" + " " * 78 + "║", "cyan"))
        print(cls._colorize("╠" + "═" * 78 + "╣", "cyan"))

        # Show motivational message
        message = "UCUP helps you build more reliable AI agents faster! 💪"
        padding = (78 - len(message)) // 2
        print(
            cls._colorize(
                "║"
                + " " * padding
                + message
                + " " * (78 - padding - len(message))
                + "║",
                "cyan",
            )
        )

        print(cls._colorize("║" + " " * 78 + "║", "cyan"))
        print(cls._colorize("╚" + "═" * 78 + "╝", "cyan"))
        print()

    @classmethod
    def show_test_benefits(
        cls,
        tracker: MetricsTracker,
        current_tests: int = 0,
        current_duration: float = 0.0,
    ) -> None:
        """Display benefits during test runs"""
        benefits = tracker.get_benefits()

        print()
        print(cls._colorize("┌" + "─" * 78 + "┐", "blue"))

        title = "🧪 UCUP Test Suite Benefits"
        padding = (78 - len(title)) // 2
        print(
            cls._colorize(
                "│" + " " * padding + title + " " * (78 - padding - len(title)) + "│",
                "blue",
            )
        )

        print(cls._colorize("├" + "─" * 78 + "┤", "blue"))

        # Current test session
        if current_tests > 0:
            print(
                cls._colorize("│ ", "blue")
                + f"Current Test Run: {cls._colorize(str(current_tests), 'bold')} tests"
                + " " * (59 - len(str(current_tests)))
                + cls._colorize("│", "blue")
            )

        # Cumulative stats
        if benefits.total_tests > 0:
            print(
                cls._colorize("│ ", "blue")
                + f"Total Tests with UCUP: {cls._colorize(str(benefits.total_tests), 'bold')}"
                + " " * (55 - len(str(benefits.total_tests)))
                + cls._colorize("│", "blue")
            )

        if benefits.total_time_saved > 0:
            time_str = cls._format_time(benefits.total_time_saved)
            print(
                cls._colorize("│ ", "blue")
                + f"Cumulative Time Saved: {cls._colorize(time_str, 'bold')}"
                + " " * (55 - len(time_str))
                + cls._colorize("│", "blue")
            )

        print(cls._colorize("│ ", "blue") + " " * 78 + cls._colorize("│", "blue"))

        # Feature highlights
        print(
            cls._colorize("│ ", "blue")
            + cls._colorize("UCUP Features Active:", "bold")
            + " " * 56
            + cls._colorize("│", "blue")
        )

        print(
            cls._colorize("│ ", "blue")
            + "  • Intelligent test generation and prioritization"
            + " " * 32
            + cls._colorize("│", "blue")
        )

        print(
            cls._colorize("│ ", "blue")
            + "  • Uncertainty-aware validation"
            + " " * 43
            + cls._colorize("│", "blue")
        )

        print(
            cls._colorize("│ ", "blue")
            + "  • Probabilistic reasoning verification"
            + " " * 36
            + cls._colorize("│", "blue")
        )

        print(
            cls._colorize("│ ", "blue")
            + "  • Automated test insights and recommendations"
            + " " * 29
            + cls._colorize("│", "blue")
        )

        print(cls._colorize("└" + "─" * 78 + "┘", "blue"))
        print()


# Global tracker instance
_global_tracker: Optional[MetricsTracker] = None


def get_global_tracker() -> MetricsTracker:
    """Get or create the global metrics tracker"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = MetricsTracker()
    return _global_tracker


def show_build_benefits() -> None:
    """Show UCUP benefits during build - convenience function"""
    tracker = get_global_tracker()
    BenefitsDisplay.show_build_benefits(tracker)


def show_test_benefits(tests_run: int = 0, duration: float = 0.0) -> None:
    """Show UCUP benefits during tests - convenience function"""
    tracker = get_global_tracker()
    BenefitsDisplay.show_test_benefits(tracker, tests_run, duration)


def record_build_metrics(**kwargs) -> None:
    """Record build metrics - convenience function"""
    tracker = get_global_tracker()
    metrics = BuildMetrics(
        timestamp=datetime.now().isoformat(),
        duration=kwargs.get("duration", 0.0),
        errors_detected=kwargs.get("errors_detected", 0),
        warnings_detected=kwargs.get("warnings_detected", 0),
        uncertainty_checks=kwargs.get("uncertainty_checks", 0),
        validation_runs=kwargs.get("validation_runs", 0),
        tests_run=kwargs.get("tests_run", 0),
        tests_passed=kwargs.get("tests_passed", 0),
        tests_failed=kwargs.get("tests_failed", 0),
    )
    tracker.record_build(metrics)


def should_show_donation_prompt(prompt_type: str) -> bool:
    """Check if donation prompt should be shown - convenience function"""
    tracker = get_global_tracker()
    return tracker.should_show_donation_prompt(prompt_type)


def record_donation_prompt_shown(prompt_type: str) -> None:
    """Record that donation prompt was shown - convenience function"""
    tracker = get_global_tracker()
    tracker.record_donation_prompt_shown(prompt_type)
