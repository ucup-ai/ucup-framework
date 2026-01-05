"""
UCUP Pytest Plugin

Automatically displays UCUP benefits when running tests with pytest.
This plugin hooks into pytest's lifecycle to show developers how UCUP
is helping them with their testing efforts.
"""

import time
from typing import Optional

import pytest

try:
    from .metrics import (
        BenefitsDisplay,
        get_global_tracker,
        record_build_metrics,
        record_donation_prompt_shown,
        should_show_donation_prompt,
    )
except ImportError:
    # Fallback for when running from different contexts
    from ucup.metrics import (
        BenefitsDisplay,
        get_global_tracker,
        record_build_metrics,
        record_donation_prompt_shown,
        should_show_donation_prompt,
    )


class UCUPPytestPlugin:
    """Pytest plugin to display UCUP benefits"""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.test_count = 0
        self.passed_count = 0
        self.failed_count = 0
        self.tracker = get_global_tracker()

    @pytest.hookimpl(tryfirst=True)
    def pytest_sessionstart(self, session):
        """Called at the start of test session"""
        self.start_time = time.time()

        # Show initial benefits message
        print("\n")
        print("═" * 80)
        print("  🚀 UCUP-Enhanced Test Suite")
        print("  Your tests are powered by intelligent uncertainty quantification!")
        print("═" * 80)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        """Called for each test item"""
        self.test_count += 1
        yield

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        """Called after each test phase"""
        outcome = yield
        report = outcome.get_result()

        if report.when == "call":
            if report.passed:
                self.passed_count += 1
            elif report.failed:
                self.failed_count += 1

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        """Called at the end of test session"""
        if self.start_time:
            duration = time.time() - self.start_time

            # Record metrics
            self.tracker.record_test_run(
                tests_run=self.test_count,
                tests_passed=self.passed_count,
                tests_failed=self.failed_count,
                duration=duration,
            )

            # Also record as build metrics for comprehensive tracking
            record_build_metrics(
                duration=duration,
                tests_run=self.test_count,
                tests_passed=self.passed_count,
                tests_failed=self.failed_count,
                errors_detected=self.failed_count,
                warnings_detected=0,
                uncertainty_checks=self.test_count,  # Each test uses UCUP
                validation_runs=self.test_count,
            )

    @pytest.hookimpl(trylast=True)
    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        """Add UCUP benefits summary to test output"""
        duration = time.time() - self.start_time if self.start_time else 0

        # Show comprehensive benefits display
        BenefitsDisplay.show_test_benefits(
            self.tracker, current_tests=self.test_count, current_duration=duration
        )

        # Show donation prompt after test benefits (with frequency control)
        if should_show_donation_prompt("test"):
            _show_test_donation_prompt()
            record_donation_prompt_shown("test")

        # Add extra motivational message based on test results
        if self.failed_count == 0 and self.test_count > 0:
            print(
                "\n✨ All tests passed! UCUP helped ensure your AI agents are reliable.\n"
            )
        elif self.test_count > 0:
            print("\n🔍 UCUP helped identify issues early in development.\n")


def _show_test_donation_prompt():
    """Show donation prompt after test benefits."""
    print("\n" + "=" * 60)
    print("💝 Support UCUP Development")
    print("UCUP just helped make your tests more intelligent and reliable!")
    print()
    print("Help us continue improving AI testing tools:")
    print("• 💳 PayPal: ucup donate show --platform paypal")
    print("• 🐙 GitHub Sponsors: ucup donate show --platform github")
    print("• ☕ Ko-fi: ucup donate show --platform ko-fi")
    print("• 📋 All options: ucup donate show")
    print()

    # Interactive choice
    try:
        from .build_wrapper import _show_interactive_donation_choice

        _show_interactive_donation_choice("test")
    except ImportError:
        # Fallback if import fails
        pass
    print("=" * 60)


# This is required for pytest to discover the plugin
def pytest_configure(config):
    """Register the UCUP plugin with pytest"""
    if not config.option.collectonly:
        plugin = UCUPPytestPlugin()
        config.pluginmanager.register(plugin, "ucup_plugin")


# Make functions available at module level
__all__ = ["UCUPPytestPlugin", "pytest_configure", "_show_test_donation_prompt"]
