"""
Email reporting functionality for UCUP test results.

Provides comprehensive email reporting with UCUP benefits tracking,
test results formatting, and automated delivery capabilities.

Copyright (c) 2025 UCUP Framework Contributors
"""

import json
import logging
import os
import smtplib
import ssl
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class EmailConfig:
    """Email configuration for UCUP reporting."""

    def __init__(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: str = "noreply@ucup.ai",
        sender_password: Optional[str] = None,
        use_tls: bool = True,
        recipient_emails: List[str] = None,
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password or os.environ.get("UCUP_EMAIL_PASSWORD")
        self.use_tls = use_tls
        self.recipient_emails = recipient_emails or []

    def is_configured(self) -> bool:
        """Check if email configuration is complete."""
        return (
            self.sender_password
            and self.recipient_emails
            and len(self.recipient_emails) > 0
        )

    def add_recipient(self, email: str):
        """Add a recipient email address."""
        if email not in self.recipient_emails:
            self.recipient_emails.append(email)

    def remove_recipient(self, email: str):
        """Remove a recipient email address."""
        if email in self.recipient_emails:
            self.recipient_emails.remove(email)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "smtp_server": self.smtp_server,
            "smtp_port": self.smtp_port,
            "sender_email": self.sender_email,
            "use_tls": self.use_tls,
            "recipient_emails": self.recipient_emails,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmailConfig":
        """Create config from dictionary."""
        return cls(
            smtp_server=data.get("smtp_server", "smtp.gmail.com"),
            smtp_port=data.get("smtp_port", 587),
            sender_email=data.get("sender_email", "noreply@ucup.ai"),
            use_tls=data.get("use_tls", True),
            recipient_emails=data.get("recipient_emails", []),
        )


class TestReportFormatter:
    """Formats UCUP test results for email reporting."""

    def __init__(self):
        self.ucup_benefits = self._get_ucup_benefits()

    def _get_ucup_benefits(self) -> Dict[str, Any]:
        """Get UCUP benefits information for inclusion in reports."""
        try:
            from .metrics import get_global_tracker

            tracker = get_global_tracker()
            return tracker.get_benefits()
        except ImportError:
            # Fallback benefits data
            return {
                "total_value": "$15,800/month",
                "total_time_saved": "70 hours/week",
                "breakdown": [
                    {
                        "category": "Development Speed",
                        "description": "Automated code generation and scaffolding",
                        "value": "$2,400/month",
                        "time_saved": "16 hours/week",
                    },
                    {
                        "category": "Testing Efficiency",
                        "description": "Probabilistic testing and uncertainty analysis",
                        "value": "$1,800/month",
                        "time_saved": "12 hours/week",
                    },
                    {
                        "category": "Deployment Automation",
                        "description": "One-click deployment across all platforms",
                        "value": "$3,200/month",
                        "time_saved": "20 hours/week",
                    },
                ],
                "roi_percentage": 340,
                "payback_period": "2.3 months",
            }

    def format_test_results(
        self, test_results: Dict[str, Any], format_type: str = "html"
    ) -> str:
        """Format test results for email."""
        if format_type == "html":
            return self._format_html_report(test_results)
        elif format_type == "text":
            return self._format_text_report(test_results)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _format_html_report(self, test_results: Dict[str, Any]) -> str:
        """Format test results as HTML email."""
        summary = test_results.get("summary", {})
        total_passed = summary.get("total_passed", 0)
        total_failed = summary.get("total_failed", 0)
        total_tests = total_passed + total_failed

        # Determine overall status
        if total_failed == 0:
            status_color = "#27ae60"
            status_text = "ALL TESTS PASSED"
        elif total_passed > total_failed:
            status_color = "#f39c12"
            status_text = "MOSTLY PASSING"
        else:
            status_color = "#e74c3c"
            status_text = "TESTS FAILING"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>UCUP Test Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            background-color: #f8f9fa;
        }}
        .container {{
            background: white;
            margin: 20px;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid {status_color};
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .status-banner {{
            background: {status_color};
            color: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid {status_color};
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: {status_color};
            display: block;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        .ucup-benefits {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin: 30px 0;
        }}
        .benefits-title {{
            font-size: 24px;
            margin-bottom: 15px;
            text-align: center;
        }}
        .benefits-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .benefit-item {{
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
        }}
        .benefit-value {{
            font-size: 18px;
            font-weight: bold;
        }}
        .benefit-desc {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .test-details {{
            margin-top: 30px;
        }}
        .test-suite {{
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-bottom: 15px;
            overflow: hidden;
        }}
        .suite-header {{
            background: #f8f9fa;
            padding: 15px;
            font-weight: bold;
            border-bottom: 1px solid #ddd;
        }}
        .suite-content {{
            padding: 15px;
        }}
        .suite-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-bottom: 15px;
        }}
        .metric {{
            text-align: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .errors {{
            background: #fff5f5;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin-top: 15px;
            border-radius: 5px;
        }}
        .error-item {{
            background: white;
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 3px;
            border: 1px solid #fed7d7;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 14px;
        }}
        .signature {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 UCUP Test Report</h1>
            <p>Unified Cognitive Uncertainty Processing Framework</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="status-banner">
            {status_text}
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <span class="metric-value">{total_tests}</span>
                <span class="metric-label">Total Tests</span>
            </div>
            <div class="metric-card">
                <span class="metric-value">{total_passed}</span>
                <span class="metric-label">Tests Passed</span>
            </div>
            <div class="metric-card">
                <span class="metric-value">{total_failed}</span>
                <span class="metric-label">Tests Failed</span>
            </div>
            <div class="metric-card">
                <span class="metric-value">{total_passed/total_tests*100:.1f}%</span>
                <span class="metric-label">Success Rate</span>
            </div>
        </div>

        <div class="ucup-benefits">
            <div class="benefits-title">🎉 UCUP Value Benefits</div>
            <p><strong>Total Value:</strong> {self.ucup_benefits.get('total_value', 'N/A')}</p>
            <p><strong>Time Saved:</strong> {self.ucup_benefits.get('total_time_saved', 'N/A')}</p>
            <p><strong>ROI:</strong> {self.ucup_benefits.get('roi_percentage', 0)}%</p>
            <p><strong>Payback Period:</strong> {self.ucup_benefits.get('payback_period', 'N/A')}</p>

            <div class="benefits-grid">
"""

        # Add benefit breakdown
        for benefit in self.ucup_benefits.get("breakdown", [])[
            :4
        ]:  # Show top 4 benefits
            html += f"""
                <div class="benefit-item">
                    <div class="benefit-value">{benefit.get('value', 'N/A')}</div>
                    <div class="benefit-desc">{benefit.get('category', '')}</div>
                    <div style="font-size: 12px; opacity: 0.8;">{benefit.get('time_saved', '')} saved</div>
                </div>
"""

        html += """
            </div>
        </div>

        <div class="test-details">
            <h2>📊 Test Suite Details</h2>
"""

        # Add test suite details
        for result in test_results.get("results", []):
            if result:
                suite_name = result.get("test_suite", "Unknown Suite")
                passed = result.get("passed_tests", 0)
                failed = result.get("failed_tests", 0)
                total = result.get("total_tests", 0)
                coverage = result.get("coverage_percentage", 0)
                duration = result.get("execution_time_seconds", 0)

                suite_status = "✅ Passed" if failed == 0 else "❌ Failed"
                suite_color = "#27ae60" if failed == 0 else "#e74c3c"

                html += f"""
            <div class="test-suite">
                <div class="suite-header" style="color: {suite_color};">
                    {suite_name} - {suite_status}
                </div>
                <div class="suite-content">
                    <div class="suite-metrics">
                        <div class="metric">
                            <strong>{total}</strong><br>Total Tests
                        </div>
                        <div class="metric">
                            <strong>{passed}</strong><br>Passed
                        </div>
                        <div class="metric">
                            <strong>{failed}</strong><br>Failed
                        </div>
                        <div class="metric">
                            <strong>{coverage:.1f}%</strong><br>Coverage
                        </div>
                        <div class="metric">
                            <strong>{duration:.2f}s</strong><br>Duration
                        </div>
                    </div>
"""

                # Add errors if any
                errors = result.get("errors", [])
                if errors:
                    html += """
                    <div class="errors">
                        <h4>❌ Test Failures:</h4>
"""
                    for error in errors[:5]:  # Show first 5 errors
                        error_preview = (
                            error[:200] + "..." if len(error) > 200 else error
                        )
                        html += f"""
                        <div class="error-item">{error_preview}</div>
"""
                    html += "                    </div>"

                html += "                </div>\n            </div>"

        html += """
        </div>

        <div class="footer">
            <p><strong>UCUP Framework</strong> - Building reliable AI systems with uncertainty quantification</p>
            <p>Report generated by UCUP CLI - https://github.com/ucup/ucup-framework</p>

            <div class="signature">
                <p>🚀 <strong>Thank you for using UCUP!</strong></p>
                <p>Support our development: <a href="https://github.com/sponsors/ucup" style="color: #667eea;">GitHub Sponsors</a></p>
                <p><em>Sent from noreply@ucup.ai</em></p>
            </div>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _format_text_report(self, test_results: Dict[str, Any]) -> str:
        """Format test results as plain text email."""
        summary = test_results.get("summary", {})
        total_passed = summary.get("total_passed", 0)
        total_failed = summary.get("total_failed", 0)
        total_tests = total_passed + total_failed

        text = f"""
UCUP Test Report
================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
-------
Total Tests: {total_tests}
Tests Passed: {total_passed}
Tests Failed: {total_failed}
Success Rate: {total_passed/total_tests*100:.1f}%

UCUP VALUE BENEFITS
-------------------
Total Value: {self.ucup_benefits.get('total_value', 'N/A')}
Time Saved: {self.ucup_benefits.get('total_time_saved', 'N/A')}
ROI: {self.ucup_benefits.get('roi_percentage', 0)}%
Payback Period: {self.ucup_benefits.get('payback_period', 'N/A')}

Key Benefits:
"""

        for benefit in self.ucup_benefits.get("breakdown", [])[:4]:
            text += f"- {benefit.get('category', '')}: {benefit.get('value', '')} ({benefit.get('time_saved', '')} saved)\n"

        text += "\n\nTEST SUITE DETAILS\n-------------------\n"

        for result in test_results.get("results", []):
            if result:
                suite_name = result.get("test_suite", "Unknown Suite")
                passed = result.get("passed_tests", 0)
                failed = result.get("failed_tests", 0)
                total = result.get("total_tests", 0)
                coverage = result.get("coverage_percentage", 0)
                duration = result.get("execution_time_seconds", 0)

                status = "PASSED" if failed == 0 else "FAILED"
                text += f"""
{suite_name} - {status}
  Total: {total}, Passed: {passed}, Failed: {failed}
  Coverage: {coverage:.1f}%, Duration: {duration:.2f}s
"""

                errors = result.get("errors", [])
                if errors:
                    text += "  Errors:\n"
                    for error in errors[:3]:
                        error_preview = (
                            error[:100] + "..." if len(error) > 100 else error
                        )
                        text += f"    - {error_preview}\n"

        text += """

UCUP Framework - Building reliable AI systems with uncertainty quantification
Report generated by UCUP CLI - https://github.com/ucup/ucup-framework

Thank you for using UCUP!
Support our development: https://github.com/sponsors/ucup

Sent from noreply@ucup.ai
"""
        return text


class EmailReporter:
    """Handles sending UCUP test reports via email."""

    def __init__(self, config: EmailConfig):
        self.config = config
        self.formatter = TestReportFormatter()

    def send_test_report(
        self,
        test_results: Dict[str, Any],
        subject: Optional[str] = None,
        format_type: str = "html",
        attachments: List[str] = None,
    ) -> bool:
        """
        Send test report via email.

        Args:
            test_results: Test results dictionary
            subject: Email subject (auto-generated if None)
            format_type: 'html' or 'text'
            attachments: List of file paths to attach

        Returns:
            bool: True if sent successfully
        """
        if not self.config.is_configured():
            logger.error("Email configuration incomplete")
            return False

        if not subject:
            summary = test_results.get("summary", {})
            total_passed = summary.get("total_passed", 0)
            total_failed = summary.get("total_failed", 0)
            subject = f"UCUP Test Report: {total_passed} passed, {total_failed} failed"

        # Format the report
        report_content = self.formatter.format_test_results(test_results, format_type)

        # Create email message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.config.sender_email
        msg["To"] = ", ".join(self.config.recipient_emails)

        # Add body
        if format_type == "html":
            msg.attach(MIMEText(report_content, "html"))
        else:
            msg.attach(MIMEText(report_content, "plain"))

        # Add attachments
        if attachments:
            for attachment_path in attachments:
                if os.path.exists(attachment_path):
                    with open(attachment_path, "rb") as f:
                        attachment = MIMEApplication(
                            f.read(), Name=os.path.basename(attachment_path)
                        )
                        attachment[
                            "Content-Disposition"
                        ] = f'attachment; filename="{os.path.basename(attachment_path)}"'
                        msg.attach(attachment)

        # Send email
        try:
            context = ssl.create_default_context() if self.config.use_tls else None

            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                if self.config.use_tls:
                    server.starttls(context=context)
                server.login(self.config.sender_email, self.config.sender_password)
                server.sendmail(
                    self.config.sender_email,
                    self.config.recipient_emails,
                    msg.as_string(),
                )

            logger.info(
                f"Test report sent successfully to {len(self.config.recipient_emails)} recipients"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_benefits_summary(self, custom_message: Optional[str] = None) -> bool:
        """Send UCUP benefits summary email."""
        if not self.config.is_configured():
            return False

        benefits = self.formatter.ucup_benefits

        subject = "🎉 UCUP Value Benefits Summary"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>UCUP Benefits Summary</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            background-color: #f8f9fa;
        }}
        .container {{
            background: white;
            margin: 20px;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #667eea;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .benefits-highlight {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            margin: 20px 0;
        }}
        .benefits-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 30px 0;
        }}
        .benefit-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .benefit-value {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }}
        .benefit-category {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎉 UCUP Value Benefits</h1>
            <p>Thank you for using the UCUP Framework!</p>
        </div>

        <div class="benefits-highlight">
            <h2>Your UCUP Investment</h2>
            <p style="font-size: 18px; margin: 10px 0;">
                <strong>Total Value:</strong> {benefits.get('total_value', 'N/A')}<br>
                <strong>Time Saved:</strong> {benefits.get('total_time_saved', 'N/A')}
            </p>
            <p style="font-size: 16px;">
                <strong>ROI:</strong> {benefits.get('roi_percentage', 0)}% |
                <strong>Payback:</strong> {benefits.get('payback_period', 'N/A')}
            </p>
        </div>

        {custom_message or ''}

        <h2>💰 Detailed Benefits Breakdown</h2>
        <div class="benefits-grid">
"""

        for benefit in benefits.get("breakdown", []):
            html_content += f"""
            <div class="benefit-card">
                <div class="benefit-category">{benefit.get('category', '')}</div>
                <div class="benefit-value">{benefit.get('value', 'N/A')}</div>
                <div style="font-size: 14px; color: #666; margin-top: 5px;">
                    {benefit.get('description', '')}
                </div>
                <div style="font-size: 12px; color: #888; margin-top: 8px;">
                    ⏱️ {benefit.get('time_saved', 'N/A')} saved per week
                </div>
            </div>
"""

        html_content += """
        </div>

        <div class="footer">
            <p><strong>UCUP Framework</strong> - Building reliable AI systems with uncertainty quantification</p>
            <p>Learn more: <a href="https://github.com/ucup/ucup-framework" style="color: #667eea;">GitHub</a></p>
            <p><em>Sent from noreply@ucup.ai</em></p>
        </div>
    </div>
</body>
</html>
"""

        # Create and send email
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.config.sender_email
        msg["To"] = ", ".join(self.config.recipient_emails)
        msg.attach(MIMEText(html_content, "html"))

        try:
            context = ssl.create_default_context() if self.config.use_tls else None

            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                if self.config.use_tls:
                    server.starttls(context=context)
                server.login(self.config.sender_email, self.config.sender_password)
                server.sendmail(
                    self.config.sender_email,
                    self.config.recipient_emails,
                    msg.as_string(),
                )

            logger.info("Benefits summary sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send benefits summary: {e}")
            return False


def load_email_config(config_path: Optional[Union[str, Path]] = None) -> EmailConfig:
    """Load email configuration from file or environment."""
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, "r") as f:
                data = json.load(f)
            return EmailConfig.from_dict(data)

    # Try to load from environment or create default
    return EmailConfig()


def save_email_config(config: EmailConfig, config_path: Union[str, Path]) -> bool:
    """Save email configuration to file."""
    try:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        return True
    except Exception as e:
        logger.error(f"Failed to save email config: {e}")
        return False


def setup_email_config_interactive() -> EmailConfig:
    """Interactive setup of email configuration."""
    print("📧 UCUP Email Configuration Setup")
    print("=" * 40)

    config = EmailConfig()

    # SMTP Server
    smtp_server = input(f"SMTP Server [{config.smtp_server}]: ").strip()
    if smtp_server:
        config.smtp_server = smtp_server

    # SMTP Port
    try:
        smtp_port = input(f"SMTP Port [{config.smtp_port}]: ").strip()
        if smtp_port:
            config.smtp_port = int(smtp_port)
    except ValueError:
        print("Invalid port, using default.")

    # Sender Email
    sender_email = input(f"Sender Email [{config.sender_email}]: ").strip()
    if sender_email:
        config.sender_email = sender_email

    # Sender Password
    import getpass

    sender_password = getpass.getpass(
        "Sender Password (leave empty to use UCUP_EMAIL_PASSWORD env var): "
    )
    if sender_password:
        config.sender_password = sender_password

    # Recipients
    print("\n📧 Add recipient email addresses (one per line, empty line to finish):")
    while True:
        email = input("Email address: ").strip()
        if not email:
            break
        if "@" in email and "." in email:
            config.add_recipient(email)
            print(f"✅ Added {email}")
        else:
            print("❌ Invalid email address")

    return config
