#!/usr/bin/env python3
"""
UCUP Command Line Interface

Provides CLI commands for UCUP framework operations including TOON format
conversion, testing, and token optimization tools.

Copyright (c) 2025 UCUP Framework Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .config import load_ucup_config
from .feature_flags import get_feature_manager
from .metrics import (
    record_build_metrics,
    record_donation_prompt_shown,
    should_show_donation_prompt,
    show_build_benefits,
)
from .toon.toon_formatter import ToonFormatter, ToonSchema
from .validation import validate_data


class UCUPCLI:
    """UCUP Command Line Interface with TOON integration."""

    def __init__(self):
        self.toon_formatter = ToonFormatter()
        self.config = load_ucup_config()
        self._load_toon_config()

    def _load_toon_config(self):
        """Load TOON-specific configuration."""
        # Load custom schemas from config if available
        toon_config = self.config.get("toon", {})
        custom_schemas = toon_config.get("schemas", {})

        for schema_name, schema_config in custom_schemas.items():
            if schema_name not in self.toon_formatter.schemas:
                # Create schema from config
                from .toon.toon_formatter import ToonSchema

                schema = ToonSchema(**schema_config)
                self.toon_formatter.schemas[schema_name] = schema

    def run(self):
        """Main CLI entry point."""
        parser = argparse.ArgumentParser(
            description="UCUP Framework CLI - Token-efficient AI testing and validation",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Convert JSON to TOON for cost savings
  ucup toon convert data.json --output data.toon

  # Compare token usage between formats
  ucup toon compare data.json

  # Create custom schema for repeated data
  ucup toon schema create user_data_schema sample_data.json

  # Validate data with UCUP
  ucup validate data.json --schema user_schema

  # Show token savings report
  ucup toon report

  # Support UCUP development - multiple options available
  ucup donate show --platform paypal    # PayPal reward developers
  ucup donate show --platform github    # GitHub Sponsors
  ucup donate show --platform ko-fi     # Ko-fi reward developers
  ucup donate show                      # All reward developers options

  # Show sponsorship information
  ucup donate sponsor

Support UCUP: 💳 @ucup.ai.2025 (PayPal) - Multiple platforms available
            """,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # TOON commands
        toon_parser = subparsers.add_parser("toon", help="TOON format operations")
        toon_subparsers = toon_parser.add_subparsers(dest="toon_command")

        # TOON convert
        convert_parser = toon_subparsers.add_parser(
            "convert", help="Convert data to TOON format"
        )
        convert_parser.add_argument("input", help="Input file (JSON) or - for stdin")
        convert_parser.add_argument(
            "--output", "-o", help="Output file (default: stdout)"
        )
        convert_parser.add_argument(
            "--schema", help="Schema name to use for optimization"
        )
        convert_parser.add_argument(
            "--format",
            choices=["toon", "json", "auto"],
            default="toon",
            help="Output format (default: toon)",
        )
        convert_parser.add_argument(
            "--compare", action="store_true", help="Show comparison with JSON format"
        )

        # TOON compare
        compare_parser = toon_subparsers.add_parser(
            "compare", help="Compare JSON vs TOON token usage"
        )
        compare_parser.add_argument("input", help="Input file (JSON) or - for stdin")
        compare_parser.add_argument("--schema", help="Schema name to use")
        compare_parser.add_argument(
            "--detailed", action="store_true", help="Show detailed comparison metrics"
        )

        # TOON schema
        schema_parser = toon_subparsers.add_parser("schema", help="Schema management")
        schema_subparsers = schema_parser.add_subparsers(dest="schema_command")

        schema_create = schema_subparsers.add_parser(
            "create", help="Create schema from sample data"
        )
        schema_create.add_argument("name", help="Schema name")
        schema_create.add_argument("sample_file", help="Sample data file")
        schema_create.add_argument("--description", help="Schema description")

        schema_list = schema_subparsers.add_parser(
            "list", help="List available schemas"
        )

        # TOON report
        report_parser = toon_subparsers.add_parser(
            "report", help="Token savings report"
        )
        report_parser.add_argument(
            "--limit",
            type=int,
            default=10,
            help="Number of recent conversions to analyze",
        )

        # Validate command
        validate_parser = subparsers.add_parser(
            "validate", help="Validate data with UCUP"
        )
        validate_parser.add_argument("input", help="Input data file")
        validate_parser.add_argument("--schema", help="Validation schema")
        validate_parser.add_argument(
            "--output", "-o", help="Output validation report file"
        )

        # Config command
        config_parser = subparsers.add_parser("config", help="Configuration management")
        config_parser.add_argument(
            "--show", action="store_true", help="Show current configuration"
        )
        config_parser.add_argument(
            "--init", action="store_true", help="Initialize default configuration"
        )

        # Donation commands
        donate_parser = subparsers.add_parser(
            "donate", help="Reward developers and sponsorship management"
        )
        donate_subparsers = donate_parser.add_subparsers(dest="donate_command")

        donate_show = donate_subparsers.add_parser(
            "show", help="Show reward developers page"
        )
        donate_show.add_argument("--platform", help="Specific platform to show")

        donate_stats = donate_subparsers.add_parser(
            "stats", help="Show reward developers statistics"
        )

        donate_link = donate_subparsers.add_parser(
            "link", help="Create reward developers link"
        )
        donate_link.add_argument("platform", help="Reward developers platform")
        donate_link.add_argument(
            "--amount", type=float, help="Reward developers amount"
        )
        donate_link.add_argument("--currency", default="USD", help="Currency code")

        donate_sponsor = donate_subparsers.add_parser(
            "sponsor", help="Show sponsorship information"
        )

        # Agent scaffolding command
        create_agent_parser = subparsers.add_parser(
            "create-agent", help="Create a complete agent project with scaffolding"
        )
        create_agent_parser.add_argument(
            "name", help="Name of the agent (e.g., customer-support)"
        )
        create_agent_parser.add_argument(
            "--template",
            "-t",
            choices=["probabilistic", "simple", "advanced"],
            default="probabilistic",
            help="Agent template type (default: probabilistic)",
        )
        create_agent_parser.add_argument(
            "--modality",
            "-m",
            choices=["text", "voice", "vision", "multimodal"],
            default="text",
            help="Primary agent modality (default: text)",
        )
        create_agent_parser.add_argument(
            "--with-tests", action="store_true", help="Include comprehensive test suite"
        )
        create_agent_parser.add_argument(
            "--with-monitoring",
            action="store_true",
            help="Include monitoring dashboard",
        )
        create_agent_parser.add_argument(
            "--with-api", action="store_true", help="Include REST API endpoints"
        )
        create_agent_parser.add_argument(
            "--output", "-o", help="Output directory (default: current directory)"
        )

        # Testing command
        test_parser = subparsers.add_parser(
            "test", help="UCUP testing framework with multi-environment support"
        )
        test_subparsers = test_parser.add_subparsers(dest="test_command")

        # Test run
        test_run = test_subparsers.add_parser("run", help="Run comprehensive tests")
        test_run.add_argument("--env", help="Specific test environment to use")
        test_run.add_argument(
            "--suite",
            choices=["unit", "integration", "performance", "reliability", "all"],
            default="all",
            help="Test suite to run (default: all)",
        )
        test_run.add_argument(
            "--parallel", action="store_true", help="Run tests in parallel"
        )
        test_run.add_argument(
            "--coverage", action="store_true", help="Generate coverage reports"
        )
        test_run.add_argument(
            "--html-report", action="store_true", help="Generate HTML test reports"
        )
        test_run.add_argument(
            "--output", "-o", help="Output directory for test results"
        )
        test_run.add_argument(
            "--email", action="store_true", help="Send test results via email"
        )
        test_run.add_argument("--email-subject", help="Custom email subject")

        # Test env
        test_env = test_subparsers.add_parser("env", help="Manage test environments")
        test_env_subparsers = test_env.add_subparsers(dest="env_command")

        test_env_create = test_env_subparsers.add_parser(
            "create", help="Create test environment"
        )
        test_env_create.add_argument("name", help="Environment name")
        test_env_create.add_argument("--python", default="3.11", help="Python version")
        test_env_create.add_argument(
            "--conda", action="store_true", help="Use conda instead of venv"
        )

        test_env_list = test_env_subparsers.add_parser(
            "list", help="List available test environments"
        )

        test_env_destroy = test_env_subparsers.add_parser(
            "destroy", help="Destroy test environment"
        )
        test_env_destroy.add_argument("name", help="Environment name to destroy")

        # Test scaffold
        test_scaffold = test_subparsers.add_parser(
            "scaffold", help="Generate test scaffolding for agents"
        )
        test_scaffold.add_argument("agent_name", help="Name of the agent")
        test_scaffold.add_argument(
            "domain", help="Domain of the agent (e.g., customer-service, data-analysis)"
        )
        test_scaffold.add_argument(
            "--output", "-o", help="Output directory for test files"
        )

        # Test report
        test_report = test_subparsers.add_parser(
            "report", help="Generate test reports and analytics"
        )
        test_report.add_argument(
            "--format",
            choices=["json", "html", "markdown"],
            default="markdown",
            help="Report format",
        )
        test_report.add_argument("--input", help="Input test results file/directory")
        test_report.add_argument("--output", "-o", help="Output file for report")

        # Monitor command
        monitor_parser = subparsers.add_parser(
            "monitor", help="Agent monitoring and validation"
        )
        monitor_subparsers = monitor_parser.add_subparsers(dest="monitor_command")

        # Monitor status
        monitor_status = monitor_subparsers.add_parser(
            "status", help="Show monitoring status and agent health"
        )
        monitor_status.add_argument(
            "agent_id", nargs="?", help="Specific agent ID to check"
        )
        monitor_status.add_argument(
            "--detailed", "-d", action="store_true", help="Show detailed information"
        )

        # Monitor start
        monitor_start = monitor_subparsers.add_parser(
            "start", help="Start agent monitoring"
        )
        monitor_start.add_argument(
            "--interval",
            "-i",
            type=int,
            default=60,
            help="Monitoring interval in seconds (default: 60)",
        )

        # Monitor stop
        monitor_stop = monitor_subparsers.add_parser(
            "stop", help="Stop agent monitoring"
        )

        # Monitor metrics
        monitor_metrics = monitor_subparsers.add_parser(
            "metrics", help="Record agent metrics"
        )
        monitor_metrics.add_argument("agent_id", help="Agent ID to record metrics for")
        monitor_metrics.add_argument(
            "--response-time", type=float, help="Average response time in seconds"
        )
        monitor_metrics.add_argument(
            "--error-rate", type=float, help="Error rate (0.0 to 1.0)"
        )
        monitor_metrics.add_argument(
            "--throughput", type=float, help="Requests per second"
        )
        monitor_metrics.add_argument("--memory", type=float, help="Memory usage in MB")
        monitor_metrics.add_argument("--cpu", type=float, help="CPU usage percentage")
        monitor_metrics.add_argument("--custom", help="Custom metrics as JSON string")

        # Monitor validate
        monitor_validate = monitor_subparsers.add_parser(
            "validate", help="Validate agent behavior"
        )
        monitor_validate.add_argument("agent_id", help="Agent ID to validate")
        monitor_validate.add_argument(
            "--data", help="Agent data as JSON file or string"
        )
        monitor_validate.add_argument(
            "--output", "-o", help="Output validation report file"
        )

        # Monitor alerts
        monitor_alerts = monitor_subparsers.add_parser(
            "alerts", help="Manage monitoring alerts"
        )
        monitor_alerts_subparsers = monitor_alerts.add_subparsers(dest="alerts_command")

        monitor_alerts_list = monitor_alerts_subparsers.add_parser(
            "list", help="List recent alerts"
        )
        monitor_alerts_list.add_argument(
            "agent_id", nargs="?", help="Specific agent ID"
        )
        monitor_alerts_list.add_argument(
            "--limit", type=int, default=10, help="Number of alerts to show"
        )

        monitor_alerts_clear = monitor_alerts_subparsers.add_parser(
            "clear", help="Clear alerts for an agent"
        )
        monitor_alerts_clear.add_argument(
            "agent_id", help="Agent ID to clear alerts for"
        )

        # Deploy command
        deploy_parser = subparsers.add_parser(
            "deploy", help="Deployment automation and cloud management"
        )
        deploy_subparsers = deploy_parser.add_subparsers(dest="deploy_command")

        # Deploy agent
        deploy_agent = deploy_subparsers.add_parser(
            "agent", help="Deploy agent to cloud platform"
        )
        deploy_agent.add_argument(
            "platform", choices=["aws", "azure", "gcp"], help="Cloud platform"
        )
        deploy_agent.add_argument(
            "--config", "-c", help="Agent configuration file (JSON)"
        )
        deploy_agent.add_argument(
            "--env",
            "-e",
            default="dev",
            choices=["dev", "staging", "prod"],
            help="Deployment environment",
        )
        deploy_agent.add_argument(
            "--strategy",
            "-s",
            default="immediate",
            choices=["immediate", "rolling_update", "blue_green", "canary"],
            help="Deployment strategy",
        )
        deploy_agent.add_argument("--name", help="Agent name (overrides config)")
        deploy_agent.add_argument("--image", help="Container image (overrides config)")

        # Deploy status
        deploy_status = deploy_subparsers.add_parser(
            "status", help="Get deployment status"
        )
        deploy_status.add_argument("deployment_id", help="Deployment ID to check")

        # Deploy list
        deploy_list = deploy_subparsers.add_parser("list", help="List all deployments")

        # Deploy scale
        deploy_scale = deploy_subparsers.add_parser("scale", help="Scale deployment")
        deploy_scale.add_argument("deployment_id", help="Deployment ID to scale")
        deploy_scale.add_argument("replicas", type=int, help="Number of replicas")

        # Deploy rollback
        deploy_rollback = deploy_subparsers.add_parser(
            "rollback", help="Rollback deployment"
        )
        deploy_rollback.add_argument("deployment_id", help="Deployment ID to rollback")

        # Deploy templates
        deploy_templates = deploy_subparsers.add_parser(
            "templates", help="Generate infrastructure templates"
        )

        # Deploy scaling
        deploy_scaling = deploy_subparsers.add_parser(
            "scaling", help="Manage auto-scaling policies"
        )
        deploy_scaling_subparsers = deploy_scaling.add_subparsers(
            dest="scaling_command"
        )

        deploy_scaling_create = deploy_scaling_subparsers.add_parser(
            "create", help="Create scaling policy"
        )
        deploy_scaling_create.add_argument("name", help="Policy name")
        deploy_scaling_create.add_argument(
            "metric", help="Metric name (cpu, memory, requests)"
        )
        deploy_scaling_create.add_argument("target", type=float, help="Target value")
        deploy_scaling_create.add_argument(
            "--min", type=int, default=1, help="Minimum instances"
        )
        deploy_scaling_create.add_argument(
            "--max", type=int, default=10, help="Maximum instances"
        )

        # Email reporting command
        email_parser = subparsers.add_parser(
            "email", help="Email reporting configuration and management"
        )
        email_subparsers = email_parser.add_subparsers(dest="email_command")

        email_setup = email_subparsers.add_parser(
            "setup", help="Setup email configuration interactively"
        )
        email_setup.add_argument(
            "--config-file", help="Configuration file to save settings"
        )

        email_config = email_subparsers.add_parser(
            "config", help="Manage email configuration"
        )
        email_config_subparsers = email_config.add_subparsers(dest="config_command")

        email_config_show = email_config_subparsers.add_parser(
            "show", help="Show current email configuration"
        )

        email_config_add = email_config_subparsers.add_parser(
            "add-recipient", help="Add email recipient"
        )
        email_config_add.add_argument("email", help="Email address to add")

        email_config_remove = email_config_subparsers.add_parser(
            "remove-recipient", help="Remove email recipient"
        )
        email_config_remove.add_argument("email", help="Email address to remove")

        email_test = email_subparsers.add_parser(
            "test", help="Send test email to verify configuration"
        )

        email_benefits = email_subparsers.add_parser(
            "benefits", help="Send UCUP benefits summary via email"
        )
        email_benefits.add_argument("--message", help="Custom message to include")

        # Plugin command - NEW
        plugin_parser = subparsers.add_parser(
            "plugin", help="Manage and execute custom AI agent plugins"
        )
        plugin_subparsers = plugin_parser.add_subparsers(dest="plugin_command")

        plugin_list = plugin_subparsers.add_parser(
            "list", help="List available plugins"
        )
        plugin_list.add_argument(
            "--category",
            choices=["ai", "agent", "model", "tool", "all"],
            default="all",
            help="Filter by plugin category",
        )

        plugin_install = plugin_subparsers.add_parser(
            "install", help="Install a plugin from file or URL"
        )
        plugin_install.add_argument("source", help="Plugin file path or URL")
        plugin_install.add_argument(
            "--name", help="Custom plugin name (optional)"
        )

        plugin_uninstall = plugin_subparsers.add_parser(
            "uninstall", help="Uninstall a plugin"
        )
        plugin_uninstall.add_argument("name", help="Plugin name to uninstall")

        plugin_execute = plugin_subparsers.add_parser(
            "execute", help="Execute a plugin with input data"
        )
        plugin_execute.add_argument("name", help="Plugin name to execute")
        plugin_execute.add_argument(
            "--input", "-i", help="Input data file or JSON string"
        )
        plugin_execute.add_argument(
            "--output", "-o", help="Output file for results"
        )
        plugin_execute.add_argument(
            "--config", "-c", help="Plugin configuration file"
        )

        plugin_info = plugin_subparsers.add_parser(
            "info", help="Show detailed information about a plugin"
        )
        plugin_info.add_argument("name", help="Plugin name")

        plugin_create = plugin_subparsers.add_parser(
            "create", help="Create a new plugin template"
        )
        plugin_create.add_argument("name", help="Plugin name")
        plugin_create.add_argument(
            "--type",
            choices=["ai_agent", "model_adapter", "tool", "workflow"],
            default="ai_agent",
            help="Plugin type",
        )
        plugin_create.add_argument(
            "--output", "-o", help="Output directory for plugin files"
        )

        # Smart recommendations command
        recommend_parser = subparsers.add_parser(
            "recommend",
            help="Smart recommendations for UCUP components and architecture",
        )
        recommend_subparsers = recommend_parser.add_subparsers(dest="recommend_command")

        recommend_project = recommend_subparsers.add_parser(
            "project", help="Get recommendations for current or described project"
        )
        recommend_project.add_argument(
            "--description",
            "-d",
            help="Project description (if not analyzing current directory)",
        )
        recommend_project.add_argument(
            "--output",
            "-o",
            choices=["text", "json"],
            default="text",
            help="Output format",
        )
        recommend_project.add_argument(
            "--detailed", action="store_true", help="Show detailed recommendations"
        )

        recommend_learn = recommend_subparsers.add_parser(
            "learn", help="View learning data and insights"
        )
        recommend_learn.add_argument(
            "--show-builds", type=int, default=5, help="Show last N successful builds"
        )
        recommend_learn.add_argument(
            "--show-patterns", action="store_true", help="Show learned project patterns"
        )

        # Feature flags command
        features_parser = subparsers.add_parser(
            "features", help="Feature flag management"
        )
        features_subparsers = features_parser.add_subparsers(dest="features_command")

        features_list = features_subparsers.add_parser(
            "list", help="List all feature flags"
        )
        features_list.add_argument(
            "--all", action="store_true", help="Show all flags including disabled"
        )

        features_enable = features_subparsers.add_parser(
            "enable", help="Enable a feature flag"
        )
        features_enable.add_argument("flag", help="Feature flag name to enable")

        features_disable = features_subparsers.add_parser(
            "disable", help="Disable a feature flag"
        )
        features_disable.add_argument("flag", help="Feature flag name to disable")

        features_status = features_subparsers.add_parser(
            "status", help="Show status of a feature flag"
        )
        features_status.add_argument("flag", help="Feature flag name")

        # Version
        parser.add_argument(
            "--version", action="version", version=f"UCUP {self.get_version()}"
        )

        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return

        # Execute command
        try:
            getattr(self, f"cmd_{args.command}")(args)
            # Show donation prompt for relevant commands (with frequency control)
            if args.command in ["toon", "validate", "config", "features"]:
                if should_show_donation_prompt("cli"):
                    self._show_general_donation_prompt()
                    record_donation_prompt_shown("cli")
        except AttributeError:
            print(f"Unknown command: {args.command}")
            parser.print_help()
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    def cmd_toon(self, args):
        """Handle TOON subcommands."""
        if not hasattr(args, "toon_command") or not args.toon_command:
            print("Use 'ucup toon --help' for TOON commands")
            return

        method_name = f"cmd_toon_{args.toon_command}"
        if hasattr(self, method_name):
            getattr(self, method_name)(args)
        else:
            print(f"Unknown TOON command: {args.toon_command}")

    def cmd_toon_convert(self, args):
        """Convert data to TOON format."""
        # Load input data
        data = self._load_input_data(args.input)

        # Convert using formatter
        result = self.toon_formatter.format_with_choice(
            data,
            preferred_format=args.format,
            schema_name=args.schema,
            show_comparison=args.compare,
        )

        # Output result
        output = (
            result.toon_output if result.format_choice == "toon" else result.json_output
        )

        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Converted data saved to {args.output}")
        else:
            print(output)

        # Show metrics if requested
        if args.compare:
            self._show_conversion_metrics(result)

    def cmd_toon_compare(self, args):
        """Compare JSON vs TOON token usage."""
        data = self._load_input_data(args.input)

        result = self.toon_formatter.json_to_toon(data, schema_name=args.schema)

        print("=== TOKEN USAGE COMPARISON ===")
        print(f"JSON Tokens: {result.metrics.json_tokens}")
        print(f"TOON Tokens: {result.metrics.toon_tokens}")
        print(f"Savings: {result.metrics.savings_percentage:.1f}%")
        print(f"Compression Ratio: {result.metrics.compression_ratio:.2f}x")
        print(f"Est. Cost Savings: ${result.metrics.estimated_cost_savings:.4f}")

        if args.detailed:
            print("\n=== RECOMMENDATIONS ===")
            for rec in result.recommendations:
                print(f"• {rec}")

            print(f"\nConversion Time: {result.conversion_time:.3f}s")

    def cmd_toon_schema(self, args):
        """Handle schema subcommands."""
        if not hasattr(args, "schema_command") or not args.schema_command:
            print("Use 'ucup toon schema --help' for schema commands")
            return

        method_name = f"cmd_toon_schema_{args.schema_command}"
        if hasattr(self, method_name):
            getattr(self, method_name)(args)

    def cmd_toon_schema_create(self, args):
        """Create a new TOON schema."""
        data = self._load_input_data(args.sample_file)

        schema = self.toon_formatter.create_custom_schema(
            args.name, data, args.description or f"Custom schema for {args.name}"
        )

        print(f"✅ Created schema '{args.name}'")
        print(f"Fields: {len(schema.fields)}")
        print(f"Array Fields: {len(schema.array_fields)}")
        print(f"Optimizations: {', '.join(schema.optimizations)}")

    def cmd_toon_schema_list(self, args):
        """List available TOON schemas."""
        print("=== AVAILABLE TOON SCHEMAS ===")
        for name, schema in self.toon_formatter.schemas.items():
            print(f"\n📋 {name}")
            print(f"   Description: {schema.description}")
            print(f"   Fields: {len(schema.fields)}")
            print(
                f"   Optimizations: {', '.join(schema.optimizations) if schema.optimizations else 'None'}"
            )

    def cmd_toon_report(self, args):
        """Show token savings report."""
        report = self.toon_formatter.get_token_savings_report(args.limit)

        if "message" in report:
            print(report["message"])
            return

        print("=== TOKEN SAVINGS REPORT ===")
        print(f"Recent Conversions: {report['total_conversions']}")
        print(f"Average Savings: {report['average_savings_percentage']:.1f}%")
        print(f"Total Token Savings: {report['total_token_savings']}")

        if report["best_conversion"]["schema_used"]:
            print(f"Best Schema: {report['best_conversion']['schema_used']}")

        print("\n💡 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"• {rec}")

        # Add reward developers prompt for significant savings
        if report["average_savings_percentage"] > 30:
            print("\n" + "=" * 50)
            print("🎉 Impressive savings! Reward UCUP developers:")
            print("   💳 PayPal: ucup donate show --platform paypal")
            print("   🐙 GitHub Sponsors: ucup donate show --platform github")
            print("   ☕ Ko-fi: ucup donate show --platform ko-fi")
            print("   🏛️ Open Collective: ucup donate show --platform opencollective")
            print("   📋 All options: ucup donate show")
            print("=" * 50)

    def cmd_validate(self, args):
        """Validate data with UCUP."""
        data = self._load_input_data(args.input)

        try:
            report = validate_data(data, schema_name=args.schema)

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(report.to_dict(), f, indent=2)
                print(f"Validation report saved to {args.output}")
            else:
                print("=== VALIDATION REPORT ===")
                print(f"Status: {'✅ PASSED' if report.is_valid else '❌ FAILED'}")
                if report.errors:
                    print("Errors:")
                    for error in report.errors:
                        print(f"  • {error}")
                if report.warnings:
                    print("Warnings:")
                    for warning in report.warnings:
                        print(f"  • {warning}")

        except Exception as e:
            print(f"Validation error: {e}", file=sys.stderr)
            sys.exit(1)

    def cmd_config(self, args):
        """Handle configuration commands."""
        if args.show:
            print("=== UCUP CONFIGURATION ===")
            print(json.dumps(self.config, indent=2))
        elif args.init:
            print("Configuration initialization not yet implemented")
        else:
            print("Use 'ucup config --help' for config commands")

    def cmd_donate(self, args):
        """Handle donation subcommands."""
        if not hasattr(args, "donate_command") or not args.donate_command:
            print("Use 'ucup donate --help' for donation commands")
            return

        # Initialize donation plugin
        try:
            from .plugins.donation_plugin import DonationPlugin

            donation_plugin = DonationPlugin()
            donation_plugin.initialize()
        except Exception as e:
            print(f"Failed to initialize donation plugin: {e}")
            return

        method_name = f"cmd_donate_{args.donate_command}"
        if hasattr(self, method_name):
            getattr(self, method_name)(args, donation_plugin)
        else:
            print(f"Unknown donation command: {args.donate_command}")

    def cmd_donate_show(self, args, donation_plugin):
        """Show reward developers page."""
        page = donation_plugin.execute(
            "show_donate_page", platform=getattr(args, "platform", None)
        )
        print(page)

    def cmd_donate_stats(self, args, donation_plugin):
        """Show reward developers statistics."""
        stats = donation_plugin.execute("get_donation_stats")
        print("=== REWARD DEVELOPERS STATISTICS ===")
        print(f"Total Reward Developers: {stats['total_donations']}")
        print(f"Total Amount: ${stats['total_amount']:.2f}")
        print(f"Monthly Reward Developers: {stats['monthly_donations']}")
        print(f"Monthly Amount: ${stats['monthly_amount']:.2f}")
        print(f"Recent Reward Developers (30 days): {stats['recent_donations']}")

        if stats["top_platforms"]:
            print("\nTop Platforms:")
            for platform in stats["top_platforms"]:
                print(
                    f"  {platform['platform']}: ${platform['amount']:.2f} ({platform['count']} reward developers)"
                )

    def cmd_donate_link(self, args, donation_plugin):
        """Create reward developers link."""
        try:
            link = donation_plugin.execute(
                "create_donation_link",
                platform=args.platform,
                amount=getattr(args, "amount", None),
                currency=getattr(args, "currency", "USD"),
            )
            print(f"Reward developers link: {link}")
        except Exception as e:
            print(f"Error creating reward developers link: {e}")

    def cmd_donate_sponsor(self, args, donation_plugin):
        """Show sponsorship information."""
        sponsorship_info = donation_plugin.execute("show_sponsorship_call")
        print(sponsorship_info)

    def cmd_features(self, args):
        """Handle feature flag subcommands."""
        if not hasattr(args, "features_command") or not args.features_command:
            print("Use 'ucup features --help' for feature flag commands")
            return

        method_name = f"cmd_features_{args.features_command}"
        if hasattr(self, method_name):
            getattr(self, method_name)(args)
        else:
            print(f"Unknown features command: {args.features_command}")

    def cmd_features_list(self, args):
        """List all feature flags."""
        manager = get_feature_manager()
        flags = manager.list_flags()

        print("=== UCUP FEATURE FLAGS ===")
        print(
            f"Global Features Enabled: {'Yes' if manager.config.global_enabled else 'No'}"
        )
        print()

        enabled_count = 0
        for name, info in flags.items():
            if info["enabled"] or args.all:
                status_icon = "✅" if info["enabled"] else "❌"
                state_display = f"{info['state'].upper()}"
                if info.get("warning"):
                    state_display += f" ⚠️ {info['warning']}"

                print(f"{status_icon} {name}")
                print(f"   State: {state_display}")
                print(f"   Description: {info['description']}")
                if info["dependencies"]:
                    print(f"   Dependencies: {', '.join(info['dependencies'])}")
                if info.get("requires_restart"):
                    print("   ⚠️  Requires restart to take effect")
                print()

                if info["enabled"]:
                    enabled_count += 1

        print(f"Total Flags: {len(flags)} | Enabled: {enabled_count}")

    def cmd_features_enable(self, args):
        """Enable a feature flag."""
        manager = get_feature_manager()
        if manager.enable_flag(args.flag):
            print(f"✅ Enabled feature flag: {args.flag}")
            config = manager.get_flag_config(args.flag)
            if config and config.warning:
                print(f"⚠️  Warning: {config.warning}")
            if config and config.requires_restart:
                print("🔄 Restart required for changes to take effect")
        else:
            print(f"❌ Failed to enable feature flag: {args.flag}")

    def cmd_features_disable(self, args):
        """Disable a feature flag."""
        manager = get_feature_manager()
        if manager.disable_flag(args.flag):
            print(f"✅ Disabled feature flag: {args.flag}")
            config = manager.get_flag_config(args.flag)
            if config and config.requires_restart:
                print("🔄 Restart required for changes to take effect")
        else:
            print(f"❌ Failed to disable feature flag: {args.flag}")

    def cmd_features_status(self, args):
        """Show status of a specific feature flag."""
        manager = get_feature_manager()
        config = manager.get_flag_config(args.flag)

        if not config:
            print(f"❌ Feature flag not found: {args.flag}")
            return

        enabled = manager.is_enabled(args.flag)
        status_icon = "✅" if enabled else "❌"

        print(f"=== FEATURE FLAG: {args.flag} ===")
        print(f"Status: {status_icon} {'Enabled' if enabled else 'Disabled'}")
        print(f"State: {config.state.value.upper()}")
        print(f"Description: {config.description}")

        if config.dependencies:
            print(f"Dependencies: {', '.join(config.dependencies)}")

        if config.warning:
            print(f"Warning: {config.warning}")

        if config.requires_restart:
            print("Requires Restart: Yes")

    def cmd_recommend(self, args):
        """Handle smart recommendations subcommands."""
        if not hasattr(args, "recommend_command") or not args.recommend_command:
            print("Use 'ucup recommend --help' for recommendation commands")
            return

        method_name = f"cmd_recommend_{args.recommend_command}"
        if hasattr(self, method_name):
            getattr(self, method_name)(args)
        else:
            print(f"Unknown recommend command: {args.recommend_command}")

    def cmd_recommend_project(self, args):
        """Get smart recommendations for a project."""
        try:
            from .smart_recommendations import SmartRecommendationEngine

            # Initialize recommendation engine
            engine = SmartRecommendationEngine()

            # Get project description
            project_description = args.description
            if not project_description:
                # Analyze current directory
                import os
                from pathlib import Path

                project_path = Path.cwd()

                # Try to infer description from project files
                try:
                    if (project_path / "pyproject.toml").exists():
                        import tomllib

                        with open(project_path / "pyproject.toml", "rb") as f:
                            data = tomllib.load(f)
                            if "project" in data and "description" in data["project"]:
                                project_description = data["project"]["description"]
                            if "project" in data and "name" in data["project"]:
                                project_description = f"Project: {data['project']['name']}. {project_description or ''}"

                    if (project_path / "README.md").exists():
                        with open(project_path / "README.md", "r") as f:
                            content = f.read()[:300]
                            first_para = (
                                content.split("\n\n")[0]
                                if "\n\n" in content
                                else content[:100]
                            )
                            project_description = (
                                f"{project_description or ''} {first_para}".strip()
                            )

                    # Look for agent files
                    agent_files = list(project_path.glob("**/*agent*.py"))
                    if agent_files:
                        project_description = f"{project_description or ''} AI agent development project".strip()

                except Exception as e:
                    print(f"⚠️  Could not analyze current directory: {e}")
                    print("Please provide a project description with --description")
                    return

            if not project_description:
                print("❌ No project description available")
                print("Either run in a project directory or provide --description")
                return

            print("🧠 Analyzing project and generating recommendations...")
            print(
                f"📝 Project: {project_description[:100]}{'...' if len(project_description) > 100 else ''}"
            )

            # Analyze project and generate recommendations
            profile = engine.analyze_project_requirements(project_description)
            recommendations = engine.generate_recommendations(profile)

            # Output format
            if args.output == "json":
                output = {
                    "project_profile": {
                        "type": profile.project_type,
                        "domain": profile.domain,
                        "complexity": profile.complexity,
                        "scale_requirements": profile.scale_requirements,
                        "performance_needs": profile.performance_needs,
                        "reliability_requirements": profile.reliability_requirements,
                        "multimodal_needs": profile.multimodal_needs,
                    },
                    "recommendations": {
                        "primary_agent": {
                            "component_name": recommendations.primary_agent.component_name,
                            "confidence_score": recommendations.primary_agent.confidence_score,
                            "reasoning": recommendations.primary_agent.reasoning,
                        },
                        "supporting_components": [
                            {
                                "component_name": comp.component_name,
                                "reasoning": comp.reasoning,
                            }
                            for comp in recommendations.supporting_components
                        ],
                        "coordination_strategy": {
                            "component_name": recommendations.coordination_strategy.component_name,
                            "reasoning": recommendations.coordination_strategy.reasoning,
                        },
                        "deployment_config": recommendations.deployment_config,
                        "estimated_benefits": recommendations.estimated_benefits,
                        "overall_confidence": recommendations.confidence_score,
                    },
                }
                print(json.dumps(output, indent=2))

            else:  # text format
                print("\n" + "=" * 70)
                print("🧠 UCUP SMART PROJECT RECOMMENDATIONS")
                print("=" * 70)

                print(f"📋 Project Profile:")
                print(f"   Type: {profile.project_type.title()}")
                print(f"   Domain: {profile.domain.title()}")
                print(f"   Complexity: {profile.complexity.title()}")
                print(f"   Scale: {profile.scale_requirements.title()}")
                print(f"   Performance: {profile.performance_needs.title()}")
                print(f"   Reliability: {profile.reliability_requirements.title()}")
                if profile.multimodal_needs:
                    print(f"   Multimodal: {', '.join(profile.multimodal_needs)}")

                print(f"\n🎯 PRIMARY AGENT:")
                print(f"   {recommendations.primary_agent.component_name}")
                print(
                    f"   Confidence: {recommendations.primary_agent.confidence_score:.1%}"
                )
                print(f"   Reasoning: {recommendations.primary_agent.reasoning}")

                if recommendations.supporting_components:
                    print(f"\n🔧 SUPPORTING COMPONENTS:")
                    for comp in recommendations.supporting_components:
                        print(f"   • {comp.component_name}: {comp.reasoning}")

                if recommendations.coordination_strategy:
                    print(f"\n🤝 COORDINATION STRATEGY:")
                    print(f"   {recommendations.coordination_strategy.component_name}")
                    print(f"   {recommendations.coordination_strategy.reasoning}")

                print(f"\n☁️  DEPLOYMENT RECOMMENDATION:")
                deploy = recommendations.deployment_config
                print(
                    f"   Target: {deploy['target']} ({deploy['performance_profile']} performance)"
                )
                print(f"   Scalability: {deploy['scalability']}")

                print(f"\n📊 ESTIMATED BENEFITS:")
                benefits = recommendations.estimated_benefits
                for benefit, value in benefits.items():
                    benefit_name = benefit.replace("_", " ").title()
                    print(f"   {benefit_name}: {value:.1%}")

                print(f"\n🎯 OVERALL CONFIDENCE: {recommendations.confidence_score:.1%}")

                if args.detailed:
                    print(f"\n📈 DETAILED ANALYSIS:")

                    # Show component alternatives
                    if recommendations.primary_agent.alternatives:
                        print(
                            f"   Agent Alternatives: {', '.join(recommendations.primary_agent.alternatives)}"
                        )

                    # Show configuration suggestions
                    if recommendations.primary_agent.configuration_suggestions:
                        print(f"   Agent Configuration:")
                        for (
                            key,
                            value,
                        ) in (
                            recommendations.primary_agent.configuration_suggestions.items()
                        ):
                            print(f"     {key}: {value}")

                    # Show benefits
                    if recommendations.primary_agent.benefits:
                        print(
                            f"   Agent Benefits: {', '.join(recommendations.primary_agent.benefits)}"
                        )

                print(
                    "\n💡 TIP: UCUP learns from successful builds - your next build will be even smarter!"
                )
                print("=" * 70)

        except ImportError:
            print("❌ Smart recommendations not available")
            print("Install with: pip install ucup[recommendations]")
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")

    def cmd_plugin(self, args):
        """Handle plugin subcommands."""
        if not hasattr(args, "plugin_command") or not args.plugin_command:
            print("Use 'ucup plugin --help' for plugin commands")
            return

        method_name = f"cmd_plugin_{args.plugin_command}"
        if hasattr(self, method_name):
            getattr(self, method_name)(args)
        else:
            print(f"Unknown plugin command: {args.plugin_command}")

    def cmd_plugin_list(self, args):
        """List available plugins."""
        try:
            # Import plugin system
            from .plugins import PluginManager

            manager = PluginManager()
            plugins = manager.list_plugins(category=args.category if args.category != "all" else None)

            if plugins:
                print("=== AVAILABLE PLUGINS ===")
                categories = {}

                # Group by category
                for plugin in plugins:
                    cat = plugin.get("category", "misc")
                    if cat not in categories:
                        categories[cat] = []
                    categories[cat].append(plugin)

                for category, plugin_list in categories.items():
                    print(f"\n📂 {category.upper()} PLUGINS:")
                    for plugin in plugin_list:
                        status_icon = "✅" if plugin.get("enabled", False) else "❌"
                        print(f"   {status_icon} {plugin['name']}")
                        print(f"      {plugin.get('description', 'No description')}")
                        if plugin.get("version"):
                            print(f"      Version: {plugin['version']}")
                        print()
            else:
                print("📋 No plugins found")
                print("Install plugins with: ucup plugin install <source>")
                print("Create a plugin with: ucup plugin create <name>")

        except ImportError:
            print("❌ Plugin system not available")
        except Exception as e:
            print(f"❌ Error listing plugins: {e}")

    def cmd_plugin_install(self, args):
        """Install a plugin from file or URL."""
        try:
            from .plugins import PluginManager

            manager = PluginManager()

            print(f"📦 Installing plugin from: {args.source}")

            # Determine if it's a URL or local file
            if args.source.startswith(("http://", "https://", "ftp://")):
                # URL installation
                success = manager.install_from_url(args.source, name=args.name)
            else:
                # Local file installation
                if not os.path.exists(args.source):
                    print(f"❌ Plugin source not found: {args.source}")
                    return

                success = manager.install_from_file(args.source, name=args.name)

            if success:
                print("✅ Plugin installed successfully")
                print("Use 'ucup plugin list' to see installed plugins")
                print("Use 'ucup plugin execute <name>' to run the plugin")
            else:
                print("❌ Failed to install plugin")

        except ImportError:
            print("❌ Plugin system not available")
        except Exception as e:
            print(f"❌ Error installing plugin: {e}")

    def cmd_plugin_uninstall(self, args):
        """Uninstall a plugin."""
        try:
            from .plugins import PluginManager

            manager = PluginManager()

            print(f"🗑️  Uninstalling plugin: {args.name}")

            success = manager.uninstall_plugin(args.name)

            if success:
                print(f"✅ Plugin '{args.name}' uninstalled successfully")
            else:
                print(f"❌ Failed to uninstall plugin '{args.name}'")
                print("Use 'ucup plugin list' to see available plugins")

        except ImportError:
            print("❌ Plugin system not available")
        except Exception as e:
            print(f"❌ Error uninstalling plugin: {e}")

    def cmd_plugin_execute(self, args):
        """Execute a plugin with input data."""
        try:
            from .plugins import PluginManager

            manager = PluginManager()

            # Load input data
            input_data = None
            if args.input:
                if os.path.exists(args.input):
                    # Load from file
                    with open(args.input, "r") as f:
                        if args.input.endswith(".json"):
                            input_data = json.load(f)
                        else:
                            input_data = f.read()
                else:
                    # Try to parse as JSON
                    try:
                        input_data = json.loads(args.input)
                    except json.JSONDecodeError:
                        input_data = args.input

            # Load config if provided
            config_data = None
            if args.config:
                if os.path.exists(args.config):
                    with open(args.config, "r") as f:
                        config_data = json.load(f)

            print(f"🚀 Executing plugin: {args.name}")

            # Execute plugin
            result = manager.execute_plugin(args.name, input_data=input_data, config=config_data)

            if result:
                print("✅ Plugin executed successfully")

                # Output result
                if args.output:
                    with open(args.output, "w") as f:
                        if isinstance(result, dict):
                            json.dump(result, f, indent=2)
                        else:
                            f.write(str(result))
                    print(f"📄 Results saved to: {args.output}")
                else:
                    # Print to console
                    if isinstance(result, dict):
                        print(json.dumps(result, indent=2))
                    else:
                        print(result)
            else:
                print("❌ Plugin execution failed or returned no result")

        except ImportError:
            print("❌ Plugin system not available")
        except Exception as e:
            print(f"❌ Error executing plugin: {e}")

    def cmd_plugin_info(self, args):
        """Show detailed information about a plugin."""
        try:
            from .plugins import PluginManager

            manager = PluginManager()

            plugin_info = manager.get_plugin_info(args.name)

            if plugin_info:
                print(f"=== PLUGIN: {args.name} ===")
                print(f"Name: {plugin_info.get('name', 'Unknown')}")
                print(f"Version: {plugin_info.get('version', 'Unknown')}")
                print(f"Category: {plugin_info.get('category', 'Unknown')}")
                print(f"Enabled: {'Yes' if plugin_info.get('enabled', False) else 'No'}")
                print(f"Description: {plugin_info.get('description', 'No description')}")
                print(f"Author: {plugin_info.get('author', 'Unknown')}")
                print(f"License: {plugin_info.get('license', 'Unknown')}")

                if plugin_info.get("dependencies"):
                    print(f"Dependencies: {', '.join(plugin_info['dependencies'])}")

                if plugin_info.get("capabilities"):
                    print(f"Capabilities: {', '.join(plugin_info['capabilities'])}")

                if plugin_info.get("supported_inputs"):
                    print(f"Supported Inputs: {', '.join(plugin_info['supported_inputs'])}")

                if plugin_info.get("output_formats"):
                    print(f"Output Formats: {', '.join(plugin_info['output_formats'])}")

                print(f"\nInstallation Path: {plugin_info.get('path', 'Unknown')}")
                print(f"Last Modified: {plugin_info.get('modified', 'Unknown')}")

                if plugin_info.get("readme"):
                    print(f"\n📖 README:")
                    print(plugin_info["readme"][:500] + "..." if len(plugin_info["readme"]) > 500 else plugin_info["readme"])

            else:
                print(f"❌ Plugin '{args.name}' not found")
                print("Use 'ucup plugin list' to see available plugins")

        except ImportError:
            print("❌ Plugin system not available")
        except Exception as e:
            print(f"❌ Error getting plugin info: {e}")

    def cmd_plugin_create(self, args):
        """Create a new plugin template."""
        try:
            from .plugins import PluginManager

            manager = PluginManager()

            output_dir = args.output or f"./{args.name}_plugin"
            template_path = manager.create_plugin_template(args.name, args.type, output_dir)

            if template_path:
                print("✅ Plugin template created successfully")
                print(f"📁 Location: {template_path}")
                print("\n📋 Next steps:")
                print(f"   1. Edit the plugin files in {template_path}")
                print("   2. Implement your AI agent logic")
                print(f"   3. Test the plugin: ucup plugin execute {args.name}")
                print(f"   4. Install permanently: ucup plugin install {template_path}")

                # Show basic structure
                print(f"\n📂 Plugin Structure:")
                for root, dirs, files in os.walk(template_path):
                    level = root.replace(template_path, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        print(f"{subindent}{file}")
            else:
                print("❌ Failed to create plugin template")

        except ImportError:
            print("❌ Plugin system not available")
        except Exception as e:
            print(f"❌ Error creating plugin template: {e}")

    def cmd_recommend_learn(self, args):
        """View learning data and insights."""
        try:
            from .smart_recommendations import SmartRecommendationEngine

            engine = SmartRecommendationEngine()

            if args.show_patterns:
                print("🧠 UCUP LEARNING PATTERNS")
                print("=" * 50)

                patterns = engine.learning_data.get("project_patterns", {})
                if patterns:
                    print(f"Learned Patterns: {len(patterns)}")
                    print()

                    for pattern_key, builds in patterns.items():
                        domain, complexity, scale = pattern_key.split("_", 2)
                        print(
                            f"📋 {domain.title()} | {complexity.title()} | {scale.title()}"
                        )
                        print(f"   Successful Builds: {len(builds)}")

                        # Show most common agent for this pattern
                        agents = {}
                        for build in builds:
                            agent = build.get("primary_agent", "Unknown")
                            agents[agent] = agents.get(agent, 0) + build.get(
                                "success_score", 0
                            )

                        if agents:
                            best_agent = max(agents.items(), key=lambda x: x[1])
                            print(
                                f"   Best Agent: {best_agent[0]} (avg score: {best_agent[1]/len(builds):.2f})"
                            )
                        print()
                else:
                    print("No learning patterns available yet.")
                    print("Complete some builds to start learning!")

            if args.show_builds > 0:
                print("\n📚 RECENT SUCCESSFUL BUILDS")
                print("=" * 50)

                builds = engine.learning_data.get("successful_builds", [])
                recent_builds = sorted(
                    builds, key=lambda x: x.get("timestamp", ""), reverse=True
                )[: args.show_builds]

                if recent_builds:
                    for i, build in enumerate(recent_builds, 1):
                        profile = build.get("project_profile", {})
                        recommendations = build.get("recommendations", {})
                        success_score = build.get("success_score", 0)

                        print(
                            f"{i}. {profile.get('domain', 'Unknown').title()} Project"
                        )
                        print(
                            f"   Agent: {recommendations.get('primary_agent', {}).get('component_name', 'Unknown')}"
                        )
                        print(f"   Complexity: {profile.get('complexity', 'Unknown')}")
                        print(f"   Success Score: {success_score:.2f}")
                        print(f"   Date: {build.get('timestamp', 'Unknown')[:10]}")
                        print()
                else:
                    print("No successful builds recorded yet.")
                    print("Complete a build to start the learning process!")

            # Show component effectiveness
            effectiveness = engine.learning_data.get("component_effectiveness", {})
            if effectiveness:
                print("\n🏆 COMPONENT EFFECTIVENESS")
                print("=" * 50)

                sorted_components = sorted(
                    effectiveness.items(), key=lambda x: x[1], reverse=True
                )
                for component, score in sorted_components[:10]:  # Top 10
                    print(".2f")
                print()

        except ImportError:
            print("❌ Smart recommendations not available")
        except Exception as e:
            print(f"❌ Error accessing learning data: {e}")

    def cmd_test(self, args):
        """Handle testing subcommands."""
        if not hasattr(args, "test_command") or not args.test_command:
            print("Use 'ucup test --help' for testing commands")
            return

        method_name = f"cmd_test_{args.test_command}"
        if hasattr(self, method_name):
            getattr(self, method_name)(args)
        else:
            print(f"Unknown test command: {args.test_command}")

    def cmd_test_run(self, args):
        """Run comprehensive tests."""
        try:
            from .test_environments import (
                TestEnvironment,
                TestEnvironmentManager,
                TestSuite,
                run_ucup_tests,
            )

            print("🚀 Starting UCUP comprehensive testing...")

            if args.env:
                # Run in specific environment
                result = run_ucup_tests(env_name=args.env)
                self._display_test_result(result)
            else:
                # Run comprehensive multi-environment tests
                manager = TestEnvironmentManager()

                # Determine test suites to run
                suites_to_run = []
                if args.suite == "all":
                    suites_to_run = [
                        "unit",
                        "integration",
                        "performance",
                        "reliability",
                    ]
                else:
                    suites_to_run = [args.suite]

                results = []
                for suite_name in suites_to_run:
                    print(f"\n📋 Running {suite_name} test suite...")

                    # Create appropriate test suite
                    if suite_name == "unit":
                        suite = TestSuite(
                            name="cli_unit_suite",
                            description="CLI unit test suite",
                            test_files=[
                                "tests/ucup-tests/test_agentgpt_model_validation.py"
                            ],
                            environment_requirements=TestEnvironment(
                                name="cli_test", python_version="3.11"
                            ),
                            parallel_execution=args.parallel,
                            generate_coverage=args.coverage,
                            generate_html_report=args.html_report,
                        )
                    elif suite_name == "integration":
                        suite = TestSuite(
                            name="cli_integration_suite",
                            description="CLI integration test suite",
                            test_files=[
                                "tests/test_advanced_probabilistic.py",
                                "tests/test_toon.py",
                            ],
                            environment_requirements=TestEnvironment(
                                name="cli_test", python_version="3.11"
                            ),
                            parallel_execution=args.parallel,
                            generate_coverage=args.coverage,
                            generate_html_report=args.html_report,
                        )
                    elif suite_name == "performance":
                        suite = TestSuite(
                            name="cli_performance_suite",
                            description="CLI performance test suite",
                            test_files=["tests/benchmark_*.py"],
                            environment_requirements=TestEnvironment(
                                name="cli_test", python_version="3.11"
                            ),
                            parallel_execution=False,  # Performance tests usually sequential
                            generate_coverage=False,
                            generate_html_report=args.html_report,
                        )
                    elif suite_name == "reliability":
                        suite = TestSuite(
                            name="cli_reliability_suite",
                            description="CLI reliability test suite",
                            test_files=[
                                "tests/stress_*.py",
                                "tests/test_self_healing.py",
                            ],
                            environment_requirements=TestEnvironment(
                                name="cli_test", python_version="3.11"
                            ),
                            parallel_execution=args.parallel,
                            generate_coverage=args.coverage,
                            generate_html_report=args.html_report,
                        )

                    # Run the suite
                    result = run_ucup_tests()
                    results.append(result)
                    self._display_test_result(result)

                # Summary
                total_passed = sum(r.total_tests - r.failed_tests for r in results if r)
                total_failed = sum(r.failed_tests for r in results if r)

                print(f"\n📊 Test Summary: {total_passed} passed, {total_failed} failed")

                # Prepare test results data for output/email
                output_data = {
                    "summary": {
                        "total_passed": total_passed,
                        "total_failed": total_failed,
                        "suites_run": len(results),
                    },
                    "results": [r.__dict__ if r else None for r in results],
                    "timestamp": datetime.now().isoformat(),
                }

                if args.output:
                    import json

                    with open(args.output, "w") as f:
                        json.dump(output_data, f, indent=2, default=str)
                    print(f"📄 Detailed results saved to {args.output}")

                # Send email report if requested
                if args.email:
                    try:
                        from .email_reporting import EmailReporter, load_email_config

                        config = load_email_config()
                        if config.is_configured():
                            reporter = EmailReporter(config)
                            subject = (
                                args.email_subject
                                or f"UCUP Test Report: {total_passed} passed, {total_failed} failed"
                            )

                            print("📧 Sending test report via email...")
                            success = reporter.send_test_report(
                                output_data, subject=subject
                            )

                            if success:
                                print(
                                    f"✅ Test report sent to {len(config.recipient_emails)} recipient(s)"
                                )
                            else:
                                print("❌ Failed to send test report via email")
                        else:
                            print(
                                "❌ Email not configured. Run 'ucup email setup' first"
                            )

                    except ImportError:
                        print("❌ Email reporting not available")
                    except Exception as e:
                        print(f"❌ Error sending email report: {e}")

        except ImportError as e:
            print(f"Testing framework not available: {e}")
            print("Try installing with: pip install ucup[test]")
        except Exception as e:
            print(f"Test execution failed: {e}")

    def cmd_test_env(self, args):
        """Handle test environment subcommands."""
        if not hasattr(args, "env_command") or not args.env_command:
            print("Use 'ucup test env --help' for environment commands")
            return

        method_name = f"cmd_test_env_{args.env_command}"
        if hasattr(self, method_name):
            getattr(self, method_name)(args)

    def cmd_test_env_create(self, args):
        """Create a test environment."""
        try:
            import asyncio

            from .test_environments import TestEnvironment, TestEnvironmentManager

            async def create_env():
                manager = TestEnvironmentManager()
                env = TestEnvironment(
                    name=args.name,
                    python_version=args.python,
                    conda_packages=["pytest", "pytest-cov"] if args.conda else None,
                    pip_packages=["ucup", "pytest", "pytest-cov"]
                    if not args.conda
                    else ["ucup"],
                )

                results = await manager.setup_test_environments([env])
                return results.get(args.name, False)

            success = asyncio.run(create_env())

            if success:
                print(f"✅ Created test environment: {args.name}")
                print(f"   Python version: {args.python}")
                print(f"   Type: {'conda' if args.conda else 'venv'}")
            else:
                print(f"❌ Failed to create test environment: {args.name}")

        except Exception as e:
            print(f"Error creating environment: {e}")

    def cmd_test_env_list(self, args):
        """List available test environments."""
        try:
            import asyncio

            from .test_environments import TestEnvironmentManager

            async def list_envs():
                manager = TestEnvironmentManager()
                conda_envs = await manager.conda_manager.list_environments()
                venv_envs = await manager.venv_manager.list_environments()
                return conda_envs, venv_envs

            conda_envs, venv_envs = asyncio.run(list_envs())

            print("=== TEST ENVIRONMENTS ===")

            if conda_envs:
                print("\n🐍 Conda Environments:")
                for env in conda_envs:
                    print(f"   • {env}")

            if venv_envs:
                print("\n📦 Virtual Environments:")
                for env in venv_envs:
                    print(f"   • {env}")

            if not conda_envs and not venv_envs:
                print("No test environments found.")
                print("Create one with: ucup test env create <name>")

        except Exception as e:
            print(f"Error listing environments: {e}")

    def cmd_test_env_destroy(self, args):
        """Destroy a test environment."""
        try:
            import asyncio

            from .test_environments import TestEnvironmentManager

            async def destroy_env():
                manager = TestEnvironmentManager()

                # Try conda first
                success = await manager.conda_manager.destroy_environment(args.name)
                if not success:
                    # Try venv
                    success = await manager.venv_manager.destroy_environment(args.name)

                return success

            success = asyncio.run(destroy_env())

            if success:
                print(f"✅ Destroyed test environment: {args.name}")
            else:
                print(f"❌ Failed to destroy test environment: {args.name}")

        except Exception as e:
            print(f"Error destroying environment: {e}")

    def cmd_test_scaffold(self, args):
        """Generate test scaffolding for agents."""
        try:
            from .test_environments import generate_test_template

            output_dir = args.output or "tests"
            test_file = generate_test_template(args.agent_name, args.domain, output_dir)

            print(f"✅ Generated test scaffolding for {args.agent_name}")
            print(f"   Domain: {args.domain}")
            print(f"   File: {test_file}")
            print("\nNext steps:")
            print(f"   1. Review and customize the generated tests in {test_file}")
            print("   2. Run tests with: pytest {test_file}")
            print("   3. Add more test cases as needed")

        except Exception as e:
            print(f"Error generating test scaffold: {e}")

    def cmd_test_report(self, args):
        """Generate test reports and analytics."""
        try:
            import json
            from pathlib import Path

            # Load test results
            input_path = Path(args.input) if args.input else None

            if not input_path or not input_path.exists():
                print("❌ Test results file not found")
                print("Run tests first with: ucup test run --output results.json")
                return

            with open(input_path, "r") as f:
                results_data = json.load(f)

            # Generate report based on format
            if args.format == "json":
                # Just output the data as-is
                output_data = results_data

            elif args.format == "markdown":
                output_data = self._generate_markdown_report(results_data)

            elif args.format == "html":
                output_data = self._generate_html_report(results_data)

            # Output report
            if args.output:
                with open(args.output, "w") as f:
                    if args.format == "json":
                        json.dump(output_data, f, indent=2)
                    else:
                        f.write(output_data)
                print(f"📄 Report saved to {args.output}")
            else:
                if args.format == "json":
                    print(json.dumps(output_data, indent=2))
                else:
                    print(output_data)

        except Exception as e:
            print(f"Error generating report: {e}")

    def _display_test_result(self, result):
        """Display a test result."""
        if not result:
            print("❌ No test result available")
            return

        status_icon = "✅" if result.success else "❌"
        print(f"{status_icon} {result.test_suite}")
        print(
            f"   Tests: {result.total_tests} total, {result.passed_tests} passed, {result.failed_tests} failed"
        )
        if result.coverage_percentage:
            print(f"   Coverage: {result.coverage_percentage:.1f}%")
        print(f"   Duration: {result.execution_time_seconds:.2f}s")
        if result.errors:
            print(f"   Errors: {len(result.errors)}")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"     • {error[:100]}...")

    def _generate_markdown_report(self, results_data):
        """Generate a markdown test report."""
        summary = results_data.get("summary", {})

        report = f"""# UCUP Test Report

## Summary

- **Total Passed**: {summary.get('total_passed', 0)}
- **Total Failed**: {summary.get('total_failed', 0)}
- **Test Suites**: {summary.get('suites_run', 0)}
- **Generated**: {results_data.get('timestamp', 'Unknown')}

## Detailed Results

"""

        for i, result in enumerate(results_data.get("results", [])):
            if result:
                report += f"""### Test Suite {i+1}: {result.get('test_suite', 'Unknown')}

- **Status**: {'✅ Passed' if result.get('success') else '❌ Failed'}
- **Total Tests**: {result.get('total_tests', 0)}
- **Passed**: {result.get('passed_tests', 0)}
- **Failed**: {result.get('failed_tests', 0)}
- **Coverage**: {result.get('coverage_percentage', 0):.1f}%
- **Duration**: {result.get('execution_time_seconds', 0):.2f}s

"""

                if result.get("errors"):
                    report += "**Errors:**\n"
                    for error in result["errors"][:5]:  # Show first 5 errors
                        report += f"- {error}\n"
                    report += "\n"

        return report

    def _generate_html_report(self, results_data):
        """Generate an HTML test report."""
        summary = results_data.get("summary", {})

        html = (
            ".1f"
            ".1f"
            f"""<!DOCTYPE html>
<html>
<head>
    <title>UCUP Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .suite {{ border: 1px solid #ddd; padding: 15px; margin-bottom: 10px; border-radius: 5px; }}
        .passed {{ border-left: 5px solid #4CAF50; }}
        .failed {{ border-left: 5px solid #f44336; }}
        h1, h2 {{ color: #333; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>UCUP Test Report</h1>

    <div class="summary">
        <h2>Summary</h2>
        <div class="metric">Total Passed: <strong>{summary.get('total_passed', 0)}</strong></div>
        <div class="metric">Total Failed: <strong>{summary.get('total_failed', 0)}</strong></div>
        <div class="metric">Test Suites: <strong>{summary.get('suites_run', 0)}</strong></div>
        <div class="metric">Generated: <strong>{results_data.get('timestamp', 'Unknown')}</strong></div>
    </div>

    <h2>Detailed Results</h2>
"""
        )

        for i, result in enumerate(results_data.get("results", [])):
            if result:
                status_class = "passed" if result.get("success") else "failed"
                status_text = "✅ Passed" if result.get("success") else "❌ Failed"

                html += (
                    ".1f"
                    ".1f"
                    f"""
    <div class="suite {status_class}">
        <h3>Test Suite {i+1}: {result.get('test_suite', 'Unknown')}</h3>
        <p><strong>Status:</strong> {status_text}</p>
        <p><strong>Total Tests:</strong> {result.get('total_tests', 0)}</p>
        <p><strong>Passed:</strong> {result.get('passed_tests', 0)}</p>
        <p><strong>Failed:</strong> {result.get('failed_tests', 0)}</p>
        <p><strong>Coverage:</strong> {result.get('coverage_percentage', 0):.1f}%</p>
        <p><strong>Duration:</strong> {result.get('execution_time_seconds', 0):.2f}s</p>
"""
                )

                if result.get("errors"):
                    html += "<h4>Errors:</h4><ul>"
                    for error in result["errors"][:5]:
                        html += f"<li>{error}</li>"
                    html += "</ul>"

                html += "    </div>"

        html += """
</body>
</html>
"""

        return html

    def cmd_monitor(self, args):
        """Handle monitoring subcommands."""
        if not hasattr(args, "monitor_command") or not args.monitor_command:
            print("Use 'ucup monitor --help' for monitoring commands")
            return

        method_name = f"cmd_monitor_{args.monitor_command}"
        if hasattr(self, method_name):
            getattr(self, method_name)(args)
        else:
            print(f"Unknown monitor command: {args.monitor_command}")

    def cmd_monitor_status(self, args):
        """Show monitoring status and agent health."""
        try:
            from .agent_monitoring import get_agent_health_report, get_all_agent_reports

            if args.agent_id:
                # Show specific agent
                report = get_agent_health_report(args.agent_id)
                if report:
                    self._display_agent_report(report, args.detailed)
                else:
                    print(f"❌ No monitoring data found for agent: {args.agent_id}")
            else:
                # Show all agents
                reports = get_all_agent_reports()
                if reports:
                    print("=== AGENT HEALTH OVERVIEW ===")
                    for agent_id, report in reports.items():
                        self._display_agent_report_summary(report)
                else:
                    print("📊 No agents are currently being monitored")
                    print("Start monitoring with: ucup monitor start")

        except ImportError:
            print("❌ Agent monitoring not available")
            print("Install with: pip install ucup[monitoring]")
        except Exception as e:
            print(f"❌ Error retrieving monitoring status: {e}")

    def cmd_monitor_start(self, args):
        """Start agent monitoring."""
        try:
            import asyncio

            from .agent_monitoring import start_agent_monitoring

            print(f"🚀 Starting agent monitoring (interval: {args.interval}s)...")

            # Start monitoring in background
            asyncio.create_task(start_agent_monitoring(args.interval))

            print("✅ Agent monitoring started")
            print("Use 'ucup monitor status' to check agent health")
            print("Use 'ucup monitor stop' to stop monitoring")

        except ImportError:
            print("❌ Agent monitoring not available")
            print("Install with: pip install ucup[monitoring]")
        except Exception as e:
            print(f"❌ Error starting monitoring: {e}")

    def cmd_monitor_stop(self, args):
        """Stop agent monitoring."""
        try:
            import asyncio

            from .agent_monitoring import stop_agent_monitoring

            print("🛑 Stopping agent monitoring...")
            asyncio.run(stop_agent_monitoring())
            print("✅ Agent monitoring stopped")

        except ImportError:
            print("❌ Agent monitoring not available")
        except Exception as e:
            print(f"❌ Error stopping monitoring: {e}")

    def cmd_monitor_metrics(self, args):
        """Record agent metrics."""
        try:
            import asyncio

            from .agent_monitoring import record_agent_metrics

            # Build metrics data
            metrics_data = {}

            if args.response_time is not None:
                metrics_data["response_time_avg"] = args.response_time
            if args.error_rate is not None:
                metrics_data["error_rate"] = args.error_rate
            if args.throughput is not None:
                metrics_data["throughput"] = args.throughput
            if args.memory is not None:
                metrics_data["memory_usage"] = args.memory
            if args.cpu is not None:
                metrics_data["cpu_usage"] = args.cpu

            if args.custom:
                try:
                    custom_data = json.loads(args.custom)
                    metrics_data.update(custom_data)
                except json.JSONDecodeError:
                    print("❌ Invalid JSON for custom metrics")
                    return

            if not metrics_data:
                print("❌ No metrics provided")
                print("Use --response-time, --error-rate, etc. to specify metrics")
                return

            # Record metrics
            asyncio.run(record_agent_metrics(args.agent_id, metrics_data))

            print(f"✅ Recorded metrics for agent: {args.agent_id}")
            print(f"   Metrics: {list(metrics_data.keys())}")

        except ImportError:
            print("❌ Agent monitoring not available")
        except Exception as e:
            print(f"❌ Error recording metrics: {e}")

    def cmd_monitor_validate(self, args):
        """Validate agent behavior."""
        try:
            import asyncio

            from .agent_monitoring import validate_agent

            # Load agent data
            if args.data:
                if os.path.isfile(args.data):
                    with open(args.data, "r") as f:
                        agent_data = json.load(f)
                else:
                    try:
                        agent_data = json.loads(args.data)
                    except json.JSONDecodeError:
                        print("❌ Invalid JSON data")
                        return
            else:
                # Default validation data structure
                agent_data = {
                    "metrics": {},
                    "behavior": {
                        "recent_responses": [],
                        "confidence_scores": [],
                        "uncertainty_scores": [],
                    },
                }

            # Validate agent
            issues = asyncio.run(validate_agent(args.agent_id, agent_data))

            if issues:
                print(
                    f"⚠️  Found {len(issues)} validation issues for agent: {args.agent_id}"
                )
                for issue in issues:
                    severity_icon = {
                        "info": "ℹ️",
                        "warning": "⚠️",
                        "error": "❌",
                        "critical": "🚨",
                    }.get(issue.severity.value, "❓")

                    print(f"   {severity_icon} {issue.message}")
                    if issue.details:
                        print(f"      Details: {issue.details}")
            else:
                print(f"✅ Agent {args.agent_id} passed validation")

            if args.output:
                output_data = {
                    "agent_id": args.agent_id,
                    "validation_time": str(datetime.now()),
                    "issues": [issue.__dict__ for issue in issues],
                }
                with open(args.output, "w") as f:
                    json.dump(output_data, f, indent=2, default=str)
                print(f"📄 Validation report saved to {args.output}")

        except ImportError:
            print("❌ Agent monitoring not available")
        except Exception as e:
            print(f"❌ Error validating agent: {e}")

    def cmd_monitor_alerts(self, args):
        """Handle monitoring alerts subcommands."""
        if not hasattr(args, "alerts_command") or not args.alerts_command:
            print("Use 'ucup monitor alerts --help' for alerts commands")
            return

        method_name = f"cmd_monitor_alerts_{args.alerts_command}"
        if hasattr(self, method_name):
            getattr(self, method_name)(args)

    def cmd_monitor_alerts_list(self, args):
        """List recent alerts."""
        try:
            from .agent_monitoring import get_agent_health_report, get_all_agent_reports

            if args.agent_id:
                # Show alerts for specific agent
                report = get_agent_health_report(args.agent_id)
                if report and report.issues:
                    print(f"=== ALERTS FOR AGENT: {args.agent_id} ===")
                    self._display_agent_issues(report.issues, args.limit)
                else:
                    print(f"ℹ️  No alerts found for agent: {args.agent_id}")
            else:
                # Show alerts for all agents
                reports = get_all_agent_reports()
                alerts_found = False

                for agent_id, report in reports.items():
                    if report.issues:
                        if not alerts_found:
                            print("=== ALL AGENT ALERTS ===")
                            alerts_found = True

                        print(f"\nAgent: {agent_id}")
                        self._display_agent_issues(report.issues, args.limit)

                if not alerts_found:
                    print("✅ No alerts found for any monitored agents")

        except ImportError:
            print("❌ Agent monitoring not available")
        except Exception as e:
            print(f"❌ Error retrieving alerts: {e}")

    def cmd_monitor_alerts_clear(self, args):
        """Clear alerts for an agent."""
        try:
            from .agent_monitoring import get_global_monitor

            monitor = get_global_monitor()

            # Clear issues for the agent
            if args.agent_id in monitor.agent_issues:
                cleared_count = len(monitor.agent_issues[args.agent_id])
                monitor.agent_issues[args.agent_id].clear()
                monitor._save_metrics()  # Persist changes

                print(f"✅ Cleared {cleared_count} alerts for agent: {args.agent_id}")
            else:
                print(f"ℹ️  No alerts found for agent: {args.agent_id}")

        except ImportError:
            print("❌ Agent monitoring not available")
        except Exception as e:
            print(f"❌ Error clearing alerts: {e}")

    def _display_agent_report(self, report, detailed=False):
        """Display a detailed agent health report."""
        health_icons = {
            "healthy": "🟢",
            "degraded": "🟡",
            "unhealthy": "🟠",
            "critical": "🔴",
            "unknown": "⚪",
        }

        print(f"=== AGENT HEALTH REPORT: {report.agent_id} ===")
        print(
            f"Status: {health_icons.get(report.health_status.value, '❓')} {report.health_status.value.upper()}"
        )
        print(f"Health Score: {report.overall_score:.1f}/100")

        print(f"\n📊 METRICS:")
        metrics = report.metrics
        print(f"   Response Time: {metrics.response_time_avg:.2f}s")
        print(f"   Error Rate: {metrics.error_rate:.1%}")
        print(f"   Throughput: {metrics.throughput:.2f} req/s")
        print(f"   Memory Usage: {metrics.memory_usage:.0f}MB")
        print(f"   CPU Usage: {metrics.cpu_usage:.1f}%")

        if detailed and metrics.custom_metrics:
            print("   Custom Metrics:")
            for key, value in metrics.custom_metrics.items():
                print(f"     {key}: {value}")

        if report.issues:
            print(f"\n⚠️  ISSUES ({len(report.issues)}):")
            self._display_agent_issues(report.issues)

        if report.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report.recommendations:
                print(f"   • {rec}")

        print(f"\n🕒 Last Updated: {report.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        uptime_hours = report.uptime_seconds / 3600
        print(f"⏱️  Uptime: {uptime_hours:.1f} hours")

    def _display_agent_report_summary(self, report):
        """Display a summary of agent health."""
        health_icons = {
            "healthy": "🟢",
            "degraded": "🟡",
            "unhealthy": "🟠",
            "critical": "🔴",
            "unknown": "⚪",
        }

        status_icon = health_icons.get(report.health_status.value, "❓")
        issue_count = len(report.issues)

        print(
            f"{status_icon} {report.agent_id}: {report.health_status.value} "
            f"(Score: {report.overall_score:.0f}, Issues: {issue_count})"
        )

    def _display_agent_issues(self, issues, limit=None):
        """Display agent validation issues."""
        displayed_issues = issues[:limit] if limit else issues

        severity_icons = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}

        for issue in displayed_issues:
            icon = severity_icons.get(issue.severity.value, "❓")
            resolved_text = " (RESOLVED)" if issue.resolved else ""
            print(f"   {icon} {issue.message}{resolved_text}")

            if issue.details:
                for key, value in issue.details.items():
                    print(f"      {key}: {value}")

        if limit and len(issues) > limit:
            print(f"   ... and {len(issues) - limit} more issues")

    def cmd_deploy(self, args):
        """Handle deployment subcommands."""
        if not hasattr(args, "deploy_command") or not args.deploy_command:
            print("Use 'ucup deploy --help' for deployment commands")
            return

        method_name = f"cmd_deploy_{args.deploy_command}"
        if hasattr(self, method_name):
            getattr(self, method_name)(args)
        else:
            print(f"Unknown deploy command: {args.deploy_command}")

    def cmd_deploy_agent(self, args):
        """Deploy agent to cloud platform."""
        try:
            from .deployment_automation import deploy_agent_to_cloud

            # Load agent configuration
            agent_config = {}
            if args.config:
                if os.path.isfile(args.config):
                    with open(args.config, "r") as f:
                        agent_config = json.load(f)
                else:
                    print("❌ Agent config file not found")
                    return
            else:
                # Create basic config
                agent_config = {
                    "name": args.name or "ucup-agent",
                    "image": args.image or "ucup/agent:latest",
                    "deployment": {"env_vars": {}, "ports": [8000], "replicas": 1},
                }

            # Override config with CLI args
            if args.name:
                agent_config["name"] = args.name
            if args.image:
                agent_config["image"] = args.image

            print(f"🚀 Deploying agent to {args.platform}...")
            print(f"   Environment: {args.env}")
            print(f"   Strategy: {args.strategy}")

            # Deploy agent
            deployment_id = deploy_agent_to_cloud(
                args.platform, agent_config, args.env, args.strategy
            )

            if deployment_id:
                print(f"\n📋 Deployment ID: {deployment_id}")
                print("Monitor deployment with: ucup deploy status {deployment_id}")
            else:
                print("❌ Deployment failed")

        except ImportError:
            print("❌ Deployment automation not available")
            print("Install with: pip install ucup[deployment]")
        except Exception as e:
            print(f"❌ Error deploying agent: {e}")

    def cmd_deploy_status(self, args):
        """Get deployment status."""
        try:
            from .deployment_automation import get_deployment_info

            status = get_deployment_info(args.deployment_id)

            if status:
                print(f"=== DEPLOYMENT STATUS: {args.deployment_id} ===")
                print(f"Pipeline: {status.get('pipeline', 'Unknown')}")
                print(f"Environment: {status.get('environment', 'Unknown')}")
                print(f"Strategy: {status.get('strategy', 'Unknown')}")
                print(f"Status: {status.get('status', 'Unknown')}")

                cloud_status = status.get("cloud_status", {})
                if cloud_status:
                    print(f"\n☁️  Cloud Status:")
                    for key, value in cloud_status.items():
                        print(f"   {key}: {value}")

                start_time = status.get("start_time")
                end_time = status.get("end_time")
                if start_time:
                    print(f"\n🕒 Started: {start_time}")
                if end_time:
                    print(f"🕒 Ended: {end_time}")

                if status.get("rollback_available"):
                    print("🔄 Rollback available")
            else:
                print(f"❌ Deployment not found: {args.deployment_id}")

        except ImportError:
            print("❌ Deployment automation not available")
        except Exception as e:
            print(f"❌ Error getting deployment status: {e}")

    def cmd_deploy_list(self, args):
        """List all deployments."""
        try:
            from .deployment_automation import get_deployment_cli

            cli = get_deployment_cli()
            deployments = cli.list_deployments()

            if deployments:
                print("=== DEPLOYMENTS ===")
                for i, deployment in enumerate(deployments, 1):
                    status = deployment.get("status", "unknown")
                    status_icon = {
                        "success": "✅",
                        "failed": "❌",
                        "in_progress": "🔄",
                        "pending": "⏳",
                        "rolling_back": "🔙",
                    }.get(status, "❓")

                    print(
                        f"{i}. {status_icon} {deployment.get('deployment_id', 'Unknown')}"
                    )
                    print(f"   Pipeline: {deployment.get('pipeline', 'Unknown')}")
                    print(f"   Environment: {deployment.get('environment', 'Unknown')}")
                    print(f"   Status: {status}")
                    print(f"   Started: {deployment.get('start_time', 'Unknown')}")
                    print()
            else:
                print("📋 No deployments found")

        except ImportError:
            print("❌ Deployment automation not available")
        except Exception as e:
            print(f"❌ Error listing deployments: {e}")

    def cmd_deploy_scale(self, args):
        """Scale deployment."""
        try:
            from .deployment_automation import scale_agent_deployment

            print(
                f"🔄 Scaling deployment {args.deployment_id} to {args.replicas} replicas..."
            )

            success = scale_agent_deployment(args.deployment_id, args.replicas)

            if success:
                print("✅ Deployment scaled successfully")
            else:
                print("❌ Failed to scale deployment")

        except ImportError:
            print("❌ Deployment automation not available")
        except Exception as e:
            print(f"❌ Error scaling deployment: {e}")

    def cmd_deploy_rollback(self, args):
        """Rollback deployment."""
        try:
            from .deployment_automation import rollback_agent_deployment

            print(f"🔙 Rolling back deployment {args.deployment_id}...")

            success = rollback_agent_deployment(args.deployment_id)

            if success:
                print("✅ Deployment rolled back successfully")
            else:
                print("❌ Failed to rollback deployment")

        except ImportError:
            print("❌ Deployment automation not available")
        except Exception as e:
            print(f"❌ Error rolling back deployment: {e}")

    def cmd_deploy_templates(self, args):
        """Generate infrastructure templates."""
        try:
            from .deployment_automation import get_deployment_cli

            cli = get_deployment_cli()
            cli.generate_infrastructure_templates()

        except ImportError:
            print("❌ Deployment automation not available")
        except Exception as e:
            print(f"❌ Error generating templates: {e}")

    def cmd_deploy_scaling(self, args):
        """Handle scaling policy subcommands."""
        if not hasattr(args, "scaling_command") or not args.scaling_command:
            print("Use 'ucup deploy scaling --help' for scaling commands")
            return

        method_name = f"cmd_deploy_scaling_{args.scaling_command}"
        if hasattr(self, method_name):
            getattr(self, method_name)(args)

    def cmd_deploy_scaling_create(self, args):
        """Create scaling policy."""
        try:
            from .deployment_automation import get_deployment_cli

            cli = get_deployment_cli()
            cli.create_scaling_policy(
                args.name, args.metric, args.target, args.min, args.max
            )

        except ImportError:
            print("❌ Deployment automation not available")
        except Exception as e:
            print(f"❌ Error creating scaling policy: {e}")

    def cmd_email(self, args):
        """Handle email reporting subcommands."""
        if not hasattr(args, "email_command") or not args.email_command:
            print("Use 'ucup email --help' for email commands")
            return

        method_name = f"cmd_email_{args.email_command}"
        if hasattr(self, method_name):
            getattr(self, method_name)(args)
        else:
            print(f"Unknown email command: {args.email_command}")

    def cmd_email_setup(self, args):
        """Setup email configuration interactively."""
        try:
            from .email_reporting import (
                save_email_config,
                setup_email_config_interactive,
            )

            config = setup_email_config_interactive()

            if config.is_configured():
                config_path = args.config_file or os.path.expanduser(
                    "~/.ucup/email_config.json"
                )
                if save_email_config(config, config_path):
                    print(f"✅ Email configuration saved to {config_path}")
                    print(f"   Recipients: {len(config.recipient_emails)}")
                    print(f"   SMTP Server: {config.smtp_server}:{config.smtp_port}")
                else:
                    print("❌ Failed to save email configuration")
            else:
                print("❌ Email configuration incomplete")

        except ImportError:
            print("❌ Email reporting not available")
        except Exception as e:
            print(f"❌ Error setting up email: {e}")

    def cmd_email_config(self, args):
        """Handle email configuration subcommands."""
        if not hasattr(args, "config_command") or not args.config_command:
            print("Use 'ucup email config --help' for config commands")
            return

        method_name = f"cmd_email_config_{args.config_command}"
        if hasattr(self, method_name):
            getattr(self, method_name)(args)

    def cmd_email_config_show(self, args):
        """Show current email configuration."""
        try:
            from .email_reporting import load_email_config

            config = load_email_config()
            if config.is_configured():
                print("=== EMAIL CONFIGURATION ===")
                print(f"SMTP Server: {config.smtp_server}:{config.smtp_port}")
                print(f"Sender Email: {config.sender_email}")
                print(f"TLS Enabled: {config.use_tls}")
                print(f"Recipients ({len(config.recipient_emails)}):")
                for email in config.recipient_emails:
                    print(f"  • {email}")
            else:
                print("❌ Email configuration not found or incomplete")
                print("Run 'ucup email setup' to configure email reporting")

        except ImportError:
            print("❌ Email reporting not available")
        except Exception as e:
            print(f"❌ Error showing email config: {e}")

    def cmd_email_config_add_recipient(self, args):
        """Add email recipient."""
        try:
            from .email_reporting import load_email_config, save_email_config

            config = load_email_config()
            config.add_recipient(args.email)

            config_path = os.path.expanduser("~/.ucup/email_config.json")
            if save_email_config(config, config_path):
                print(f"✅ Added recipient: {args.email}")
            else:
                print("❌ Failed to save configuration")

        except ImportError:
            print("❌ Email reporting not available")
        except Exception as e:
            print(f"❌ Error adding recipient: {e}")

    def cmd_email_config_remove_recipient(self, args):
        """Remove email recipient."""
        try:
            from .email_reporting import load_email_config, save_email_config

            config = load_email_config()
            config.remove_recipient(args.email)

            config_path = os.path.expanduser("~/.ucup/email_config.json")
            if save_email_config(config, config_path):
                print(f"✅ Removed recipient: {args.email}")
            else:
                print("❌ Failed to save configuration")

        except ImportError:
            print("❌ Email reporting not available")
        except Exception as e:
            print(f"❌ Error removing recipient: {e}")

    def cmd_email_test(self, args):
        """Send test email to verify configuration."""
        try:
            from .email_reporting import EmailReporter, load_email_config

            config = load_email_config()
            if not config.is_configured():
                print("❌ Email configuration incomplete")
                print("Run 'ucup email setup' first")
                return

            reporter = EmailReporter(config)

            # Create test data
            test_results = {
                "summary": {"total_passed": 5, "total_failed": 0, "suites_run": 1},
                "results": [
                    {
                        "test_suite": "Email Test Suite",
                        "success": True,
                        "total_tests": 5,
                        "passed_tests": 5,
                        "failed_tests": 0,
                        "coverage_percentage": 95.0,
                        "execution_time_seconds": 1.2,
                    }
                ],
                "timestamp": datetime.now().isoformat(),
            }

            print("📧 Sending test email...")
            success = reporter.send_test_report(test_results, subject="UCUP Email Test")

            if success:
                print("✅ Test email sent successfully!")
                print(f"   Sent to {len(config.recipient_emails)} recipient(s)")
            else:
                print("❌ Failed to send test email")

        except ImportError:
            print("❌ Email reporting not available")
        except Exception as e:
            print(f"❌ Error sending test email: {e}")

    def cmd_email_benefits(self, args):
        """Send UCUP benefits summary via email."""
        try:
            from .email_reporting import EmailReporter, load_email_config

            config = load_email_config()
            if not config.is_configured():
                print("❌ Email configuration incomplete")
                print("Run 'ucup email setup' first")
                return

            reporter = EmailReporter(config)

            custom_message = getattr(args, "message", None)
            if custom_message:
                custom_message = (
                    f"<p><strong>Custom Message:</strong> {custom_message}</p>"
                )

            print("📧 Sending UCUP benefits summary...")
            success = reporter.send_benefits_summary(custom_message)

            if success:
                print("✅ Benefits summary sent successfully!")
                print(f"   Sent to {len(config.recipient_emails)} recipient(s)")
            else:
                print("❌ Failed to send benefits summary")

        except ImportError:
            print("❌ Email reporting not available")
        except Exception as e:
            print(f"❌ Error sending benefits summary: {e}")

    def _load_input_data(self, input_path: str) -> Any:
        """Load data from file or stdin."""
        if input_path == "-":
            return json.load(sys.stdin)
        else:
            with open(input_path, "r") as f:
                return json.load(f)

    def _show_conversion_metrics(self, result: "ToonConversionResult"):
        """Display conversion metrics."""
        print("\n=== CONVERSION METRICS ===", file=sys.stderr)
        print(f"Format Chosen: {result.format_choice.upper()}", file=sys.stderr)
        print(f"JSON Tokens: {result.metrics.json_tokens}", file=sys.stderr)
        print(f"TOON Tokens: {result.metrics.toon_tokens}", file=sys.stderr)
        print(f"Savings: {result.metrics.savings_percentage:.1f}%", file=sys.stderr)
        print(
            f"Cost Savings: ${result.metrics.estimated_cost_savings:.4f}",
            file=sys.stderr,
        )
        print(f"Conversion Time: {result.conversion_time:.3f}s", file=sys.stderr)

    def _show_general_donation_prompt(self):
        """Show donation prompt after general CLI usage."""
        print("\n" + "=" * 60)
        print("💝 Support UCUP Development")
        print("UCUP just helped enhance your development workflow!")
        print()
        print("Help us continue improving AI development tools:")
        print("• 💳 PayPal: ucup donate show --platform paypal")
        print("• 🐙 GitHub Sponsors: ucup donate show --platform github")
        print("• ☕ Ko-fi: ucup donate show --platform ko-fi")
        print("• 📋 All options: ucup donate show")
        print()

        # Interactive choice
        try:
            from .build_wrapper import _show_interactive_donation_choice

            _show_interactive_donation_choice("cli")
        except ImportError:
            # Fallback if import fails
            pass
        print("=" * 60)

    def get_version(self) -> str:
        """Get UCUP version."""
        try:
            from importlib.metadata import version

            return version("ucup")
        except:
            return "0.2.0"


def main():
    """CLI entry point."""
    cli = UCUPCLI()
    cli.run()


if __name__ == "__main__":
    main()
