#!/usr/bin/env python3
"""
UCUP Build Wrapper

Wraps common build commands to display UCUP benefits and smart recommendations.
Can be used as: ucup-build <command>
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import (
    get_global_tracker,
    record_build_metrics,
    record_donation_prompt_shown,
    should_show_donation_prompt,
    show_build_benefits,
)
from .smart_recommendations import SmartRecommendationEngine


def run_build_command(command: List[str]) -> int:
    """Run a build command and track metrics with smart recommendations"""
    print("\n🚀 Starting UCUP-enhanced build...\n")

    start_time = time.time()

    # Analyze project and provide smart recommendations before build
    project_analysis = _analyze_project_for_build()
    if project_analysis:
        _show_build_recommendations(project_analysis)

    try:
        # Run the actual build command
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        # Parse output for metrics (basic implementation)
        output = result.stdout
        duration = time.time() - start_time

        # Count errors and warnings in output
        lines = output.split("\n")
        errors = sum(1 for line in lines if "error" in line.lower())
        warnings = sum(1 for line in lines if "warning" in line.lower())

        # Print the output
        print(output)

        # Record metrics
        record_build_metrics(
            duration=duration,
            errors_detected=errors,
            warnings_detected=warnings,
            uncertainty_checks=1,
            validation_runs=1,
            tests_run=0,
            tests_passed=0,
            tests_failed=0,
        )

        # Show benefits
        show_build_benefits()

        # Save successful build patterns for learning
        if result.returncode == 0 and project_analysis:
            _save_build_success(
                project_analysis,
                {
                    "duration": duration,
                    "errors": errors,
                    "warnings": warnings,
                    "command": command,
                    "success": True,
                },
            )

        # Show donation prompt after build benefits (with frequency control)
        if should_show_donation_prompt("build"):
            _show_build_donation_prompt()
            record_donation_prompt_shown("build")

        return result.returncode

    except Exception as e:
        print(f"Error running build command: {e}", file=sys.stderr)
        return 1


def _show_build_donation_prompt():
    """Show interactive donation prompt after successful build."""
    print("\n" + "=" * 60)
    print("💝 Support UCUP Development")
    print("UCUP just helped make your build more reliable and efficient!")
    print()
    print("Help us continue improving AI development tools:")
    print("• 💳 PayPal: ucup donate show --platform paypal")
    print("• 🐙 GitHub Sponsors: ucup donate show --platform github")
    print("• ☕ Ko-fi: ucup donate show --platform ko-fi")
    print("• 📋 All options: ucup donate show")
    print()

    # Interactive choice
    _show_interactive_donation_choice("build")
    print("=" * 60)


def _show_interactive_donation_choice(workflow_type: str):
    """Show interactive donation choice menu."""
    # Only show interactive choices if running in interactive terminal
    if not os.isatty(0) or os.getenv("CI") or os.getenv("NON_INTERACTIVE"):
        return

    print("Choose an option:")
    print("1. 💝 Reward Developers (opens donation page)")
    print("2. ⏰ Ask me again")
    print()

    try:
        choice = input("Enter your choice (1-2): ").strip()

        if choice == "1":
            # Open donation page
            print("\n🚀 Opening donation page...")
            try:
                from .plugins.donation_plugin import DonationPlugin

                donation_plugin = DonationPlugin()
                donation_plugin.initialize()
                page = donation_plugin.execute("show_donate_page")
                print(page)
                # Try to open in browser if possible
                import webbrowser

                webbrowser.open("https://www.paypal.com/ncp/payment/PUMSTXGUWEZC8")
            except Exception as e:
                print(f"Could not open browser: {e}")
                print("Please visit: https://www.paypal.com/ncp/payment/PUMSTXGUWEZC8")

        elif choice == "2":
            # Ask me again - reset the last shown date to yesterday so it shows tomorrow
            tracker = get_global_tracker()
            if tracker.benefits.donation_history:
                from datetime import datetime, timedelta

                yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
                if workflow_type == "build":
                    tracker.benefits.donation_history.last_build_prompt = yesterday
                elif workflow_type == "test":
                    tracker.benefits.donation_history.last_test_prompt = yesterday
                elif workflow_type == "cli":
                    tracker.benefits.donation_history.last_cli_prompt = yesterday
                tracker._save_metrics()
            print("\n⏰ Will ask again tomorrow!")

        else:
            print("\n❓ Invalid choice. Please enter 1 or 2.")

    except (EOFError, KeyboardInterrupt):
        # Handle non-interactive environments gracefully
        print("\nContinuing without user input...")


def _analyze_project_for_build() -> Optional[Dict[str, Any]]:
    """Analyze the current project for smart recommendations during build."""
    try:
        # Get current working directory as project path
        project_path = Path.cwd()

        # Try to detect project type and requirements from files
        project_description = _infer_project_description(project_path)

        if not project_description:
            return None

        # Initialize smart recommendation engine
        engine = SmartRecommendationEngine()

        # Analyze project requirements
        profile = engine.analyze_project_requirements(project_description)

        # Generate recommendations
        recommendations = engine.generate_recommendations(profile)

        return {
            "profile": profile,
            "recommendations": recommendations,
            "project_path": str(project_path),
            "project_description": project_description,
        }

    except Exception as e:
        # Silently fail if analysis fails - don't break the build
        print(
            f"Note: Could not analyze project for recommendations: {e}", file=sys.stderr
        )
        return None


def _infer_project_description(project_path: Path) -> Optional[str]:
    """Infer project description from project files."""
    try:
        descriptions = []

        # Check for common project files
        if (project_path / "pyproject.toml").exists():
            try:
                import tomllib

                with open(project_path / "pyproject.toml", "rb") as f:
                    data = tomllib.load(f)
                    if "project" in data and "description" in data["project"]:
                        descriptions.append(data["project"]["description"])
                    if "project" in data and "name" in data["project"]:
                        descriptions.append(f"Project: {data['project']['name']}")
            except:
                pass

        if (project_path / "setup.py").exists():
            try:
                with open(project_path / "setup.py", "r") as f:
                    content = f.read()
                    # Look for description in setup.py
                    if "description=" in content:
                        # Simple extraction
                        start = content.find("description=") + 13
                        end = content.find(",", start)
                        if end > start:
                            desc = content[start:end].strip("\"'" '"')
                            descriptions.append(desc)
            except:
                pass

        # Check for README files
        for readme_name in ["README.md", "README.rst", "README.txt"]:
            readme_path = project_path / readme_name
            if readme_path.exists():
                try:
                    with open(readme_path, "r") as f:
                        content = f.read()[:500]  # First 500 chars
                        # Extract first meaningful paragraph
                        lines = content.split("\n")
                        for line in lines[:10]:  # First 10 lines
                            line = line.strip()
                            if line and not line.startswith("#") and len(line) > 20:
                                descriptions.append(line)
                                break
                    break
                except:
                    pass

        # Look for agent-related files to infer project type
        agent_files = list(project_path.glob("**/*agent*.py"))
        if agent_files:
            descriptions.append("AI agent development project")

        multimodal_files = list(project_path.glob("**/*vision*.py")) + list(
            project_path.glob("**/*audio*.py")
        )
        if multimodal_files:
            descriptions.append("multimodal AI project")

        # Combine descriptions
        if descriptions:
            return ". ".join(descriptions[:3])  # Limit to first 3 descriptions

        # Fallback based on file types
        python_files = list(project_path.glob("**/*.py"))
        if python_files:
            return "Python software project"

        return None

    except Exception as e:
        return None


def _show_build_recommendations(analysis: Dict[str, Any]):
    """Display smart recommendations during build."""
    try:
        recommendations = analysis["recommendations"]
        profile = analysis["profile"]

        print("\n" + "=" * 70)
        print("🧠 UCUP SMART RECOMMENDATIONS")
        print("=" * 70)

        print(
            f"📋 Project Profile: {profile.project_type.title()} | {profile.domain.title()} | {profile.complexity.title()} complexity"
        )

        if profile.multimodal_needs:
            print(f"🎯 Multimodal Needs: {', '.join(profile.multimodal_needs)}")

        print(f"\n🎯 RECOMMENDED PRIMARY AGENT:")
        print(f"   {recommendations.primary_agent.component_name}")
        print(f"   Confidence: {recommendations.primary_agent.confidence_score:.1%}")
        print(f"   Reasoning: {recommendations.primary_agent.reasoning}")

        if recommendations.supporting_components:
            print(f"\n🔧 RECOMMENDED SUPPORTING COMPONENTS:")
            for comp in recommendations.supporting_components:
                print(f"   • {comp.component_name}: {comp.reasoning}")

        if recommendations.coordination_strategy:
            print(f"\n🤝 RECOMMENDED COORDINATION:")
            print(f"   {recommendations.coordination_strategy.component_name}")
            print(f"   {recommendations.coordination_strategy.reasoning}")

        print(f"\n☁️  RECOMMENDED DEPLOYMENT:")
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

        print(
            "\n💡 TIP: UCUP learns from successful builds to improve future recommendations!"
        )
        print("=" * 70)

    except Exception as e:
        print(f"Note: Could not display recommendations: {e}", file=sys.stderr)


def _save_build_success(analysis: Dict[str, Any], build_metrics: Dict[str, Any]):
    """Save successful build patterns for learning."""
    try:
        engine = SmartRecommendationEngine()
        profile = analysis["profile"]
        recommendations = analysis["recommendations"]

        # Prepare build metrics for learning
        learning_metrics = {
            "build_duration": build_metrics["duration"],
            "build_errors": build_metrics["errors"],
            "build_warnings": build_metrics["warnings"],
            "build_command": " ".join(build_metrics["command"]),
            "test_pass_rate": 0.0,  # Could be enhanced with actual test results
            "performance_score": 0.8
            if build_metrics["duration"] < 60
            else 0.6,  # Simple heuristic
            "reliability_score": 0.9 if build_metrics["errors"] == 0 else 0.7,
            "project_path": analysis["project_path"],
            "project_description": analysis["project_description"],
        }

        # Save to learning data
        engine.save_build_success(profile, recommendations, learning_metrics)

        print(
            f"\n🧠 UCUP learned from this successful build! Future recommendations will be improved."
        )

    except Exception as e:
        # Silently fail - don't break the build
        print(f"Note: Could not save build learning data: {e}", file=sys.stderr)


def main():
    """Main entry point for ucup-build command"""
    if len(sys.argv) < 2:
        print("Usage: ucup-build <command> [args...]")
        print("\nExamples:")
        print("  ucup-build python setup.py build")
        print("  ucup-build pip install -e .")
        print("  ucup-build python -m build")
        sys.exit(1)

    command = sys.argv[1:]
    exit_code = run_build_command(command)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
